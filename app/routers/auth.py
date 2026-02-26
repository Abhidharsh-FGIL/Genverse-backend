import uuid
import secrets
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import DBSession, CurrentUser
from app.models.user import User, UserRole
from app.models.organization import Organization, OrgMember
from app.models.subscription import Subscription
from app.schemas.auth import (
    SignupRequest,
    OrgSignupRequest,
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    ChangePasswordRequest,
)
from app.schemas.user import UserResponse
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
)
from app.core.exceptions import CredentialsException, ConflictException

router = APIRouter()


async def _process_pending_enrollments(db: AsyncSession, user: User) -> None:
    """After a new user signs up, auto-enroll them in any classes they were pre-invited to."""
    from app.models.classes import PendingClassEnrollment, ClassStudent
    from app.models.organization import OrgMember as OrgMemberModel

    pending_result = await db.execute(
        select(PendingClassEnrollment).where(PendingClassEnrollment.email == user.email)
    )
    pending_enrollments = pending_result.scalars().all()

    for pending in pending_enrollments:
        # Check not already enrolled
        existing = await db.execute(
            select(ClassStudent).where(
                ClassStudent.class_id == pending.class_id,
                ClassStudent.student_id == user.id,
            )
        )
        if not existing.scalar_one_or_none():
            db.add(ClassStudent(
                class_id=pending.class_id,
                student_id=user.id,
                roll_no=pending.roll_no,
            ))

        # Ensure org membership so the student can see the org workspace
        if pending.org_id:
            existing_member = await db.execute(
                select(OrgMemberModel).where(
                    OrgMemberModel.org_id == pending.org_id,
                    OrgMemberModel.user_id == user.id,
                    OrgMemberModel.role == "student",
                )
            )
            org_member = existing_member.scalar_one_or_none()
            if org_member:
                org_member.status = "active"
            else:
                db.add(OrgMemberModel(
                    org_id=pending.org_id,
                    user_id=user.id,
                    role="student",
                    status="active",
                ))

        # Remove the pending record now that it's been processed
        await db.delete(pending)


async def _create_free_subscription(
    db: AsyncSession, user_id: uuid.UUID = None, org_id: uuid.UUID = None, workspace_type: str = "individual"
) -> Subscription:
    from datetime import timedelta
    trial_ends = datetime.now(timezone.utc) + timedelta(days=14)
    sub = Subscription(
        user_id=user_id,
        org_id=org_id,
        plan="free",
        status="trialing",
        workspace_type=workspace_type,
        points_balance=100,
        points_monthly_quota=100,
        storage_limit_mb=100,
        trial_ends_at=trial_ends,
    )
    db.add(sub)
    return sub


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, db: DBSession):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise ConflictException("Email already registered")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        name=payload.name,
    )
    db.add(user)
    await db.flush()

    role = UserRole(user_id=user.id, role=payload.role)
    db.add(role)

    await _create_free_subscription(db, user_id=user.id, workspace_type="individual")

    # Auto-enroll in any classes the user was pre-invited to before having an account
    await _process_pending_enrollments(db, user)

    await db.commit()
    await db.refresh(user)

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/signup/organization", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def org_signup(payload: OrgSignupRequest, db: DBSession):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise ConflictException("Email already registered")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        name=payload.admin_name,
    )
    db.add(user)
    await db.flush()

    role = UserRole(user_id=user.id, role="org_admin")
    db.add(role)

    org = Organization(
        name=payload.org_name,
        product_type=payload.product_type,
        has_genverse=payload.product_type in ("genverse", "genverse_evaluation"),
        has_evaluation=payload.product_type in ("evaluation", "genverse_evaluation"),
    )
    db.add(org)
    await db.flush()

    member = OrgMember(org_id=org.id, user_id=user.id, role="org_admin", status="active")
    db.add(member)

    # Personal subscription for admin
    await _create_free_subscription(db, user_id=user.id, workspace_type="individual")
    # Org subscription
    await _create_free_subscription(db, org_id=org.id, workspace_type="organization")

    await db.commit()
    await db.refresh(user)

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: DBSession):
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise CredentialsException("Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    user.last_login_date = datetime.now(timezone.utc)
    await db.commit()

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(payload: RefreshTokenRequest, db: DBSession):
    token_data = verify_refresh_token(payload.refresh_token)
    if not token_data:
        raise CredentialsException("Invalid or expired refresh token")
    user_id = token_data.get("sub")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise CredentialsException("User not found or inactive")

    access_token = create_access_token(str(user.id))
    new_refresh = create_refresh_token(str(user.id))
    return TokenResponse(access_token=access_token, refresh_token=new_refresh)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: CurrentUser):
    return current_user


@router.post("/change-password")
async def change_password(payload: ChangePasswordRequest, current_user: CurrentUser, db: DBSession):
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")
    current_user.hashed_password = hash_password(payload.new_password)
    await db.commit()
    return {"message": "Password changed successfully"}


@router.post("/logout")
async def logout(current_user: CurrentUser):
    # JWT is stateless - client should discard the token
    # Optionally implement a token blacklist here
    return {"message": "Logged out successfully"}
