import uuid
import asyncio
import secrets
import random
import hashlib
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.dependencies import DBSession, CurrentUser
from app.models.user import User, UserRole, EmailVerification, PasswordResetToken
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
    GoogleLoginRequest,
    SendOtpRequest,
    VerifyOtpRequest,
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
from app.config import settings
from app.services.email_service import send_verification_otp

router = APIRouter()


async def _process_pending_enrollments(db: AsyncSession, user: User) -> None:
    """After a new user signs up, auto-enroll them in any classes they were pre-invited to."""
    from app.models.classes import PendingClassEnrollment, ClassStudent, Class
    from app.models.organization import OrgMember as OrgMemberModel

    pending_result = await db.execute(
        select(PendingClassEnrollment).where(PendingClassEnrollment.email == user.email)
    )
    pending_enrollments = pending_result.scalars().all()

    for pending in pending_enrollments:
        # Skip archived classes — don't auto-enroll into inactive classes
        class_check = await db.execute(
            select(Class.is_active).where(Class.id == pending.class_id)
        )
        is_active = class_check.scalar_one_or_none()
        if is_active is False:
            # Clean up the stale pending enrollment
            await db.delete(pending)
            continue

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


async def _process_pending_org_invitations(db: AsyncSession, user: User) -> None:
    """After signup, auto-accept any pending org invitations for this email.

    Creates OrgMember records and grants the invited role (e.g. teacher)
    so the user gets both personal workspace and org access immediately.
    """
    from app.models.organization import OrgInvitation, OrgMember as OrgMemberModel

    result = await db.execute(
        select(OrgInvitation).where(
            OrgInvitation.email == user.email,
            OrgInvitation.status == "pending",
        )
    )
    pending_invitations = result.scalars().all()

    now = datetime.now(timezone.utc)
    for inv in pending_invitations:
        # Skip expired invitations
        if inv.expires_at and inv.expires_at < now:
            inv.status = "expired"
            continue

        # Add as org member if not already
        existing = await db.execute(
            select(OrgMemberModel).where(
                OrgMemberModel.org_id == inv.org_id,
                OrgMemberModel.user_id == user.id,
            )
        )
        member = existing.scalar_one_or_none()
        if member:
            member.role = inv.role
            member.status = "active"
        else:
            db.add(OrgMemberModel(
                org_id=inv.org_id,
                user_id=user.id,
                role=inv.role,
                status="active",
            ))

        # Grant the role (e.g. "teacher") if user doesn't already have it
        role_exists = await db.execute(
            select(UserRole).where(UserRole.user_id == user.id, UserRole.role == inv.role)
        )
        if not role_exists.scalar_one_or_none():
            db.add(UserRole(user_id=user.id, role=inv.role))

        inv.status = "accepted"


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


def _hash_otp(otp: str) -> str:
    """SHA-256 hash of the OTP for secure storage."""
    return hashlib.sha256(otp.encode()).hexdigest()


@router.post("/send-otp")
async def send_otp(payload: SendOtpRequest, db: DBSession):
    """Generate a 6-digit OTP and email it to the user for verification."""
    print("I am send otp")
    email = payload.email.lower().strip()
    print(f"payload is {payload}")

    # Check if email is already registered
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise ConflictException("Email already registered. Please sign in.")

    # Clean up expired OTPs for this email
    await db.execute(
        delete(EmailVerification).where(
            EmailVerification.email == email,
            EmailVerification.expires_at <= datetime.now(timezone.utc),
        )
    )

    # Rate-limit: max 5 OTPs per email in the last hour
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    recent = await db.execute(
        select(EmailVerification).where(
            EmailVerification.email == email,
            EmailVerification.created_at >= one_hour_ago,
        )
    )
    if len(recent.scalars().all()) >= 5:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many OTP requests. Please try again later.",
        )

    # Generate 6-digit OTP
    otp = f"{random.randint(100000, 999999)}"

    # Store hashed OTP with 10-minute expiry
    verification = EmailVerification(
        email=email,
        otp_hash=_hash_otp(otp),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    db.add(verification)
    await db.commit()

    # Send the OTP email via thread pool to avoid blocking the event loop
    sent = await asyncio.to_thread(send_verification_otp, email, otp)
    if not sent:
        # If SMTP is not configured, still allow signup in dev mode
        if settings.ENVIRONMENT == "development":
            print(f"[DEV] OTP for {email}: {otp}", flush=True)
            return {"message": "OTP sent to your email.", "dev_otp": otp}
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email. Please try again.",
        )

    return {"message": "OTP sent to your email."}


async def _verify_otp(db: AsyncSession, email: str, otp: str) -> bool:
    """Verify the OTP for the given email. Returns True if valid."""
    result = await db.execute(
        select(EmailVerification)
        .where(
            EmailVerification.email == email,
            EmailVerification.expires_at > datetime.now(timezone.utc),
        )
        .order_by(EmailVerification.created_at.desc())
    )
    # Check the most recent non-expired OTP
    record = result.scalars().first()
    if not record:
        return False

    # Check max attempts (5)
    if record.attempts >= 5:
        return False

    record.attempts += 1

    if record.otp_hash != _hash_otp(otp):
        await db.commit()
        return False

    # OTP is valid — delete all OTPs for this email
    await db.execute(
        delete(EmailVerification).where(EmailVerification.email == email)
    )
    await db.commit()
    return True


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, db: DBSession):
    # Verify OTP first
    email = payload.email.lower().strip()
    otp_valid = await _verify_otp(db, email, payload.otp)
    if not otp_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code. Please request a new one.",
        )

    existing = await db.execute(select(User).where(User.email == email))
    existing_user = existing.scalar_one_or_none()
    if existing_user:
        if getattr(existing_user, "auth_provider", "email") == "google":
            raise ConflictException(
                "An account with this email already exists via Google. Please sign in with Google."
            )
        raise ConflictException("Email already registered. Please sign in with your email and password.")

    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        name=payload.name,
    )
    db.add(user)
    await db.flush()

    role = UserRole(user_id=user.id, role=payload.role)
    db.add(role)

    await _create_free_subscription(db, user_id=user.id, workspace_type="individual")

    # Auto-enroll in any classes the user was pre-invited to before having an account.
    # Wrapped in try/except so a missing table or any unexpected error never breaks signup.
    try:
        await _process_pending_enrollments(db, user)
    except Exception:
        pass

    # Auto-accept any pending org invitations for this email
    try:
        await _process_pending_org_invitations(db, user)
    except Exception:
        pass

    await db.commit()
    await db.refresh(user)

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/signup/organization", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def org_signup(payload: OrgSignupRequest, db: DBSession):
    existing = await db.execute(select(User).where(User.email == payload.email))
    existing_user = existing.scalar_one_or_none()
    if existing_user:
        if getattr(existing_user, "auth_provider", "email") == "google":
            raise ConflictException(
                "An account with this email already exists via Google. Please sign in with Google."
            )
        raise ConflictException("Email already registered. Please sign in with your email and password.")

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
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with this email. Please sign up to create an account.",
        )

    # If user signed up via Google, guide them to use Google sign-in
    if getattr(user, "auth_provider", "email") == "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account was created with Google. Please sign in using Google instead.",
        )

    if not verify_password(payload.password, user.hashed_password):
        raise CredentialsException("Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your account has been deactivated. Please contact support for assistance.")

    # Update streak: consecutive daily logins
    now = datetime.now(timezone.utc)
    today = now.date()
    if user.last_login_date:
        prev_date = user.last_login_date.date()
        delta = (today - prev_date).days
        if delta == 1:
            user.streak = (user.streak or 0) + 1
        elif delta > 1:
            user.streak = 1  # gap in logins — start fresh
        # delta == 0: same-day login, keep streak unchanged
    else:
        user.streak = 1  # first ever login
    user.last_login_date = now
    await db.commit()

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/google", response_model=TokenResponse)
async def google_auth(payload: GoogleLoginRequest, db: DBSession):
    """
    Authenticate via Google ID token.
    - If user exists: log them in.
    - If user doesn't exist and auto_signup=True: create account and log in.
    - If user doesn't exist and auto_signup=False: return 404.
    """
    # The frontend uses the implicit OAuth flow which returns an access_token.
    # Use Google's userinfo endpoint to verify the token and get user info.
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {payload.token}"},
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token",
        )

    token_info = resp.json()

    google_email = token_info.get("email")
    if not google_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account has no email",
        )

    # Look up user by email
    result = await db.execute(select(User).where(User.email == google_email))
    user = result.scalar_one_or_none()
    is_new_user = False

    if user:
        # Existing user — check auth provider
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account has been deactivated. Please contact support for assistance.",
            )

        # If user signed up with email/password, guide them
        if getattr(user, "auth_provider", "email") == "email":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This account was created with email and password. Please sign in using your email and password.",
            )

        # Update streak
        now = datetime.now(timezone.utc)
        today = now.date()
        if user.last_login_date:
            prev_date = user.last_login_date.date()
            delta = (today - prev_date).days
            if delta == 1:
                user.streak = (user.streak or 0) + 1
            elif delta > 1:
                user.streak = 1
        else:
            user.streak = 1
        user.last_login_date = now

        # Update avatar from Google if not set
        google_picture = token_info.get("picture")
        if google_picture and not user.avatar_url:
            user.avatar_url = google_picture

        await db.commit()
    else:
        # No account found
        if not payload.auto_signup:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No account found with this email. Please sign up first.",
            )

        # Auto-create account
        is_new_user = True
        google_name = token_info.get("name", google_email.split("@")[0])
        google_picture = token_info.get("picture")

        user = User(
            email=google_email,
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            name=google_name,
            avatar_url=google_picture,
            auth_provider="google",
            onboarding_completed=False,
        )
        db.add(user)
        await db.flush()

        role = UserRole(user_id=user.id, role="normal_user")
        db.add(role)

        await _create_free_subscription(db, user_id=user.id, workspace_type="individual")

        try:
            await _process_pending_enrollments(db, user)
        except Exception:
            pass

        # Auto-accept any pending org invitations for this email
        try:
            await _process_pending_org_invitations(db, user)
        except Exception:
            pass

        await db.commit()
        await db.refresh(user)

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))
    return TokenResponse(access_token=access_token, refresh_token=refresh_token, is_new_user=is_new_user)


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


@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest, db: DBSession):
    """Send a password reset link if the email exists and was created via email signup."""
    email = payload.email.lower().strip()

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        # Don't reveal whether the email exists — always return success
        return {"message": "If an account with that email exists, we've sent a reset link."}

    # Google users can't reset password — they should use Google sign-in
    if getattr(user, "auth_provider", "email") == "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account was created with Google. Please sign in using Google instead.",
        )

    if not user.is_active:
        return {"message": "If an account with that email exists, we've sent a reset link."}

    # Rate-limit: max 3 reset tokens per email per hour
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    recent = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.email == email,
            PasswordResetToken.created_at >= one_hour_ago,
        )
    )
    if len(recent.scalars().all()) >= 3:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many reset requests. Please try again later.",
        )

    # Clean up expired/used tokens
    await db.execute(
        delete(PasswordResetToken).where(
            PasswordResetToken.email == email,
            (PasswordResetToken.expires_at <= datetime.now(timezone.utc)) | (PasswordResetToken.used == True),
        )
    )

    # Generate secure token
    raw_token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

    reset_record = PasswordResetToken(
        email=email,
        token_hash=token_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    db.add(reset_record)
    await db.commit()

    # Build reset link — use frontend_url from request if provided, else fallback to config
    base_url = payload.frontend_url.rstrip("/") if payload.frontend_url else settings.FRONTEND_URL
    reset_link = f"{base_url}/auth/reset?token={raw_token}"

    from app.services.email_service import send_password_reset_email
    sent = await asyncio.to_thread(send_password_reset_email, email, reset_link)
    if not sent:
        if settings.ENVIRONMENT == "development":
            print(f"[DEV] Reset link for {email}: {reset_link}", flush=True)
            return {"message": "If an account with that email exists, we've sent a reset link.", "dev_link": reset_link}

    return {"message": "If an account with that email exists, we've sent a reset link."}


@router.get("/verify-reset-token")
async def verify_reset_token(token: str, db: DBSession):
    """Check if a password reset token is still valid (not expired, not used)."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    result = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.expires_at > datetime.now(timezone.utc),
            PasswordResetToken.used == False,
        )
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset link. Please request a new one.",
        )
    return {"valid": True}


@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest, db: DBSession):
    """Reset password using a valid token from the reset email."""
    token_hash = hashlib.sha256(payload.token.encode()).hexdigest()

    result = await db.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.expires_at > datetime.now(timezone.utc),
            PasswordResetToken.used == False,
        )
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset link. Please request a new one.",
        )

    # Find the user
    user_result = await db.execute(select(User).where(User.email == record.email))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No account found with this email. The reset link may be invalid.",
        )

    # Update password
    user.hashed_password = hash_password(payload.password)

    # Mark token as used
    record.used = True

    # Invalidate all other reset tokens for this email
    await db.execute(
        delete(PasswordResetToken).where(
            PasswordResetToken.email == record.email,
            PasswordResetToken.id != record.id,
        )
    )

    await db.commit()
    return {"message": "Password updated successfully. You can now sign in."}


@router.post("/logout")
async def logout(current_user: CurrentUser):
    # JWT is stateless - client should discard the token
    # Optionally implement a token blacklist here
    return {"message": "Logged out successfully"}
