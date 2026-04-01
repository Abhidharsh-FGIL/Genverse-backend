import uuid
import secrets
from datetime import date, datetime, timezone, timedelta
from fastapi import APIRouter, BackgroundTasks, HTTPException, status, Query, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import DBSession, CurrentUser
from app.models.organization import Organization, OrgMember, OrgInvitation, OrgModuleOverride
from app.models.subscription import Subscription, PlanDefinition
from app.models.user import User
from app.models.content import Ebook, MindMap, VideoProject, UserLibraryItem
from app.models.assessment import PracticeAssessment
from app.models.evaluation import EvaluationAssessment
from app.schemas.organization import (
    OrganizationCreate,
    OrganizationUpdate,
    OrganizationResponse,
    OrgMemberResponse,
    InviteMemberRequest,
    BulkInviteRequest,
    UpdateMemberRoleRequest,
    OrgInvitationResponse,
    ModuleOverrideRequest,
    ModuleOverrideResponse,
    DirectAddMemberRequest,
)
from app.config import settings
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.email_service import send_organization_invitation_email

import logging
logger = logging.getLogger(__name__)


def _send_invite_email_safe(**kwargs):
    """Wrapper to catch and log any exception from the background email task."""
    try:
        logger.info(f"[BG-Email] Sending org invite email to {kwargs.get('to_email')}...")
        result = send_organization_invitation_email(**kwargs)
        logger.info(f"[BG-Email] Result for {kwargs.get('to_email')}: {result}")
        return result
    except Exception as e:
        logger.error(f"[BG-Email] FAILED for {kwargs.get('to_email')}: {type(e).__name__}: {e}", exc_info=True)
        return False


router = APIRouter()


class UpdateInvitationRequest(BaseModel):
    status: str  # "expired" | "revoked" | "pending"


# Fallback quotas if PlanDefinition table has no data yet
_PLAN_QUOTA_FALLBACKS = {
    "org_basic":      {"monthly_points": 5000,  "storage_mb": 5120,  "max_seats": 50},
    "org_pro":        {"monthly_points": 20000, "storage_mb": 51200, "max_seats": 1000},
    "org_evaluation": {"monthly_points": 3000,  "storage_mb": 2048,  "max_seats": 100},
    "free":           {"monthly_points": 100,   "storage_mb": 100,   "max_seats": 5},
}


@router.post("/", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(payload: OrganizationCreate, current_user: CurrentUser, db: DBSession):
    """Create a new organization and add the current user as org_admin."""
    org = Organization(
        name=payload.name,
        product_type=payload.product_type,
        has_genverse=payload.has_genverse,
        has_evaluation=payload.has_evaluation,
        logo_url=payload.logo_url,
    )
    db.add(org)
    await db.flush()

    member = OrgMember(org_id=org.id, user_id=current_user.id, role="org_admin", status="active")
    db.add(member)

    # Resolve plan quotas from PlanDefinition table, fall back to hardcoded defaults
    selected_plan = payload.plan or "free"
    plan_def_result = await db.execute(
        select(PlanDefinition).where(PlanDefinition.plan == selected_plan)
    )
    plan_def = plan_def_result.scalar_one_or_none()
    if plan_def:
        monthly_points = plan_def.monthly_points
        storage_mb = plan_def.storage_mb
        max_seats = plan_def.max_seats
    else:
        fallback = _PLAN_QUOTA_FALLBACKS.get(selected_plan, _PLAN_QUOTA_FALLBACKS["free"])
        monthly_points = fallback["monthly_points"]
        storage_mb = fallback["storage_mb"]
        max_seats = fallback["max_seats"]

    now = datetime.now(timezone.utc)
    org_subscription = Subscription(
        org_id=org.id,
        plan=selected_plan,
        status="active",
        workspace_type="organization",
        points_balance=monthly_points,
        points_monthly_quota=monthly_points,
        storage_limit_mb=storage_mb,
        max_seats=max_seats,
        current_period_start=now,
        current_period_end=now + timedelta(days=30),
    )
    db.add(org_subscription)

    await db.commit()
    await db.refresh(org)
    return org


async def _require_org_admin(user_id: uuid.UUID, org_id: uuid.UUID, db: AsyncSession):
    result = await db.execute(
        select(OrgMember).where(
            OrgMember.org_id == org_id,
            OrgMember.user_id == user_id,
            OrgMember.role == "org_admin",
            OrgMember.status == "active",
        )
    )
    if not result.scalar_one_or_none():
        raise ForbiddenException("Organization admin access required")


@router.get("/my", response_model=list[OrganizationResponse])
async def get_my_organizations(current_user: CurrentUser, db: DBSession):
    """Return all organizations the current user is a member of."""
    result = await db.execute(
        select(Organization)
        .join(OrgMember, OrgMember.org_id == Organization.id)
        .where(OrgMember.user_id == current_user.id, OrgMember.status == "active")
    )
    return result.scalars().all()


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(org_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")
    return org


@router.patch("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: uuid.UUID, payload: OrganizationUpdate, current_user: CurrentUser, db: DBSession
):
    await _require_org_admin(current_user.id, org_id, db)
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(org, key, value)
    await db.commit()
    await db.refresh(org)
    return org


@router.post("/{org_id}/logo", response_model=OrganizationResponse)
async def upload_org_logo(
    org_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
    file: UploadFile = File(...),
):
    """Upload or replace the organisation logo/icon."""
    await _require_org_admin(current_user.id, org_id, db)
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")

    from app.services.storage_service import StorageService
    storage = StorageService()
    file_info = await storage.upload_file(
        file=file, bucket="org-logos", prefix=str(org_id)
    )
    org.logo_url = file_info["url"]
    await db.commit()
    await db.refresh(org)
    return org


# ---- Members ----

@router.get("/{org_id}/members", response_model=list[OrgMemberResponse])
async def list_members(
    org_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
    role: str | None = Query(None),
):
    await _require_org_admin(current_user.id, org_id, db)
    q = select(OrgMember, User).join(User, OrgMember.user_id == User.id).where(OrgMember.org_id == org_id)
    if role:
        q = q.where(OrgMember.role == role)
    result = await db.execute(q)
    rows = result.all()
    members = []
    for member, user in rows:
        members.append(
            OrgMemberResponse(
                id=member.id,
                org_id=member.org_id,
                user_id=member.user_id,
                role=member.role,
                status=member.status,
                joined_at=member.joined_at,
                user_name=user.name,
                user_email=user.email,
            )
        )
    return members


@router.post("/{org_id}/members", response_model=OrgMemberResponse, status_code=status.HTTP_201_CREATED)
async def add_member_by_user_id(
    org_id: uuid.UUID,
    payload: DirectAddMemberRequest,
    current_user: CurrentUser,
    db: DBSession,
    background_tasks: BackgroundTasks,
):
    """Add a user to the org directly by user_id (used when user already exists in the system)."""
    await _require_org_admin(current_user.id, org_id, db)

    user_id = uuid.UUID(payload.user_id) if isinstance(payload.user_id, str) else payload.user_id
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise NotFoundException("User not found")

    # Fetch org for email context
    org_result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = org_result.scalar_one_or_none()
    admin_name = current_user.name or current_user.email

    existing = await db.execute(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
    )
    existing_member = existing.scalar_one_or_none()
    if existing_member:
        existing_member.role = payload.role
        existing_member.status = "active"
    else:
        existing_member = OrgMember(org_id=org_id, user_id=user_id, role=payload.role, status="active")
        db.add(existing_member)

    # Ensure user has the corresponding UserRole (e.g. "teacher")
    from app.models.user import UserRole
    role_result = await db.execute(
        select(UserRole).where(UserRole.user_id == user_id, UserRole.role == payload.role)
    )
    if not role_result.scalar_one_or_none():
        db.add(UserRole(user_id=user_id, role=payload.role))

    await db.commit()
    await db.refresh(existing_member)

    # Send notification email in background
    if org:
        background_tasks.add_task(
            _send_invite_email_safe,
            to_email=user.email,
            admin_name=admin_name,
            organization_name=org.name,
            role=payload.role,
            accept_url="",
            is_existing_user=True,
        )

    return OrgMemberResponse(
        id=existing_member.id, org_id=existing_member.org_id, user_id=existing_member.user_id,
        role=existing_member.role, status=existing_member.status, joined_at=existing_member.joined_at,
        user_name=user.name, user_email=user.email,
    )


@router.post("/{org_id}/members/invite", response_model=OrgInvitationResponse)
async def invite_member(
    org_id: uuid.UUID, payload: InviteMemberRequest, current_user: CurrentUser, db: DBSession,
    background_tasks: BackgroundTasks,
):
    await _require_org_admin(current_user.id, org_id, db)

    # Validate mandatory fields for students
    if payload.role == "student":
        missing = _validate_student_mandatory_fields(payload.model_dump())
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing mandatory fields for student: {', '.join(missing)}",
            )

    # Fetch org name and admin name for the email
    org_result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = org_result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")
    admin_name = current_user.name or current_user.email

    # Check if user already exists in the system
    user_result = await db.execute(select(User).where(User.email == payload.email))
    existing_user = user_result.scalar_one_or_none()

    if existing_user:
        # User exists — add them directly as a member
        existing_member = await db.execute(
            select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == existing_user.id)
        )
        member = existing_member.scalar_one_or_none()
        if member:
            member.role = payload.role
            member.status = "active"
        else:
            db.add(OrgMember(org_id=org_id, user_id=existing_user.id, role=payload.role, status="active"))

        # Ensure user has the corresponding UserRole (e.g. "teacher")
        from app.models.user import UserRole
        role_result = await db.execute(
            select(UserRole).where(UserRole.user_id == existing_user.id, UserRole.role == payload.role)
        )
        if not role_result.scalar_one_or_none():
            db.add(UserRole(user_id=existing_user.id, role=payload.role))

    # Always create the invitation record for audit trail
    token = secrets.token_urlsafe(32)
    invitation = OrgInvitation(
        org_id=org_id,
        email=payload.email,
        role=payload.role,
        invited_by=current_user.id,
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        status="accepted" if existing_user else "pending",
    )
    db.add(invitation)
    await db.commit()
    await db.refresh(invitation)

    # Send email in background (non-blocking)
    accept_url = f"{settings.FRONTEND_URL}/genverse/accept-invite?token={token}"
    background_tasks.add_task(
        _send_invite_email_safe,
        to_email=payload.email,
        admin_name=admin_name,
        organization_name=org.name,
        role=payload.role,
        accept_url=accept_url,
        is_existing_user=bool(existing_user),
    )

    return invitation


@router.post("/{org_id}/members/invite/bulk")
async def bulk_invite(
    org_id: uuid.UUID, payload: BulkInviteRequest, current_user: CurrentUser, db: DBSession,
    background_tasks: BackgroundTasks,
):
    await _require_org_admin(current_user.id, org_id, db)

    org_result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = org_result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")
    admin_name = current_user.name or current_user.email

    invitations = []
    skipped = []
    validation_errors = []
    profile_fields = [
        "name", "phone", "grade", "section", "board_preference", "subjects",
        "date_of_birth", "gender", "blood_group", "city", "state", "pincode",
        "roll_number", "parent_name", "parent_phone",
        "employee_id", "department", "qualification",
        "emergency_contact_name", "emergency_contact_phone",
        "emergency_contact_relation", "address",
    ]
    for item in payload.members:
        # Validate mandatory fields for students
        if item.role == "student":
            missing = _validate_student_mandatory_fields(item.model_dump())
            if missing:
                validation_errors.append({
                    "email": item.email,
                    "missing_fields": missing,
                })
                continue
        # Check if user already exists
        user_result = await db.execute(select(User).where(User.email == item.email))
        existing_user = user_result.scalar_one_or_none()

        if existing_user:
            existing_member_result = await db.execute(
                select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == existing_user.id)
            )
            member = existing_member_result.scalar_one_or_none()
            if member:
                # Skip already-existing members — do NOT overwrite their role
                skipped.append({"email": item.email, "role": member.role, "reason": "already a member"})
                continue
            else:
                db.add(OrgMember(org_id=org_id, user_id=existing_user.id, role=item.role, status="active"))

            from app.models.user import UserRole
            role_result = await db.execute(
                select(UserRole).where(UserRole.user_id == existing_user.id, UserRole.role == item.role)
            )
            if not role_result.scalar_one_or_none():
                db.add(UserRole(user_id=existing_user.id, role=item.role))

            # Apply optional profile fields from CSV
            item_data = item.model_dump(include=set(profile_fields), exclude_unset=True)
            for key, value in item_data.items():
                if value is not None:
                    if key == "date_of_birth":
                        try:
                            value = date.fromisoformat(value)
                        except (ValueError, TypeError):
                            continue
                    setattr(existing_user, key, value)

        token = secrets.token_urlsafe(32)
        inv = OrgInvitation(
            org_id=org_id,
            email=item.email,
            role=item.role,
            invited_by=current_user.id,
            token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            status="accepted" if existing_user else "pending",
        )
        db.add(inv)

        accept_url = f"{settings.FRONTEND_URL}/genverse/accept-invite?token={token}"
        background_tasks.add_task(
            _send_invite_email_safe,
            to_email=item.email,
            admin_name=admin_name,
            organization_name=org.name,
            role=item.role,
            accept_url=accept_url,
            is_existing_user=bool(existing_user),
        )

        invitations.append({"email": item.email, "role": item.role})
    await db.commit()
    return {
        "invited": len(invitations),
        "skipped": len(skipped),
        "skipped_members": skipped,
        "validation_errors": validation_errors,
        "members": invitations,
    }


@router.post("/{org_id}/members/add-direct", response_model=OrgMemberResponse)
async def add_member_direct(
    org_id: uuid.UUID, payload: DirectAddMemberRequest, current_user: CurrentUser, db: DBSession
):
    """Directly add a user as an org member by email (no invitation email required)."""
    await _require_org_admin(current_user.id, org_id, db)

    user_result = await db.execute(select(User).where(User.email == payload.email))
    user = user_result.scalar_one_or_none()
    if not user:
        raise NotFoundException(f"No user found with email: {payload.email}. They must sign up first.")

    existing = await db.execute(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user.id)
    )
    existing_member = existing.scalar_one_or_none()
    if existing_member:
        # Update role if already a member
        existing_member.role = payload.role
        existing_member.status = "active"
        await db.commit()
        await db.refresh(existing_member)
        return OrgMemberResponse(
            id=existing_member.id, org_id=existing_member.org_id, user_id=existing_member.user_id,
            role=existing_member.role, status=existing_member.status, joined_at=existing_member.joined_at,
            user_name=user.name, user_email=user.email,
        )

    member = OrgMember(org_id=org_id, user_id=user.id, role=payload.role, status="active")
    db.add(member)
    await db.commit()
    await db.refresh(member)
    return OrgMemberResponse(
        id=member.id, org_id=member.org_id, user_id=member.user_id,
        role=member.role, status=member.status, joined_at=member.joined_at,
        user_name=user.name, user_email=user.email,
    )


@router.patch("/{org_id}/members/{user_id}", response_model=OrgMemberResponse)
async def update_member_role(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: UpdateMemberRoleRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    await _require_org_admin(current_user.id, org_id, db)
    result = await db.execute(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
    )
    member = result.scalar_one_or_none()
    if not member:
        raise NotFoundException("Member not found")
    if payload.role is not None:
        member.role = payload.role
    if payload.status is not None and payload.status in ("active", "deactivated"):
        member.status = payload.status
    await db.commit()
    await db.refresh(member)
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    return OrgMemberResponse(
        id=member.id, org_id=member.org_id, user_id=member.user_id,
        role=member.role, status=member.status, joined_at=member.joined_at,
        user_name=user.name if user else None, user_email=user.email if user else None,
    )


@router.delete("/{org_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    org_id: uuid.UUID, user_id: uuid.UUID, current_user: CurrentUser, db: DBSession
):
    await _require_org_admin(current_user.id, org_id, db)
    result = await db.execute(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
    )
    member = result.scalar_one_or_none()
    if not member:
        raise NotFoundException("Member not found")
    member.status = "deactivated"
    await db.commit()


class AdminUpdateMemberProfile(BaseModel):
    name: str | None = None
    phone: str | None = None
    address: str | None = None
    grade: int | None = None
    section: str | None = None
    subjects: list[str] | None = None
    language: str | None = None
    board_preference: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None
    blood_group: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    emergency_contact_name: str | None = None
    emergency_contact_phone: str | None = None
    emergency_contact_relation: str | None = None
    employee_id: str | None = None
    department: str | None = None
    qualification: str | None = None
    roll_number: str | None = None
    parent_name: str | None = None
    parent_phone: str | None = None


@router.get("/{org_id}/members/{user_id}/profile")
async def get_member_profile(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
):
    """Get full profile fields for a member."""
    await _require_org_admin(current_user.id, org_id, db)
    result = await db.execute(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
    )
    member = result.scalar_one_or_none()
    if not member:
        raise NotFoundException("Member not found in this organization")

    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise NotFoundException("User not found")

    return {
        "name": user.name,
        "phone": user.phone,
        "address": user.address,
        "grade": user.grade,
        "section": user.section,
        "subjects": user.subjects,
        "board_preference": user.board_preference,
        "date_of_birth": str(user.date_of_birth) if user.date_of_birth else None,
        "gender": user.gender,
        "blood_group": user.blood_group,
        "city": user.city,
        "state": user.state,
        "pincode": user.pincode,
        "emergency_contact_name": user.emergency_contact_name,
        "emergency_contact_phone": user.emergency_contact_phone,
        "emergency_contact_relation": user.emergency_contact_relation,
        "employee_id": user.employee_id,
        "department": user.department,
        "qualification": user.qualification,
        "roll_number": user.roll_number,
        "parent_name": user.parent_name,
        "parent_phone": user.parent_phone,
    }


@router.patch("/{org_id}/members/{user_id}/profile", response_model=OrgMemberResponse)
async def admin_update_member_profile(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: AdminUpdateMemberProfile,
    current_user: CurrentUser,
    db: DBSession,
):
    """Org admin can update a member's profile fields."""
    await _require_org_admin(current_user.id, org_id, db)
    # Verify user is a member of this org
    result = await db.execute(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
    )
    member = result.scalar_one_or_none()
    if not member:
        raise NotFoundException("Member not found in this organization")

    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise NotFoundException("User not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "date_of_birth" in update_data and update_data["date_of_birth"]:
        update_data["date_of_birth"] = date.fromisoformat(update_data["date_of_birth"])
    for key, value in update_data.items():
        setattr(user, key, value)
    await db.commit()
    await db.refresh(user)
    await db.refresh(member)

    return OrgMemberResponse(
        id=member.id, org_id=member.org_id, user_id=member.user_id,
        role=member.role, status=member.status, joined_at=member.joined_at,
        user_name=user.name, user_email=user.email,
    )


# ---- Content Governance ----

@router.get("/{org_id}/content")
async def get_org_content(
    org_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
):
    """Aggregate all content created by org members for the Content Governance page."""
    await _require_org_admin(current_user.id, org_id, db)

    # Get member list with user info
    mem_result = await db.execute(
        select(OrgMember, User).join(User, OrgMember.user_id == User.id)
        .where(OrgMember.org_id == org_id)
    )
    mem_rows = mem_result.all()
    member_map = {}
    role_map = {}
    for member, user in mem_rows:
        member_map[str(member.user_id)] = user.name or "Unknown"
        role_map[str(member.user_id)] = member.role

    # Assessments
    a_result = await db.execute(
        select(PracticeAssessment).where(PracticeAssessment.org_id == org_id)
        .order_by(PracticeAssessment.created_at.desc()).limit(200)
    )
    assessments = [
        {
            "id": str(a.id), "title": a.title, "subject": a.subject,
            "difficulty": a.difficulty, "mode": a.mode,
            "question_count": a.question_count,
            "created_by": str(a.created_by),
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in a_result.scalars().all()
    ]

    # eBooks
    e_result = await db.execute(
        select(Ebook).where(Ebook.org_id == org_id)
        .order_by(Ebook.created_at.desc()).limit(200)
    )
    ebooks = [
        {
            "id": str(e.id), "title": e.title, "subject": e.subject,
            "source_type": e.source_type, "page_count": e.page_count,
            "user_id": str(e.user_id),
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in e_result.scalars().all()
    ]

    # Mind Maps
    m_result = await db.execute(
        select(MindMap).where(MindMap.org_id == org_id)
        .order_by(MindMap.created_at.desc()).limit(200)
    )
    mindmaps = [
        {
            "id": str(m.id), "title": m.title, "topic": m.topic, "subject": m.subject,
            "user_id": str(m.user_id),
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in m_result.scalars().all()
    ]

    # Video Projects
    v_result = await db.execute(
        select(VideoProject).where(VideoProject.org_id == org_id)
        .order_by(VideoProject.created_at.desc()).limit(200)
    )
    videos = [
        {
            "id": str(v.id), "title": v.title, "topic": v.topic, "subject": v.subject,
            "status": v.status,
            "user_id": str(v.user_id),
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in v_result.scalars().all()
    ]

    # Library Documents
    l_result = await db.execute(
        select(UserLibraryItem).where(UserLibraryItem.org_id == org_id)
        .order_by(UserLibraryItem.created_at.desc()).limit(200)
    )
    library = [
        {
            "id": str(li.id), "title": li.title, "file_type": li.file_type,
            "file_size_mb": li.file_size_mb,
            "user_id": str(li.user_id),
            "created_at": li.created_at.isoformat() if li.created_at else None,
        }
        for li in l_result.scalars().all()
    ]

    # Evaluation Assessments
    ev_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.org_id == org_id)
        .order_by(EvaluationAssessment.created_at.desc()).limit(200)
    )
    evaluations = [
        {
            "id": str(ev.id), "title": ev.title, "mode": ev.mode,
            "difficulty": ev.difficulty, "question_count": ev.question_count,
            "max_score": ev.max_score, "status": ev.status,
            "grade": ev.grade, "board": ev.board,
            "created_by": str(ev.created_by),
            "created_at": ev.created_at.isoformat() if ev.created_at else None,
        }
        for ev in ev_result.scalars().all()
    ]

    return {
        "members": member_map,
        "roles": role_map,
        "assessments": assessments,
        "evaluations": evaluations,
        "ebooks": ebooks,
        "mindmaps": mindmaps,
        "videos": videos,
        "library": library,
    }


# ---- Module Overrides ----

@router.get("/{org_id}/modules", response_model=list[ModuleOverrideResponse])
async def get_module_overrides(org_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(OrgModuleOverride).where(OrgModuleOverride.org_id == org_id)
    )
    return result.scalars().all()


@router.put("/{org_id}/modules", response_model=ModuleOverrideResponse)
async def set_module_override(
    org_id: uuid.UUID, payload: ModuleOverrideRequest, current_user: CurrentUser, db: DBSession
):
    await _require_org_admin(current_user.id, org_id, db)
    result = await db.execute(
        select(OrgModuleOverride).where(
            OrgModuleOverride.org_id == org_id,
            OrgModuleOverride.feature_key == payload.feature_key,
        )
    )
    override = result.scalar_one_or_none()
    if not override:
        override = OrgModuleOverride(org_id=org_id, **payload.model_dump())
        db.add(override)
    else:
        override.enabled = payload.enabled
        override.access_role = payload.access_role
    # Cascade: when disabling a parent module, disable all its sub-features
    if not payload.enabled and '.' not in payload.feature_key:
        await db.execute(
            update(OrgModuleOverride)
            .where(OrgModuleOverride.org_id == org_id)
            .where(OrgModuleOverride.feature_key.like(f"{payload.feature_key}.%"))
            .values(enabled=False)
        )
    await db.commit()
    await db.refresh(override)
    return override


# ---- Module Overrides (frontend-compatible aliases using /module-overrides path) ----

@router.get("/{org_id}/module-overrides", response_model=list[ModuleOverrideResponse])
async def get_module_overrides_alias(org_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(OrgModuleOverride).where(OrgModuleOverride.org_id == org_id)
    )
    return result.scalars().all()


@router.post("/{org_id}/module-overrides", response_model=ModuleOverrideResponse)
async def set_module_override_alias(
    org_id: uuid.UUID, payload: ModuleOverrideRequest, current_user: CurrentUser, db: DBSession
):
    await _require_org_admin(current_user.id, org_id, db)
    result = await db.execute(
        select(OrgModuleOverride).where(
            OrgModuleOverride.org_id == org_id,
            OrgModuleOverride.feature_key == payload.feature_key,
        )
    )
    override = result.scalar_one_or_none()
    if not override:
        override = OrgModuleOverride(org_id=org_id, **payload.model_dump())
        db.add(override)
    else:
        override.enabled = payload.enabled
        override.access_role = payload.access_role
    # Cascade: when disabling a parent module, disable all its sub-features
    if not payload.enabled and '.' not in payload.feature_key:
        await db.execute(
            update(OrgModuleOverride)
            .where(OrgModuleOverride.org_id == org_id)
            .where(OrgModuleOverride.feature_key.like(f"{payload.feature_key}.%"))
            .values(enabled=False)
        )
    await db.commit()
    await db.refresh(override)
    return override


@router.put("/{org_id}/module-overrides/bulk", response_model=list[ModuleOverrideResponse])
async def set_module_overrides_bulk(
    org_id: uuid.UUID, payloads: list[ModuleOverrideRequest], current_user: CurrentUser, db: DBSession
):
    """Bulk upsert module/feature overrides."""
    await _require_org_admin(current_user.id, org_id, db)
    results = []
    for payload in payloads:
        result = await db.execute(
            select(OrgModuleOverride).where(
                OrgModuleOverride.org_id == org_id,
                OrgModuleOverride.feature_key == payload.feature_key,
            )
        )
        override = result.scalar_one_or_none()
        if not override:
            override = OrgModuleOverride(org_id=org_id, **payload.model_dump())
            db.add(override)
        else:
            override.enabled = payload.enabled
            override.access_role = payload.access_role
        results.append(override)
    await db.commit()
    for o in results:
        await db.refresh(o)
    return results


# ---- Invitations ----

@router.get("/{org_id}/invitations", response_model=list[OrgInvitationResponse])
async def list_invitations(
    org_id: uuid.UUID, current_user: CurrentUser, db: DBSession,
    invitation_status: str | None = Query(None, alias="status"),
):
    """List pending (or all) invitations for an org."""
    await _require_org_admin(current_user.id, org_id, db)
    q = select(OrgInvitation).where(OrgInvitation.org_id == org_id)
    if invitation_status:
        q = q.where(OrgInvitation.status == invitation_status)
    else:
        q = q.where(OrgInvitation.status == "pending")
    q = q.order_by(OrgInvitation.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.patch("/invitations/{invitation_id}")
async def update_invitation_status(
    invitation_id: uuid.UUID, payload: UpdateInvitationRequest, current_user: CurrentUser, db: DBSession
):
    """Update an invitation's status (e.g. expire or revoke it)."""
    result = await db.execute(select(OrgInvitation).where(OrgInvitation.id == invitation_id))
    invitation = result.scalar_one_or_none()
    if not invitation:
        raise NotFoundException("Invitation not found")
    await _require_org_admin(current_user.id, invitation.org_id, db)
    invitation.status = payload.status
    await db.commit()
    await db.refresh(invitation)
    return {"id": str(invitation.id), "status": invitation.status}


# ---- Accept Invitation ----

@router.post("/invitations/{token}/accept")
async def accept_invitation(token: str, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(OrgInvitation).where(
            OrgInvitation.token == token,
            OrgInvitation.status == "pending",
        )
    )
    invitation = result.scalar_one_or_none()
    if not invitation:
        raise NotFoundException("Invitation not found or already used")
    if invitation.expires_at and invitation.expires_at < datetime.now(timezone.utc):
        invitation.status = "expired"
        await db.commit()
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Invitation has expired")

    existing = await db.execute(
        select(OrgMember).where(
            OrgMember.org_id == invitation.org_id,
            OrgMember.user_id == current_user.id,
        )
    )
    if not existing.scalar_one_or_none():
        member = OrgMember(
            org_id=invitation.org_id,
            user_id=current_user.id,
            role=invitation.role,
            status="active",
        )
        db.add(member)

    # Ensure the user has the corresponding UserRole (e.g. "org_admin", "teacher")
    from app.models.user import UserRole
    role_check = await db.execute(
        select(UserRole).where(UserRole.user_id == current_user.id, UserRole.role == invitation.role)
    )
    if not role_check.scalar_one_or_none():
        db.add(UserRole(user_id=current_user.id, role=invitation.role))

    invitation.status = "accepted"
    await db.commit()
    return {"message": "Invitation accepted", "org_id": str(invitation.org_id), "role": invitation.role}


# ---- Academic Year ----

class AcademicYearRequest(BaseModel):
    academic_year: str  # e.g. "2025-2026"


@router.get("/{org_id}/academic-year")
async def get_academic_year(org_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")
    return {"academic_year": org.current_academic_year}


@router.patch("/{org_id}/academic-year")
async def set_academic_year(
    org_id: uuid.UUID, payload: AcademicYearRequest, current_user: CurrentUser, db: DBSession
):
    await _require_org_admin(current_user.id, org_id, db)
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")
    org.current_academic_year = payload.academic_year
    await db.commit()
    return {"academic_year": org.current_academic_year}


@router.get("/{org_id}/academic-years")
async def list_academic_years(org_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Get all academic years that have data (classes or class teachers) for this org."""
    from app.models.classes import Class, GradeSectionTeacher

    # Get distinct years from classes
    class_years = await db.execute(
        select(Class.academic_year).where(
            Class.org_id == org_id,
            Class.academic_year.isnot(None),
        ).distinct()
    )
    # Get distinct years from class teacher assignments
    ct_years = await db.execute(
        select(GradeSectionTeacher.academic_year).where(
            GradeSectionTeacher.org_id == org_id,
        ).distinct()
    )

    years = set()
    for r in class_years.scalars().all():
        if r:
            years.add(r)
    for r in ct_years.scalars().all():
        if r:
            years.add(r)

    # Also include the current academic year from org settings
    org_result = await db.execute(select(Organization.current_academic_year).where(Organization.id == org_id))
    current = org_result.scalar_one_or_none()
    if current:
        years.add(current)

    return {"years": sorted(years, reverse=True), "current": current}


# ---- Board usage check ----

@router.get("/{org_id}/board-usage")
async def check_board_usage(org_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Check which boards are in use by classes and students in this org."""
    await _require_org_admin(current_user.id, org_id, db)

    from app.models.classes import Class

    # Get board usage in classes
    class_result = await db.execute(
        select(Class.board, func.count(Class.id))
        .where(Class.org_id == org_id, Class.is_active == True)
        .group_by(Class.board)
    )
    class_usage = {row[0]: row[1] for row in class_result.all()}

    # Get board usage in student profiles
    student_result = await db.execute(
        select(User.board_preference, func.count(User.id))
        .join(OrgMember, OrgMember.user_id == User.id)
        .where(
            OrgMember.org_id == org_id,
            OrgMember.role == "student",
            OrgMember.status == "active",
            User.board_preference.isnot(None),
        )
        .group_by(User.board_preference)
    )
    student_usage = {row[0]: row[1] for row in student_result.all()}

    # Combine all boards that are in use
    all_boards = set(class_usage.keys()) | set(student_usage.keys())
    usage = {}
    for board in all_boards:
        usage[board] = {
            "classes": class_usage.get(board, 0),
            "students": student_usage.get(board, 0),
        }

    return usage


# ---- Students by grade+section ----

@router.get("/{org_id}/students-by-grade")
async def get_students_by_grade(
    org_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
    grade: int = Query(...),
    section: str = Query(None),
):
    """Get students in the org filtered by grade and optionally section."""
    await _require_org_admin(current_user.id, org_id, db)
    q = (
        select(User, OrgMember)
        .join(OrgMember, OrgMember.user_id == User.id)
        .where(
            OrgMember.org_id == org_id,
            OrgMember.role == "student",
            OrgMember.status == "active",
            User.grade == grade,
            User.is_active == True,
        )
    )
    if section:
        q = q.where(User.section == section)
    result = await db.execute(q)
    rows = result.all()
    return [
        {
            "user_id": str(user.id),
            "user_name": user.name,
            "user_email": user.email,
            "roll_number": user.roll_number,
            "grade": user.grade,
            "section": user.section,
            "board_preference": user.board_preference,
        }
        for user, member in rows
    ]


# ---- Student mandatory field validation helper ----

def _validate_student_mandatory_fields(data: dict) -> list[str]:
    """Return list of missing mandatory fields for a student."""
    required = [
        "grade", "section", "board_preference", "roll_number",
        "emergency_contact_name", "emergency_contact_phone", "emergency_contact_relation",
    ]
    missing = [f for f in required if not data.get(f)]
    return missing


# ---- Grade Promotion ----

class GradePromotionRequest(BaseModel):
    from_grade: int
    to_grade: int
    section: str | None = None
    new_section: str | None = None
    exclude_student_ids: list[str] = []


@router.post("/{org_id}/promote")
async def promote_students(
    org_id: uuid.UUID, payload: GradePromotionRequest, current_user: CurrentUser, db: DBSession
):
    """Promote students from one grade to the next. Org admin only."""
    await _require_org_admin(current_user.id, org_id, db)

    from app.models.classes import ClassStudent, Class, GradePromotion

    # Get org for academic year
    org_result = await db.execute(select(Organization).where(Organization.id == org_id))
    org = org_result.scalar_one_or_none()
    if not org:
        raise NotFoundException("Organization not found")

    # Find all students in the org with matching grade (and optionally section)
    q = (
        select(User)
        .join(OrgMember, OrgMember.user_id == User.id)
        .where(
            OrgMember.org_id == org_id,
            OrgMember.role == "student",
            OrgMember.status == "active",
            User.grade == payload.from_grade,
        )
    )
    if payload.section:
        q = q.where(User.section == payload.section)

    result = await db.execute(q)
    students = result.scalars().all()

    exclude_set = set(payload.exclude_student_ids)
    promoted = []
    excluded = []

    for student in students:
        if str(student.id) in exclude_set:
            excluded.append({"id": str(student.id), "name": student.name, "email": student.email})
            continue

        old_grade = student.grade
        old_section = student.section

        # Update student grade and section
        student.grade = payload.to_grade
        if payload.new_section is not None:
            student.section = payload.new_section

        # Remove from all current class enrollments (classes in this org)
        class_ids_result = await db.execute(
            select(Class.id).where(Class.org_id == org_id, Class.is_active == True)
        )
        org_class_ids = [r[0] for r in class_ids_result.all()]
        if org_class_ids:
            enrollments_result = await db.execute(
                select(ClassStudent).where(
                    ClassStudent.student_id == student.id,
                    ClassStudent.class_id.in_(org_class_ids),
                )
            )
            for enrollment in enrollments_result.scalars().all():
                await db.delete(enrollment)

        # Create audit record
        db.add(GradePromotion(
            org_id=org_id,
            student_id=student.id,
            from_grade=old_grade,
            to_grade=payload.to_grade,
            from_section=old_section,
            to_section=payload.new_section or old_section,
            academic_year=org.current_academic_year,
            promoted_by=current_user.id,
        ))

        promoted.append({
            "id": str(student.id),
            "name": student.name,
            "email": student.email,
            "from_grade": old_grade,
            "to_grade": payload.to_grade,
        })

    await db.commit()
    return {
        "promoted_count": len(promoted),
        "excluded_count": len(excluded),
        "promoted": promoted,
        "excluded": excluded,
    }
