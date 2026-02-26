import uuid
import secrets
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import DBSession, CurrentUser
from app.models.organization import Organization, OrgMember, OrgInvitation, OrgModuleOverride
from app.models.subscription import Subscription, PlanDefinition
from app.models.user import User
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
from app.core.exceptions import NotFoundException, ForbiddenException

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
):
    """Add a user to the org directly by user_id (used when user already exists in the system)."""
    await _require_org_admin(current_user.id, org_id, db)

    user_id = uuid.UUID(payload.user_id) if isinstance(payload.user_id, str) else payload.user_id
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise NotFoundException("User not found")

    existing = await db.execute(
        select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id)
    )
    existing_member = existing.scalar_one_or_none()
    if existing_member:
        existing_member.role = payload.role
        existing_member.status = "active"
        await db.commit()
        await db.refresh(existing_member)
        return OrgMemberResponse(
            id=existing_member.id, org_id=existing_member.org_id, user_id=existing_member.user_id,
            role=existing_member.role, status=existing_member.status, joined_at=existing_member.joined_at,
            user_name=user.name, user_email=user.email,
        )

    member = OrgMember(org_id=org_id, user_id=user_id, role=payload.role, status="active")
    db.add(member)
    await db.commit()
    await db.refresh(member)
    return OrgMemberResponse(
        id=member.id, org_id=member.org_id, user_id=member.user_id,
        role=member.role, status=member.status, joined_at=member.joined_at,
        user_name=user.name, user_email=user.email,
    )


@router.post("/{org_id}/members/invite", response_model=OrgInvitationResponse)
async def invite_member(
    org_id: uuid.UUID, payload: InviteMemberRequest, current_user: CurrentUser, db: DBSession
):
    await _require_org_admin(current_user.id, org_id, db)
    token = secrets.token_urlsafe(32)
    invitation = OrgInvitation(
        org_id=org_id,
        email=payload.email,
        role=payload.role,
        invited_by=current_user.id,
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    db.add(invitation)
    await db.commit()
    await db.refresh(invitation)
    # TODO: Send invitation email
    return invitation


@router.post("/{org_id}/members/invite/bulk")
async def bulk_invite(
    org_id: uuid.UUID, payload: BulkInviteRequest, current_user: CurrentUser, db: DBSession
):
    await _require_org_admin(current_user.id, org_id, db)
    invitations = []
    for item in payload.members:
        token = secrets.token_urlsafe(32)
        inv = OrgInvitation(
            org_id=org_id,
            email=item.email,
            role=item.role,
            invited_by=current_user.id,
            token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )
        db.add(inv)
        invitations.append({"email": item.email, "role": item.role})
    await db.commit()
    return {"invited": len(invitations), "members": invitations}


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
    member.role = payload.role
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
    await db.delete(member)
    await db.commit()


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
    await db.commit()
    await db.refresh(override)
    return override


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

    invitation.status = "accepted"
    await db.commit()
    return {"message": "Invitation accepted", "org_id": str(invitation.org_id), "role": invitation.role}
