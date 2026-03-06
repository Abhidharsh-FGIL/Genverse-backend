import uuid
from fastapi import APIRouter, HTTPException, status, Query, UploadFile, File
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.dependencies import DBSession, CurrentUser
from app.models.user import User, UserRole
from app.models.ai import AiContextSession
from app.schemas.user import UserResponse, UserUpdate, AiContextRequest, AiContextResponse
from app.services.storage_service import StorageService

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: CurrentUser):
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_profile(payload: UserUpdate, current_user: CurrentUser, db: DBSession):
    update_data = payload.model_dump(exclude_unset=True)

    # Handle role update separately (lives in user_roles table, not profiles)
    new_role = update_data.pop("role", None)
    if new_role:
        existing_role = await db.execute(
            select(UserRole).where(UserRole.user_id == current_user.id)
        )
        role_record = existing_role.scalar_one_or_none()
        if role_record:
            role_record.role = new_role
        else:
            db.add(UserRole(user_id=current_user.id, role=new_role))

    for key, value in update_data.items():
        setattr(current_user, key, value)
    await db.commit()
    # Reload with roles so the role property is accessible
    result = await db.execute(
        select(User).options(selectinload(User.roles)).where(User.id == current_user.id)
    )
    return result.scalar_one()


@router.post("/me/avatar", response_model=UserResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Upload a profile photo. Accepts JPEG, PNG, WebP (max 2 MB)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    content = await file.read()
    if len(content) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be under 2 MB")
    await file.seek(0)

    storage = StorageService()
    file_info = await storage.upload_file(
        file=file,
        bucket="avatars",
        prefix=str(current_user.id),
    )

    from app.config import settings as _settings
    from pathlib import Path as _Path

    try:
        storage_root = _Path(_settings.STORAGE_ROOT).resolve()
        abs_path = _Path(file_info["path"]).resolve()
        rel = abs_path.relative_to(storage_root)
        avatar_url = f"/uploads/{rel.as_posix()}"
    except Exception:
        avatar_url = file_info.get("path", "")

    current_user.avatar_url = avatar_url
    await db.commit()

    result = await db.execute(
        select(User).options(selectinload(User.roles)).where(User.id == current_user.id)
    )
    return result.scalar_one()


@router.get("/me/context", response_model=AiContextResponse)
async def get_ai_context(workspace_id: str = "personal", current_user: CurrentUser = None, db: DBSession = None):
    result = await db.execute(
        select(AiContextSession).where(
            AiContextSession.user_id == current_user.id,
            AiContextSession.workspace_id == workspace_id,
        )
    )
    ctx = result.scalar_one_or_none()
    if not ctx:
        ctx = AiContextSession(user_id=current_user.id, workspace_id=workspace_id)
        db.add(ctx)
        await db.commit()
        await db.refresh(ctx)
    return ctx


@router.put("/me/context", response_model=AiContextResponse)
async def update_ai_context(payload: AiContextRequest, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(AiContextSession).where(
            AiContextSession.user_id == current_user.id,
            AiContextSession.workspace_id == payload.workspace_id,
        )
    )
    ctx = result.scalar_one_or_none()
    if not ctx:
        ctx = AiContextSession(user_id=current_user.id, **payload.model_dump())
        db.add(ctx)
    else:
        for key, value in payload.model_dump(exclude_unset=True).items():
            setattr(ctx, key, value)
    await db.commit()
    await db.refresh(ctx)
    return ctx


@router.get("/search", response_model=list[UserResponse])
async def search_users(
    current_user: CurrentUser,
    db: DBSession,
    email: str | None = Query(None),
):
    """Search users by email. Used by org admins to find existing users to add."""
    if not email:
        return []
    result = await db.execute(
        select(User).options(selectinload(User.roles)).where(User.email == email.strip().lower())
    )
    user = result.scalar_one_or_none()
    return [user] if user else []


@router.get("/{user_id}/org-role")
async def get_user_org_roles(user_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Return the user's active organization membership roles."""
    from app.models.organization import OrgMember
    result = await db.execute(
        select(OrgMember).where(
            OrgMember.user_id == user_id,
            OrgMember.status == "active",
        )
    )
    members = result.scalars().all()
    return [{"role": m.role, "org_id": str(m.org_id)} for m in members]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(User).options(selectinload(User.roles)).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.get("/me/workspaces")
async def get_workspaces(current_user: CurrentUser, db: DBSession):
    from app.models.organization import OrgMember, Organization
    from app.schemas.organization import WorkspaceItem

    result = await db.execute(
        select(OrgMember, Organization)
        .join(Organization, OrgMember.org_id == Organization.id)
        .where(OrgMember.user_id == current_user.id, OrgMember.status == "active")
    )
    memberships = result.all()

    workspaces = [
        WorkspaceItem(id="personal", name="Personal Workspace", type="personal")
    ]
    for member, org in memberships:
        workspaces.append(
            WorkspaceItem(
                id=str(org.id),
                name=org.name,
                type="organization",
                role=member.role,
                logo_url=org.logo_url,
                has_genverse=org.has_genverse,
                has_evaluation=org.has_evaluation,
            )
        )
    return workspaces
