import uuid
from fastapi import APIRouter, status, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.content import VideoProject
from app.schemas.content import VideoScriptRequest, VideoProjectResponse
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService, get_ai_service
from app.services.points_service import PointsService

router = APIRouter()


def _parse_org_id(org_id: str | None) -> uuid.UUID | None:
    if not org_id or org_id == "personal":
        return None
    try:
        return uuid.UUID(org_id)
    except ValueError:
        return None


def _apply_org_filter(q, column, org_id_param: str | None):
    parsed = _parse_org_id(org_id_param)
    if parsed is None:
        return q.where(column.is_(None))
    return q.where(column == parsed)


@router.post("/script", response_model=VideoProjectResponse, status_code=status.HTTP_201_CREATED)
async def generate_video_script(payload: VideoScriptRequest, current_user: CurrentUser, db: DBSession):
    """Generate a video script using AI. Cost: 10 pts per script."""
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="video_studio", db=db, org_id=_parse_org_id(payload.org_id))
    await points_service.deduct(user_id=current_user.id, action="generate_video_script", db=db, org_id=_parse_org_id(payload.org_id))

    ai = get_ai_service()
    script_json = await ai.generate_video_script(
        topic=payload.topic,
        subject=payload.subject,
        grade=payload.grade,
        duration_minutes=payload.duration_minutes,
        style=payload.style,
    )

    project = VideoProject(
        user_id=current_user.id,
        org_id=_parse_org_id(payload.org_id),
        title=f"Video: {payload.topic}",
        topic=payload.topic,
        subject=payload.subject,
        grade=payload.grade,
        script_json=script_json,
        status="script_ready",
        points_used=10,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.post("/{project_id}/visuals", response_model=VideoProjectResponse)
async def generate_video_visuals(project_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Generate visual references for an existing video project."""
    result = await db.execute(
        select(VideoProject).where(VideoProject.id == project_id, VideoProject.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundException("Video project not found")

    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="video_studio", db=db, org_id=project.org_id)
    await points_service.deduct(user_id=current_user.id, action="generate_video_visuals", db=db, org_id=project.org_id)

    ai = get_ai_service()
    visuals_json = await ai.generate_video_visuals(script_json=project.script_json)
    project.visuals_json = visuals_json
    project.status = "visuals_ready"
    await db.commit()
    await db.refresh(project)
    return project


@router.get("/", response_model=list[VideoProjectResponse])
async def list_video_projects(current_user: CurrentUser, db: DBSession, org_id: str | None = Query(None)):
    q = select(VideoProject).where(VideoProject.user_id == current_user.id)
    if org_id is not None:
        q = _apply_org_filter(q, VideoProject.org_id, org_id)
    q = q.order_by(VideoProject.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{project_id}", response_model=VideoProjectResponse)
async def get_video_project(project_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(VideoProject).where(VideoProject.id == project_id, VideoProject.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundException("Video project not found")
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video_project(project_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(VideoProject).where(VideoProject.id == project_id, VideoProject.user_id == current_user.id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundException("Video project not found")
    await db.delete(project)
    await db.commit()
