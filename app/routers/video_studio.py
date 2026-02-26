import uuid
from fastapi import APIRouter, status
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.content import VideoProject
from app.schemas.content import VideoScriptRequest, VideoProjectResponse
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()


@router.post("/script", response_model=VideoProjectResponse, status_code=status.HTTP_201_CREATED)
async def generate_video_script(payload: VideoScriptRequest, current_user: CurrentUser, db: DBSession):
    """Generate a video script using AI. Cost: 10 pts per script."""
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="generate_video_script", db=db)

    ai = AIService()
    script_json = await ai.generate_video_script(
        topic=payload.topic,
        subject=payload.subject,
        grade=payload.grade,
        duration_minutes=payload.duration_minutes,
        style=payload.style,
    )

    project = VideoProject(
        user_id=current_user.id,
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
    await points_service.deduct(user_id=current_user.id, action="generate_video_visuals", db=db)

    ai = AIService()
    visuals_json = await ai.generate_video_visuals(script_json=project.script_json)
    project.visuals_json = visuals_json
    project.status = "visuals_ready"
    await db.commit()
    await db.refresh(project)
    return project


@router.get("/", response_model=list[VideoProjectResponse])
async def list_video_projects(current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(VideoProject)
        .where(VideoProject.user_id == current_user.id)
        .order_by(VideoProject.created_at.desc())
    )
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
