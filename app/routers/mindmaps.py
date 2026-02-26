import uuid
from fastapi import APIRouter, status
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.content import MindMap
from app.schemas.content import MindMapGenerateRequest, MindMapResponse
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()


@router.post("/generate", response_model=MindMapResponse, status_code=status.HTTP_201_CREATED)
async def generate_mindmap(payload: MindMapGenerateRequest, current_user: CurrentUser, db: DBSession):
    """Generate a visual mind map using AI."""
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="generate_mindmap", db=db)

    ai = AIService()
    mindmap_json = await ai.generate_mindmap(
        topic=payload.topic,
        subject=payload.subject,
        grade=payload.grade,
        board=payload.board,
        depth=payload.depth,
    )

    mindmap = MindMap(
        user_id=current_user.id,
        title=f"Mind Map: {payload.topic}",
        topic=payload.topic,
        subject=payload.subject,
        grade=payload.grade,
        board=payload.board,
        mindmap_json=mindmap_json,
    )
    db.add(mindmap)

    # Award XP
    current_user.xp = (current_user.xp or 0) + 15
    await db.commit()
    await db.refresh(mindmap)
    return mindmap


@router.get("/", response_model=list[MindMapResponse])
async def list_mindmaps(current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(MindMap)
        .where(MindMap.user_id == current_user.id)
        .order_by(MindMap.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{mindmap_id}", response_model=MindMapResponse)
async def get_mindmap(mindmap_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(MindMap).where(MindMap.id == mindmap_id, MindMap.user_id == current_user.id)
    )
    mindmap = result.scalar_one_or_none()
    if not mindmap:
        raise NotFoundException("Mind map not found")
    return mindmap


@router.delete("/{mindmap_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mindmap(mindmap_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(MindMap).where(MindMap.id == mindmap_id, MindMap.user_id == current_user.id)
    )
    mindmap = result.scalar_one_or_none()
    if not mindmap:
        raise NotFoundException("Mind map not found")
    await db.delete(mindmap)
    await db.commit()
