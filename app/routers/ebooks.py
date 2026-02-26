import uuid
from fastapi import APIRouter, status, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.content import Ebook, Audiobook
from app.schemas.content import (
    EbookGenerateRequest, EbookResponse, AudiobookGenerateRequest, AudiobookResponse
)
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()


@router.post("/generate", response_model=EbookResponse, status_code=status.HTTP_201_CREATED)
async def generate_ebook(payload: EbookGenerateRequest, current_user: CurrentUser, db: DBSession):
    """Generate a structured eBook using AI. Cost: 5 pts per page."""
    points_service = PointsService()
    cost = payload.page_count * 5
    await points_service.deduct_custom(
        user_id=current_user.id,
        action="generate_ebook",
        db=db,
    )

    ai = AIService()
    ebook_json = await ai.generate_ebook(
        title=payload.title,
        subject=payload.subject,
        grade=payload.grade,
        language=payload.language,
        source_type=payload.source_type,
        outline=payload.outline,
        page_count=payload.page_count,
    )

    ebook = Ebook(
        user_id=current_user.id,
        title=payload.title,
        subject=payload.subject,
        grade=payload.grade,
        language=payload.language,
        source_type=payload.source_type,
        source_ref_id=payload.source_ref_id,
        ebook_json=ebook_json,
        page_count=payload.page_count,
        points_used=cost,
    )
    db.add(ebook)

    # Award XP
    current_user.xp = (current_user.xp or 0) + 25

    await db.commit()
    await db.refresh(ebook)
    return ebook


@router.get("/", response_model=list[EbookResponse])
async def list_ebooks(current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(Ebook)
        .where(Ebook.user_id == current_user.id)
        .order_by(Ebook.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{ebook_id}", response_model=EbookResponse)
async def get_ebook(ebook_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(Ebook).where(Ebook.id == ebook_id, Ebook.user_id == current_user.id)
    )
    ebook = result.scalar_one_or_none()
    if not ebook:
        raise NotFoundException("eBook not found")
    return ebook


@router.delete("/{ebook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ebook(ebook_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(Ebook).where(Ebook.id == ebook_id, Ebook.user_id == current_user.id)
    )
    ebook = result.scalar_one_or_none()
    if not ebook:
        raise NotFoundException("eBook not found")
    await db.delete(ebook)
    await db.commit()


@router.post("/{ebook_id}/audiobook", response_model=AudiobookResponse)
async def generate_audiobook(
    ebook_id: uuid.UUID,
    payload: AudiobookGenerateRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate an audio version of an eBook."""
    ebook_result = await db.execute(
        select(Ebook).where(Ebook.id == ebook_id, Ebook.user_id == current_user.id)
    )
    ebook = ebook_result.scalar_one_or_none()
    if not ebook:
        raise NotFoundException("eBook not found")

    # Deduct points
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="generate_audiobook", db=db)

    ai = AIService()
    audio_data = await ai.generate_audiobook(
        ebook_json=ebook.ebook_json,
        language=payload.language,
        voice_profile=payload.voice_profile,
    )

    audiobook = Audiobook(
        ebook_id=ebook_id,
        user_id=current_user.id,
        audio_path=audio_data.get("audio_path"),
        language=payload.language,
        voice_profile=payload.voice_profile,
        duration_seconds=audio_data.get("duration_seconds"),
    )
    db.add(audiobook)
    await db.commit()
    await db.refresh(audiobook)
    return audiobook
