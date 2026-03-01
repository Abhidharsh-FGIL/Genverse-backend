import urllib.parse
import uuid

from fastapi import APIRouter, status, Query
from fastapi.responses import Response
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.dependencies import DBSession, CurrentUser
from app.models.content import Ebook, Audiobook
from app.schemas.content import (
    EbookGenerateRequest, EbookResponse, AudiobookGenerateRequest, AudiobookResponse,
    AudiobookVoicesResponse,
    EbookOutlineRequest, EbookOutlineResponse, EbookGeneratedContent, EbookCreateRequest,
)
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.points_service import PointsService
from app.services.ebook_export_service import generate_pdf, generate_docx
from app.services.audiobook_service import get_available_voices

router = APIRouter()

BOOK_SIZE_PAGE_COUNTS = {
    "short": 15,
    "medium": 30,
    "large": 60,
}

# Chapter count ranges per book size: (min_chapters, max_chapters)
BOOK_SIZE_CHAPTER_RANGES = {
    "short": (3, 5),
    "medium": (6, 10),
    "large": (11, 20),
}


@router.post("/outline", response_model=EbookOutlineResponse)
async def generate_outline(payload: EbookOutlineRequest, current_user: CurrentUser):
    """Generate a chapter outline (titles + descriptions) using AI. No points deducted."""
    resolved_size = payload.book_size or "short"
    chapter_range = BOOK_SIZE_CHAPTER_RANGES.get(resolved_size, (3, 5))

    ai = AIService()
    chapters = await ai.generate_ebook_outline(
        title=payload.title,
        topic=payload.topic,
        subject=payload.subject,
        language=payload.language,
        chapter_range=chapter_range,
        tone=payload.tone or "academic",
    )
    return EbookOutlineResponse(chapters=chapters)


@router.post("/generate", response_model=EbookGeneratedContent)
async def generate_ebook(payload: EbookGenerateRequest, current_user: CurrentUser, db: DBSession):
    """Generate eBook content using AI and deduct points. Does NOT save to DB — call POST / to save."""
    resolved_page_count = payload.page_count
    resolved_size = payload.book_size or "short"
    if resolved_page_count is None:
        resolved_page_count = BOOK_SIZE_PAGE_COUNTS.get(resolved_size, 15)
    chapter_range = BOOK_SIZE_CHAPTER_RANGES.get(resolved_size, (3, 5))

    points_service = PointsService()
    cost = resolved_page_count * 5
    await points_service.deduct_custom(
        user_id=current_user.id,
        action="generate_ebook",
        db=db,
    )

    # Convert structured chapters to flat outline if no legacy outline provided
    outline = payload.outline
    if not outline and payload.chapters:
        outline = [
            ch.title + (f": {ch.description}" if ch.description else "")
            for ch in payload.chapters
        ]

    ai = AIService()
    ebook_json = await ai.generate_ebook(
        title=payload.title,
        author=payload.author or "",
        subject=payload.topic or payload.subject,
        grade=payload.grade,
        language=payload.language,
        source_type=payload.source_type,
        outline=outline,
        page_count=resolved_page_count,
        chapter_range=chapter_range,
        tone=payload.tone or "academic",
        book_size=resolved_size,
        chapters=[ch.model_dump() for ch in payload.chapters] if payload.chapters else None,
        image_density=payload.image_density or "standard",
        image_types=payload.image_types,
        assessment_config=payload.assessment_config.model_dump() if payload.assessment_config else None,
    )

    return EbookGeneratedContent(ebook_json=ebook_json, page_count=resolved_page_count, points_used=cost)


@router.post("/", response_model=EbookResponse, status_code=status.HTTP_201_CREATED)
async def save_ebook(payload: EbookCreateRequest, current_user: CurrentUser, db: DBSession):
    """Persist a generated eBook to the database."""
    resolved_page_count = payload.page_count or BOOK_SIZE_PAGE_COUNTS.get("short", 15)
    cost = payload.points_used or (resolved_page_count * 5)

    ebook = Ebook(
        user_id=current_user.id,
        title=payload.title,
        subject=payload.subject,
        grade=payload.grade,
        language=payload.language,
        source_type=payload.source_type,
        source_ref_id=payload.source_ref_id,
        ebook_json=payload.ebook_json,
        page_count=resolved_page_count,
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


@router.get("/{ebook_id}/download/pdf")
async def download_ebook_pdf(ebook_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Generate and stream a professional PDF for the given eBook."""
    result = await db.execute(
        select(Ebook).where(Ebook.id == ebook_id, Ebook.user_id == current_user.id)
    )
    ebook = result.scalar_one_or_none()
    if not ebook:
        raise NotFoundException("eBook not found")

    pdf_bytes = await run_in_threadpool(generate_pdf, ebook.ebook_json, ebook.title)
    safe_name = urllib.parse.quote(ebook.title, safe="")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{safe_name}.pdf",
            "Content-Length": str(len(pdf_bytes)),
        },
    )


@router.get("/{ebook_id}/download/doc")
async def download_ebook_doc(ebook_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Generate and stream a professional DOCX for the given eBook."""
    result = await db.execute(
        select(Ebook).where(Ebook.id == ebook_id, Ebook.user_id == current_user.id)
    )
    ebook = result.scalar_one_or_none()
    if not ebook:
        raise NotFoundException("eBook not found")

    docx_bytes = await run_in_threadpool(generate_docx, ebook.ebook_json, ebook.title)
    safe_name = urllib.parse.quote(ebook.title, safe="")
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{safe_name}.docx",
            "Content-Length": str(len(docx_bytes)),
        },
    )


@router.get("/{ebook_id}/audiobooks", response_model=list[AudiobookResponse])
async def list_audiobooks(ebook_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """List audiobooks for a given eBook."""
    result = await db.execute(
        select(Audiobook).where(Audiobook.ebook_id == ebook_id, Audiobook.user_id == current_user.id)
    )
    return result.scalars().all()


@router.get("/{ebook_id}/audiobook", response_model=AudiobookResponse | None)
async def get_audiobook(ebook_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Get the audiobook for a given eBook."""
    result = await db.execute(
        select(Audiobook).where(Audiobook.ebook_id == ebook_id, Audiobook.user_id == current_user.id)
    )
    audiobook = result.scalar_one_or_none()
    if not audiobook:
        raise NotFoundException("Audiobook not found")
    return audiobook


@router.get("/{ebook_id}/download/audio")
async def download_ebook_audio(ebook_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Stream the generated audiobook MP3 for the given eBook."""
    from pathlib import Path

    result = await db.execute(
        select(Audiobook).where(Audiobook.ebook_id == ebook_id, Audiobook.user_id == current_user.id)
    )
    audiobook = result.scalar_one_or_none()
    if not audiobook or not audiobook.audio_path:
        raise NotFoundException("Audiobook not found. Generate audio first.")

    audio_file = Path(audiobook.audio_path)
    if not audio_file.exists():
        raise NotFoundException("Audio file not found on disk.")

    audio_bytes = audio_file.read_bytes()
    # Get the ebook title for the filename
    ebook_result = await db.execute(
        select(Ebook).where(Ebook.id == ebook_id)
    )
    ebook = ebook_result.scalar_one_or_none()
    safe_name = urllib.parse.quote(ebook.title if ebook else "audiobook", safe="")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{safe_name}.mp3",
            "Content-Length": str(len(audio_bytes)),
        },
    )


@router.get("/voices/{language}", response_model=AudiobookVoicesResponse)
async def list_voices(language: str, current_user: CurrentUser):
    """List available neural voice profiles for a given language."""
    voices = get_available_voices(language)
    return AudiobookVoicesResponse(language=language, voices=voices)


@router.post("/{ebook_id}/audiobook", response_model=AudiobookResponse)
async def generate_audiobook(
    ebook_id: uuid.UUID,
    payload: AudiobookGenerateRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate an industry-grade audiobook with neural TTS, chapter-aware narration, and timestamps."""
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
        narration_style=payload.narration_style or "standard",
    )

    # Check if an audiobook already exists for this ebook (unique constraint on ebook_id)
    existing_result = await db.execute(
        select(Audiobook).where(Audiobook.ebook_id == ebook_id)
    )
    audiobook = existing_result.scalar_one_or_none()

    if audiobook:
        # Update existing record
        audiobook.audio_path = audio_data.get("audio_path")
        audiobook.language = payload.language
        audiobook.voice_profile = payload.voice_profile
        audiobook.narration_style = payload.narration_style or "standard"
        audiobook.duration_seconds = audio_data.get("duration_seconds")
        audiobook.chapter_timestamps = audio_data.get("chapter_timestamps")
    else:
        # Create new record
        audiobook = Audiobook(
            ebook_id=ebook_id,
            user_id=current_user.id,
            audio_path=audio_data.get("audio_path"),
            language=payload.language,
            voice_profile=payload.voice_profile,
            narration_style=payload.narration_style or "standard",
            duration_seconds=audio_data.get("duration_seconds"),
            chapter_timestamps=audio_data.get("chapter_timestamps"),
        )
        db.add(audiobook)

    await db.commit()
    await db.refresh(audiobook)
    return audiobook
