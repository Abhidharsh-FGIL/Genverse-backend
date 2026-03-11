import asyncio
import json
import urllib.parse
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, status, Query, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.dependencies import DBSession, CurrentUser
from app.models.content import Ebook, Audiobook
from app.models.subscription import PointCost
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
    resolved_size = payload.book_size or "short"
    chapter_range = BOOK_SIZE_CHAPTER_RANGES.get(resolved_size, (3, 5))

    # Calculate page count from actual chapters (~3 pages per chapter)
    PAGES_PER_CHAPTER = {"short": 3, "medium": 3, "large": 3}
    num_chapters = len(payload.chapters) if payload.chapters else chapter_range[0]
    resolved_page_count = payload.page_count
    if resolved_page_count is None:
        resolved_page_count = num_chapters * PAGES_PER_CHAPTER.get(resolved_size, 3)

    points_service = PointsService()
    # Gate medium/large books by plan
    if resolved_size == "large":
        await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ebook_large", db=db)
    elif resolved_size == "medium":
        await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ebook_medium", db=db)
    # Look up per-page cost from DB (seed value = 1 pt/page)
    pc_result = await db.execute(select(PointCost).where(PointCost.action == "generate_ebook"))
    pc = pc_result.scalars().first()
    per_page_cost = pc.cost if pc else 1
    cost = resolved_page_count * per_page_cost
    await points_service.deduct_custom(
        user_id=current_user.id,
        action="generate_ebook",
        db=db,
        cost_override=cost,
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


@router.post("/generate-stream")
@router.post("/generate-stream/", include_in_schema=False)
async def generate_ebook_stream(
    payload: EbookGenerateRequest,
    current_user: CurrentUser,
    db: DBSession,
    request: Request,
):
    """Generate eBook content via SSE — streams real progress events to avoid gateway timeouts."""
    resolved_size = payload.book_size or "short"
    chapter_range = BOOK_SIZE_CHAPTER_RANGES.get(resolved_size, (3, 5))

    PAGES_PER_CHAPTER = {"short": 3, "medium": 3, "large": 3}
    num_chapters = len(payload.chapters) if payload.chapters else chapter_range[0]
    resolved_page_count = payload.page_count
    if resolved_page_count is None:
        resolved_page_count = num_chapters * PAGES_PER_CHAPTER.get(resolved_size, 3)

    # ── Do ALL database work BEFORE streaming ──────────────────────────────
    # (FastAPI closes the DB session once we return the StreamingResponse)
    points_service = PointsService()
    if resolved_size == "large":
        await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ebook_large", db=db)
    elif resolved_size == "medium":
        await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ebook_medium", db=db)

    pc_result = await db.execute(select(PointCost).where(PointCost.action == "generate_ebook"))
    pc = pc_result.scalars().first()
    per_page_cost = pc.cost if pc else 1
    cost = resolved_page_count * per_page_cost
    await points_service.deduct_custom(user_id=current_user.id, action="generate_ebook", db=db, cost_override=cost)

    # Pre-build outline
    outline = payload.outline
    if not outline and payload.chapters:
        outline = [
            ch.title + (f": {ch.description}" if ch.description else "")
            for ch in payload.chapters
        ]

    # Pre-serialize chapter dicts
    chapters_dicts = [ch.model_dump() for ch in payload.chapters] if payload.chapters else None
    assessment_dict = payload.assessment_config.model_dump() if payload.assessment_config else None

    # Capture values needed inside the generator (session/user may be gone once streaming starts)
    user_id = current_user.id
    org_id_val = payload.org_id

    # ── Stream the AI generation (no DB needed) ───────────────────────────
    # Progress: 10 points_done → 15 structure → 20..55 chapters → 60 chapters_done → 65..85 images → 92 finalizing → 100 complete
    total_ch = num_chapters

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'stage': 'points_done', 'progress': 10, 'message': 'Points deducted successfully'})}\n\n"

            yield f"data: {json.dumps({'stage': 'writing', 'progress': 15, 'message': f'Writing {total_ch} chapters in parallel...'})}\n\n"

            # Callback: each chapter completion pushes an SSE event via queue
            progress_queue: asyncio.Queue = asyncio.Queue()

            async def on_chapter_done(ch_num: int, ch_total: int):
                # Map chapter progress to 15–55 range
                pct = 15 + int((ch_num / ch_total) * 40)
                await progress_queue.put({
                    "stage": "chapter_done",
                    "progress": pct,
                    "message": f"Chapter {ch_num}/{ch_total} written",
                    "chapter_number": ch_num,
                    "total_chapters": ch_total,
                })

            # Run generation in a background task so we can drain the queue
            ai = AIService()
            gen_task = asyncio.create_task(ai.generate_ebook_content_only(
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
                chapters=chapters_dicts,
                assessment_config=assessment_dict,
                on_chapter_done=on_chapter_done,
            ))

            # Drain progress events while generation is running
            while not gen_task.done():
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=5.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive (prevents proxy timeouts)
                    yield ": heartbeat\n\n"

            # Drain any remaining events in the queue
            while not progress_queue.empty():
                event = progress_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"

            # Get the result (will re-raise if generation failed)
            ebook_data = await gen_task

            yield f"data: {json.dumps({'stage': 'chapters_done', 'progress': 58, 'message': 'All chapters written successfully'})}\n\n"

            # Generate images
            image_density = payload.image_density or "standard"
            if image_density != "minimal":
                yield f"data: {json.dumps({'stage': 'images', 'progress': 62, 'message': 'Creating images...'})}\n\n"
                try:
                    generated_chapters = ebook_data.get("chapters", [])
                    images = await ai.generate_ebook_images(
                        title=payload.title,
                        chapters=generated_chapters,
                        image_density=image_density,
                        image_types=payload.image_types,
                        subject=payload.topic or payload.subject,
                        grade=payload.grade,
                        tone=payload.tone or "academic",
                    )
                    ebook_data["images"] = images
                except Exception as e:
                    print(f"[EbookImage] Image generation error: {e}")

                yield f"data: {json.dumps({'stage': 'images_done', 'progress': 85, 'message': 'Images created'})}\n\n"
            else:
                yield f"data: {json.dumps({'stage': 'images_done', 'progress': 85, 'message': 'Skipping images (minimal density)'})}\n\n"

            yield f"data: {json.dumps({'stage': 'finalizing', 'progress': 92, 'message': 'Saving eBook...'})}\n\n"

            # Save to DB server-side (avoids sending huge base64 images over SSE)
            from app.database import AsyncSessionLocal
            saved_id = None
            try:
                async with AsyncSessionLocal() as save_db:
                    ebook_json_with_config = {
                        **ebook_data,
                        "config": {
                            "author": payload.author,
                            "bookSize": resolved_size,
                            "tone": payload.tone or "academic",
                            "imageDensity": payload.image_density or "standard",
                            "imageTypes": payload.image_types,
                            "assessmentConfig": assessment_dict,
                        },
                    }
                    ebook = Ebook(
                        user_id=user_id,
                        org_id=_parse_org_id(org_id_val),
                        title=payload.title,
                        subject=payload.topic or payload.subject,
                        grade=payload.grade,
                        language=payload.language,
                        source_type=payload.source_type or "topic",
                        ebook_json=ebook_json_with_config,
                        page_count=resolved_page_count,
                        points_used=cost,
                    )
                    save_db.add(ebook)
                    await save_db.commit()
                    await save_db.refresh(ebook)
                    saved_id = str(ebook.id)
                    print(f"[Ebook] Saved to DB: {saved_id}", flush=True)
            except Exception as save_err:
                print(f"[Ebook] DB save failed: {save_err}", flush=True)

            result = {
                "stage": "complete",
                "progress": 100,
                "message": "eBook generated successfully",
                "saved_id": saved_id,
                "page_count": resolved_page_count,
                "points_used": cost,
            }
            yield f"data: {json.dumps(result)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'progress': 0, 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/", response_model=EbookResponse, status_code=status.HTTP_201_CREATED)
async def save_ebook(payload: EbookCreateRequest, current_user: CurrentUser, db: DBSession):
    """Persist a generated eBook to the database."""
    resolved_page_count = payload.page_count or BOOK_SIZE_PAGE_COUNTS.get("short", 15)
    cost = payload.points_used or (resolved_page_count * 5)

    ebook = Ebook(
        user_id=current_user.id,
        org_id=_parse_org_id(payload.org_id),
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

    await db.commit()
    await db.refresh(ebook)
    return ebook


@router.get("/", response_model=list[EbookResponse])
async def list_ebooks(current_user: CurrentUser, db: DBSession, org_id: str | None = Query(None)):
    q = select(Ebook).where(Ebook.user_id == current_user.id)
    if org_id is not None:
        q = _apply_org_filter(q, Ebook.org_id, org_id)
    q = q.order_by(Ebook.created_at.desc())
    result = await db.execute(q)
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

    # Check usage + deduct download cost
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ebook_pdf", db=db)
    await points_service.deduct(user_id=current_user.id, action="ebook_download_pdf", db=db)

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

    # Check usage + deduct download cost
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ebook_docx", db=db)
    await points_service.deduct(user_id=current_user.id, action="ebook_download_docx", db=db)

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

    # Check feature access, then deduct points
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ebook_audio", db=db)
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
