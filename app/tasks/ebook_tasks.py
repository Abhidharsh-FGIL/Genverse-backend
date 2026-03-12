"""
Celery task for eBook generation.

Offloads heavy LLM calls (chapter generation, image generation, assessments)
to a Celery worker process. Progress events are published to a Redis pub/sub
channel so the FastAPI SSE endpoint can forward them to the client in real time.

Start the worker:
    celery -A app.celery_app worker --loglevel=info --concurrency=4
"""
import asyncio
import json
import uuid

import redis as sync_redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.celery_app import celery
from app.config import settings

# Sync DB engine for Celery worker (same pattern as library_tasks)
_sync_engine = create_engine(
    settings.sync_database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
SyncSessionLocal = sessionmaker(bind=_sync_engine, expire_on_commit=False)


def _publish(r: sync_redis.Redis, channel: str, event: dict):
    """Publish a progress event to the Redis pub/sub channel."""
    r.publish(channel, json.dumps(event))


def _run_async(coro):
    """Run an async coroutine from sync Celery context."""
    return asyncio.run(coro)


@celery.task(bind=True, name="generate_ebook_task", max_retries=0)
def generate_ebook_task(self, params: dict):
    """
    Generate an eBook in a Celery worker.

    params dict keys:
        title, author, subject, grade, language, source_type, outline,
        page_count, chapter_range, tone, book_size, chapters,
        assessment_config, image_density, image_types,
        user_id, org_id, cost
    """
    task_id = self.request.id
    channel = f"ebook:progress:{task_id}"
    r = sync_redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    try:
        _run_async(_generate_ebook_async(params, channel, r))
    except Exception as exc:
        _publish(r, channel, {
            "stage": "error",
            "progress": 0,
            "message": str(exc),
        })
    finally:
        # Sentinel to signal SSE stream to close
        _publish(r, channel, {"stage": "__DONE__"})
        r.close()

    return {"status": "dispatched", "task_id": task_id}


async def _generate_ebook_async(params: dict, channel: str, r: sync_redis.Redis):
    """Async ebook generation — runs inside asyncio.run() in the Celery worker."""
    from app.services.ai_service import AIService
    from app.models.content import Ebook

    ai = AIService()
    try:
        await _do_generate(ai, params, channel, r)
    finally:
        # Explicitly close the async OpenAI/httpx client before asyncio.run()
        # shuts down the event loop — prevents "Event loop is closed" errors.
        if ai._openai_client:
            await ai._openai_client.close()
            ai._openai_client = None


async def _do_generate(ai, params: dict, channel: str, r: sync_redis.Redis):
    """Inner generation logic — separated so _generate_ebook_async can do cleanup."""
    from app.models.content import Ebook

    chapter_range = tuple(params["chapter_range"])
    user_id = uuid.UUID(params["user_id"])
    org_id = uuid.UUID(params["org_id"]) if params.get("org_id") else None
    cost = params["cost"]
    page_count = params["page_count"]

    # ── Callback: publish chapter progress to Redis ──
    async def on_chapter_done(ch_num: int, ch_total: int):
        pct = 15 + int((ch_num / ch_total) * 40)
        _publish(r, channel, {
            "stage": "chapter_done",
            "progress": pct,
            "message": f"Chapter {ch_num}/{ch_total} written",
            "chapter_number": ch_num,
            "total_chapters": ch_total,
        })

    _publish(r, channel, {
        "stage": "writing",
        "progress": 15,
        "message": f"Writing {len(params.get('chapters') or params.get('outline') or [])} chapters in parallel...",
    })

    # ── Generate content (no images) ──
    ebook_data = await ai.generate_ebook_content_only(
        title=params["title"],
        author=params.get("author", ""),
        subject=params.get("subject"),
        grade=params.get("grade"),
        language=params["language"],
        source_type=params.get("source_type", "topic"),
        outline=params.get("outline"),
        page_count=page_count,
        chapter_range=chapter_range,
        tone=params.get("tone", "academic"),
        book_size=params.get("book_size", "short"),
        chapters=params.get("chapters"),
        assessment_config=params.get("assessment_config"),
        on_chapter_done=on_chapter_done,
    )

    _publish(r, channel, {
        "stage": "chapters_done",
        "progress": 58,
        "message": "All chapters written successfully",
    })

    # ── Generate images ──
    image_density = params.get("image_density", "standard")
    if image_density != "minimal":
        _publish(r, channel, {
            "stage": "images",
            "progress": 62,
            "message": "Creating images...",
        })
        try:
            generated_chapters = ebook_data.get("chapters", [])
            images = await ai.generate_ebook_images(
                title=params["title"],
                chapters=generated_chapters,
                image_density=image_density,
                image_types=params.get("image_types"),
                subject=params.get("subject"),
                grade=params.get("grade"),
                tone=params.get("tone", "academic"),
            )
            ebook_data["images"] = images
        except Exception as e:
            print(f"[EbookTask] Image generation error: {e}", flush=True)

        _publish(r, channel, {
            "stage": "images_done",
            "progress": 85,
            "message": "Images created",
        })
    else:
        _publish(r, channel, {
            "stage": "images_done",
            "progress": 85,
            "message": "Skipping images (minimal density)",
        })

    # ── Save to DB (sync session in Celery worker) ──
    _publish(r, channel, {
        "stage": "finalizing",
        "progress": 92,
        "message": "Saving eBook...",
    })

    ebook_json_with_config = {
        **ebook_data,
        "config": {
            "author": params.get("author"),
            "bookSize": params.get("book_size", "short"),
            "tone": params.get("tone", "academic"),
            "imageDensity": image_density,
            "imageTypes": params.get("image_types"),
            "assessmentConfig": params.get("assessment_config"),
        },
    }

    db = SyncSessionLocal()
    saved_id = None
    try:
        ebook = Ebook(
            user_id=user_id,
            org_id=org_id,
            title=params["title"],
            subject=params.get("subject") or params.get("topic"),
            grade=params.get("grade"),
            language=params["language"],
            source_type=params.get("source_type", "topic"),
            ebook_json=ebook_json_with_config,
            page_count=page_count,
            points_used=cost,
        )
        db.add(ebook)
        db.commit()
        db.refresh(ebook)
        saved_id = str(ebook.id)
        print(f"[EbookTask] Saved to DB: {saved_id}", flush=True)
    except Exception as save_err:
        db.rollback()
        print(f"[EbookTask] DB save failed: {save_err}", flush=True)
    finally:
        db.close()

    _publish(r, channel, {
        "stage": "complete",
        "progress": 100,
        "message": "eBook generated successfully",
        "saved_id": saved_id,
        "page_count": page_count,
        "points_used": cost,
    })
