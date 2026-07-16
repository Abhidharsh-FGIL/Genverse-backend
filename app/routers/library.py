import asyncio
import json
import uuid
import logging
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, BackgroundTasks, status, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, delete as sql_delete, func as sa_func

logger = logging.getLogger(__name__)

from app.dependencies import DBSession, CurrentUser
from app.core.security import verify_access_token
from app.models.content import UserLibraryItem, DocChunk
from app.models.subscription import Subscription, PlanDefinition
from app.schemas.content import LibraryItemResponse, LibraryItemListResponse, LibraryItemUpdate, VaultQueryRequest, VaultQueryResponse
from app.core.exceptions import NotFoundException
from app.services.storage_service import StorageService
from app.services.ai_service import AIService, get_ai_service
from app.services.faiss_service import FAISSService
from app.services.points_service import PointsService
from app.tasks.library_tasks import process_vault_embeddings, delete_vault_embeddings
from app.config import settings

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


def _faiss() -> FAISSService:
    return FAISSService(settings.STORAGE_ROOT)


def _publish_vault_event_sync(user_id: str, event: dict) -> None:
    try:
        import redis as sync_redis
        r = sync_redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        channel = f"vault:events:{user_id}"
        r.publish(channel, json.dumps(event))
    except Exception as exc:
        logger.warning("[Vault] Failed to publish SSE event for user=%s: %s", user_id, exc)


async def _process_vault_item_bg(item_id: str, user_id: str, filename: str) -> None:
    """Process a vault item inline as a FastAPI background task.

    Runs after the HTTP response is sent. Uses the async DB session and
    async AI service — no Celery or Redis queue required.
    """
    from app.database import AsyncSessionLocal
    print(f"[BG] Starting processing for '{filename}' (item={item_id}, user={user_id})")

    async with AsyncSessionLocal() as db:
        try:
            item = await db.get(UserLibraryItem, uuid.UUID(item_id))
            if not item:
                print(f"[BG] Item {item_id} not found in DB, aborting")
                return
            if item.processing_status == "ready":
                print(f"[BG] '{filename}' already ready, skipping")
                return

            ai = get_ai_service()
            faiss_svc = _faiss()

            # Remove any stale chunks from a previous failed attempt
            old_chunks_result = await db.execute(
                select(DocChunk.id).where(DocChunk.library_item_id == item.id)
            )
            old_chunk_ids = [str(r[0]) for r in old_chunks_result.all()]
            if old_chunk_ids:
                print(f"[BG] Removing {len(old_chunk_ids)} stale chunk(s) for '{filename}'")
                faiss_svc.remove_chunks(user_id=user_id, chunk_ids=set(old_chunk_ids))
                await db.execute(sql_delete(DocChunk).where(DocChunk.library_item_id == item.id))
                await db.flush()

            # 1. Extract text
            print(f"[BG] Extracting text from '{filename}' (path: {item.storage_path})")
            extracted_text = await ai.extract_text_from_file(item.storage_path)
            if not extracted_text:
                item.processing_status = "failed"
                await db.commit()
                print(f"[BG] No text extracted from '{filename}', marked as failed")
                await asyncio.to_thread(_publish_vault_event_sync, user_id, {
                    "type": "item_status", "item_id": item_id, "processing_status": "failed",
                })
                return

            print(f"[BG] Extracted {len(extracted_text)} chars from '{filename}'")

            # 2. Semantic chunking
            chunks = ai.semantic_chunk_text(extracted_text)
            print(f"[BG] '{filename}' → {len(chunks)} chunks")

            # 3. Embed each chunk
            chunk_ids: list[str] = []
            embeddings: list[list[float]] = []
            for i, chunk_text in enumerate(chunks):
                doc_chunk = DocChunk(
                    library_item_id=item.id,
                    chunk_text=chunk_text,
                    chunk_order=i,
                )
                db.add(doc_chunk)
                await db.flush()

                embedding = await ai.generate_embedding(chunk_text)
                if embedding:
                    chunk_ids.append(str(doc_chunk.id))
                    embeddings.append(embedding)

            # 4. Add to FAISS
            if chunk_ids:
                faiss_svc.add_batch(user_id=user_id, chunk_ids=chunk_ids, embeddings=embeddings)

            # 5. Mark ready
            item.is_processed = True
            item.extracted_text_ref = "processed"
            item.processing_status = "ready"
            await db.commit()

            print(f"[BG] '{filename}' done — {len(chunk_ids)} embeddings stored")
            await asyncio.to_thread(_publish_vault_event_sync, user_id, {
                "type": "item_status",
                "item_id": item_id,
                "processing_status": "ready",
                "is_processed": True,
                "extracted_text_ref": "processed",
                "chunks_embedded": len(chunk_ids),
            })

        except Exception as exc:
            logger.exception("[BG] Failed processing '%s' (item=%s): %s", filename, item_id, exc)
            print(f"[BG] ERROR processing '{filename}': {exc}")
            try:
                item = await db.get(UserLibraryItem, uuid.UUID(item_id))
                if item:
                    item.processing_status = "failed"
                    await db.commit()
                    await asyncio.to_thread(_publish_vault_event_sync, user_id, {
                        "type": "item_status", "item_id": item_id, "processing_status": "failed",
                    })
            except Exception:
                pass


@router.get("/storage-usage")
async def get_storage_usage(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str = Query(None),
):
    """Return total storage used in MB by summing file_size_mb of all library items."""
    parsed_org = _parse_org_id(org_id)
    q = select(sa_func.coalesce(sa_func.sum(UserLibraryItem.file_size_mb), 0)).where(
        UserLibraryItem.user_id == current_user.id
    )
    if parsed_org is None:
        q = q.where(UserLibraryItem.org_id.is_(None))
    else:
        q = q.where(UserLibraryItem.org_id == parsed_org)
    result = await db.execute(q)
    total_mb = result.scalar() or 0
    return {"storage_used_mb": round(float(total_mb), 2)}


@router.post("/upload", response_model=LibraryItemResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    folder: str = Form(None),
    tags: str = Form(None),
    org_id: str = Form(None),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Upload a document to the knowledge vault. Text extraction + embedding happens inline."""
    # Enforce plan-based file size limit
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    await file.seek(0)  # reset for StorageService

    parsed_oid = _parse_org_id(org_id)
    if parsed_oid:
        sub_q = select(Subscription).where(
            Subscription.org_id == parsed_oid,
            Subscription.status.in_(["active", "trialing"]),
        )
    else:
        sub_q = select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.workspace_type == "individual",
            Subscription.status.in_(["active", "trialing"]),
        )
    sub_result = await db.execute(sub_q.order_by(Subscription.updated_at.desc()))
    sub = sub_result.scalars().first()
    plan_name = sub.plan if sub else "free"

    plan_def_result = await db.execute(
        select(PlanDefinition).where(PlanDefinition.plan == plan_name)
    )
    plan_def = plan_def_result.scalars().first()
    if not plan_def:
        logger.warning("PlanDefinition not found for plan=%s, using default max_file_mb=5", plan_name)
    max_file_mb = plan_def.max_file_size_mb if plan_def else 5

    if file_size_mb > max_file_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "max_size_mb": max_file_mb,
                "file_size_mb": round(file_size_mb, 1),
                "message": f"File is {round(file_size_mb, 1)} MB. Your plan allows up to {max_file_mb} MB per file.",
            },
        )

    storage = StorageService()
    file_info = await storage.upload_file(
        file=file,
        bucket="user-library",
        prefix=str(current_user.id),
    )

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    item = UserLibraryItem(
        user_id=current_user.id,
        org_id=_parse_org_id(org_id),
        title=file.filename or "Untitled",
        file_type=file_info.get("type"),
        storage_path=file_info.get("path"),
        file_size_mb=file_info.get("size_mb"),
        folder=folder,
        tags=tag_list,
        is_processed=False,
        processing_status="processing",
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)

    # Try Celery first; fall back to an in-process BackgroundTask if the
    # broker is unavailable or the dispatch throws for any reason.
    _dispatched_via_celery = False
    try:
        process_vault_embeddings.delay(str(item.id), str(current_user.id), item.title)
        _dispatched_via_celery = True
        print(f"[Upload] Celery task queued for '{item.title}' (item_id={item.id})")
    except Exception as exc:
        print(f"[Upload] Celery dispatch failed ({exc!r}), falling back to BackgroundTask")

    if not _dispatched_via_celery:
        background_tasks.add_task(
            _process_vault_item_bg,
            str(item.id),
            str(current_user.id),
            item.title,
        )

    return item


@router.get("/", response_model=LibraryItemListResponse)
async def list_library_items(
    current_user: CurrentUser,
    db: DBSession,
    folder: str | None = Query(None),
    org_id: str | None = Query(None),
    limit: int = Query(12, ge=1, le=50),
    cursor: str | None = Query(None, description="ISO datetime cursor for pagination"),
):
    from datetime import datetime as dt
    from sqlalchemy import func as sa_func

    # Include all items including failed ones so users can see and retry them.
    base = select(UserLibraryItem).where(
        UserLibraryItem.user_id == current_user.id,
    )
    if org_id is not None:
        base = _apply_org_filter(base, UserLibraryItem.org_id, org_id)
    if folder:
        base = base.where(UserLibraryItem.folder == folder)
    else:
        base = base.where(
            (UserLibraryItem.folder != "ocr") | (UserLibraryItem.folder.is_(None))
        )

    # Total count
    count_q = select(sa_func.count()).select_from(base.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    # Apply cursor
    q = base
    if cursor:
        try:
            cursor_dt = dt.fromisoformat(cursor)
            q = q.where(UserLibraryItem.created_at < cursor_dt)
        except (ValueError, TypeError):
            pass

    q = q.order_by(UserLibraryItem.created_at.desc()).limit(limit + 1)
    result = await db.execute(q)
    rows = result.scalars().all()

    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].created_at.isoformat() if has_more and items else None

    return LibraryItemListResponse(items=items, next_cursor=next_cursor, has_more=has_more, total=total)


@router.get("/events")
async def vault_events_stream(token: str = Query(..., description="Bearer JWT (EventSource cannot set headers)")):
    """Server-Sent Events stream of Knowledge Vault status changes for the
    authenticated user. The Celery worker (`process_vault_embeddings`) publishes
    `item_status` events to a per-user Redis channel when an item flips from
    `processing` → `ready`/`failed`; this endpoint forwards them as SSE.

    The frontend uses this to drive UI updates instead of polling the list
    endpoint, eliminating the steady-state request overhead while files are
    being processed.

    NOTE: auth is taken from `?token=` because the browser EventSource API
    cannot attach an Authorization header. The token is the same access JWT
    used by every other endpoint.
    """
    payload = verify_access_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    user_id = str(payload["sub"])

    import redis.asyncio as aioredis
    aio_redis = aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = aio_redis.pubsub()
    channel = f"vault:events:{user_id}"
    await pubsub.subscribe(channel)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Initial "open" frame so the client knows the stream is live.
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            while True:
                # Block up to 15s waiting for a real event; otherwise emit a
                # heartbeat comment so proxies/load balancers don't kill the
                # idle connection.
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if msg and msg.get("type") == "message":
                    try:
                        event = json.loads(msg["data"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                else:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            # Client disconnected — clean exit.
            pass
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except Exception:
                pass
            try:
                await aio_redis.close()
            except Exception:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# Inline attachment limits — kept intentionally lower than vault upload limits
# so the AI assistant stays fast. Large files should go to the Knowledge Vault.
_INLINE_MAX_BYTES = 8 * 1024 * 1024   # 8 MB hard cap
_INLINE_MAX_WORDS = 6_000             # truncate extracted text after this many words
_INLINE_MAX_PDF_PAGES = 15            # only process first N pages inline


@router.post("/extract-text-inline")
async def extract_text_inline(
    file: UploadFile = File(...),
    current_user: CurrentUser = None,
):
    """Extract text from an uploaded file without saving it anywhere.

    Used by the AI assistant 'From Device' attachment feature so users can ask
    questions about a document without it being stored in the Knowledge Vault.
    Files above 8 MB or beyond 15 PDF pages are rejected / truncated to keep
    extraction fast. Large files should use the Knowledge Vault instead.
    """
    import tempfile
    import os
    from pathlib import Path as _Path

    content = await file.read()
    if len(content) > _INLINE_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is too large for direct AI chat "
                f"({len(content) / (1024*1024):.1f} MB — limit is 8 MB). "
                "Upload it to your Knowledge Vault and select it from there instead."
            ),
        )

    ai = get_ai_service()
    suffix = _Path(file.filename or "file").suffix.lower() or ".tmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = await ai.extract_text_from_file(tmp_path, inline_page_limit=_INLINE_MAX_PDF_PAGES)
    finally:
        os.unlink(tmp_path)

    truncated = False
    words = text.split() if text else []
    if len(words) > _INLINE_MAX_WORDS:
        text = " ".join(words[:_INLINE_MAX_WORDS])
        truncated = True

    return {
        "text": text or "",
        "word_count": len(words[:_INLINE_MAX_WORDS]),
        "filename": file.filename,
        "truncated": truncated,
        "page_limit_hit": not text and bool(content),  # True when PDF was too long
    }


@router.get("/signed-url")
@router.get("/signed-url/", include_in_schema=False)
async def get_signed_url(
    path: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Convert an absolute storage path to a serving URL (local-storage implementation)."""
    from pathlib import Path as _Path
    from app.config import settings as _settings
    try:
        storage_root = _Path(_settings.STORAGE_ROOT).resolve()
        abs_path = _Path(path).resolve()
        rel = abs_path.relative_to(storage_root)
        url = f"/uploads/{rel.as_posix()}"
    except Exception:
        url = None
    return {"url": url}


@router.get("/items", response_model=list[LibraryItemResponse])
async def list_library_items_by_fields(
    current_user: CurrentUser,
    db: DBSession,
    fields: Optional[str] = Query(None),
    folder: Optional[str] = Query(None),
    org_id: str | None = Query(None),
):
    """Alias for GET / that accepts an optional ?fields= param (ignored server-side)."""
    q = select(UserLibraryItem).where(
        UserLibraryItem.user_id == current_user.id,
        UserLibraryItem.processing_status == "ready",
    )
    if org_id is not None:
        q = _apply_org_filter(q, UserLibraryItem.org_id, org_id)
    if folder:
        q = q.where(UserLibraryItem.folder == folder)
    else:
        # Exclude OCR extractions from vault view by default
        q = q.where(
            (UserLibraryItem.folder != "ocr") | (UserLibraryItem.folder.is_(None))
        )
    q = q.order_by(UserLibraryItem.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{item_id}/status")
async def get_library_item_status(item_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Poll processing status of an uploaded vault item."""
    result = await db.execute(
        select(
            UserLibraryItem.processing_status,
            UserLibraryItem.is_processed,
            UserLibraryItem.title,
        ).where(
            UserLibraryItem.id == item_id,
            UserLibraryItem.user_id == current_user.id,
        )
    )
    row = result.one_or_none()
    if not row:
        raise NotFoundException("Library item not found")
    return {
        "id": str(item_id),
        "processing_status": row.processing_status,
        "is_processed": row.is_processed,
        "title": row.title,
    }


@router.post("/{item_id}/reprocess")
async def reprocess_library_item(
    item_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
):
    """Re-trigger embedding for a vault item stuck in 'processing' or 'failed' state.
    Safe to call multiple times — skips items that are already 'ready'.
    """
    result = await db.execute(
        select(UserLibraryItem).where(
            UserLibraryItem.id == item_id,
            UserLibraryItem.user_id == current_user.id,
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise NotFoundException("Library item not found")

    item.processing_status = "processing"
    item.is_processed = False
    await db.commit()
    await db.refresh(item)

    _dispatched_via_celery = False
    try:
        process_vault_embeddings.delay(str(item.id), str(current_user.id), item.title)
        _dispatched_via_celery = True
        print(f"[Reprocess] Celery task queued for '{item.title}' (item_id={item.id})")
    except Exception as exc:
        print(f"[Reprocess] Celery dispatch failed ({exc!r}), falling back to BackgroundTask")

    if not _dispatched_via_celery:
        background_tasks.add_task(
            _process_vault_item_bg,
            str(item.id),
            str(current_user.id),
            item.title,
        )

    return {"status": "queued", "item_id": str(item.id)}


@router.get("/{item_id}", response_model=LibraryItemResponse)
async def get_library_item(item_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(UserLibraryItem).where(
            UserLibraryItem.id == item_id,
            UserLibraryItem.user_id == current_user.id,
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise NotFoundException("Library item not found")
    return item


@router.get("/{item_id}/text")
async def get_library_item_text(item_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Return the full extracted text for a library item by joining its chunks."""
    result = await db.execute(
        select(UserLibraryItem).where(
            UserLibraryItem.id == item_id,
            UserLibraryItem.user_id == current_user.id,
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise NotFoundException("Library item not found")

    chunks_result = await db.execute(
        select(DocChunk)
        .where(DocChunk.library_item_id == item_id)
        .order_by(DocChunk.chunk_order)
    )
    chunks = chunks_result.scalars().all()
    full_text = " ".join(c.chunk_text for c in chunks)
    return {"text": full_text}


@router.patch("/{item_id}", response_model=LibraryItemResponse)
async def update_library_item(
    item_id: uuid.UUID, payload: LibraryItemUpdate, current_user: CurrentUser, db: DBSession
):
    result = await db.execute(
        select(UserLibraryItem).where(
            UserLibraryItem.id == item_id,
            UserLibraryItem.user_id == current_user.id,
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise NotFoundException("Library item not found")

    update_data = payload.model_dump(exclude_unset=True)
    extracted_text = update_data.pop("extracted_text", None)

    for key, value in update_data.items():
        setattr(item, key, value)

    # Re-chunk updated extracted text: remove old FAISS vectors, add new ones
    if extracted_text is not None:
        # Collect old chunk IDs before deletion
        old_chunks_result = await db.execute(
            select(DocChunk.id).where(DocChunk.library_item_id == item_id)
        )
        old_chunk_ids = {str(row[0]) for row in old_chunks_result.all()}

        # Remove old vectors from FAISS
        if old_chunk_ids:
            _faiss().remove_chunks(
                user_id=str(current_user.id),
                chunk_ids=old_chunk_ids,
            )

        # Delete old DB rows
        await db.execute(sql_delete(DocChunk).where(DocChunk.library_item_id == item_id))

        if extracted_text:
            ai = get_ai_service()
            new_chunk_ids: list[str] = []
            new_embeddings: list[list[float]] = []

            for i, chunk_text in enumerate(ai.semantic_chunk_text(extracted_text)):
                doc_chunk = DocChunk(
                    library_item_id=item_id,
                    chunk_text=chunk_text,
                    chunk_order=i,
                )
                db.add(doc_chunk)
                await db.flush()

                embedding = await ai.generate_embedding(chunk_text)
                if embedding:
                    new_chunk_ids.append(str(doc_chunk.id))
                    new_embeddings.append(embedding)

            if new_chunk_ids:
                _faiss().add_batch(
                    user_id=str(current_user.id),
                    chunk_ids=new_chunk_ids,
                    embeddings=new_embeddings,
                )

        item.is_processed = bool(extracted_text)

    await db.commit()
    await db.refresh(item)
    return item


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library_item(item_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(UserLibraryItem).where(
            UserLibraryItem.id == item_id,
            UserLibraryItem.user_id == current_user.id,
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise NotFoundException("Library item not found")

    # Snapshot chunk IDs before cascade-delete removes the rows.
    chunks_result = await db.execute(
        select(DocChunk.id).where(DocChunk.library_item_id == item_id)
    )
    chunk_ids = {str(row[0]) for row in chunks_result.all()}

    storage = StorageService()
    if item.storage_path:
        await storage.delete_file(item.storage_path)

    await db.delete(item)
    await db.commit()

    if chunk_ids:
        # Try Celery first; fall back to inline removal if broker is unavailable.
        _del_via_celery = False
        try:
            delete_vault_embeddings.delay(str(current_user.id), list(chunk_ids))
            _del_via_celery = True
            print(f"[Delete] Celery queued FAISS cleanup ({len(chunk_ids)} chunk(s)) for user={current_user.id}")
        except Exception as exc:
            print(f"[Delete] Celery dispatch failed ({exc!r}), removing FAISS chunks inline")

        if not _del_via_celery:
            try:
                _faiss().remove_chunks(user_id=str(current_user.id), chunk_ids=chunk_ids)
                print(f"[Delete] Removed {len(chunk_ids)} FAISS chunk(s) inline for user={current_user.id}")
            except Exception as exc2:
                logger.warning("[Delete] Inline FAISS cleanup failed for user=%s: %s", current_user.id, exc2)


@router.post("/vault/query", response_model=VaultQueryResponse)
async def query_vault(payload: VaultQueryRequest, current_user: CurrentUser, db: DBSession):
    """RAG query against the user's uploaded documents (Knowledge Vault).

    Uses FAISS cosine-similarity search to find the most relevant chunks, then
    feeds them to the AI as context.  Falls back to recency-based retrieval when
    the user has no FAISS index yet.
    """
    # Check feature access, then deduct points
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="vault_ai_chat", db=db)
    await points_service.deduct(user_id=current_user.id, action="rag_query", db=db)

    ai = get_ai_service()
    faiss_svc = _faiss()

    # --- Vector similarity search via FAISS (preferred path) ---
    query_embedding = await ai.generate_query_embedding(payload.query)
    chunks: list[DocChunk] = []

    if query_embedding is not None and faiss_svc.user_has_index(str(current_user.id)):
        ranked_ids = faiss_svc.search(
            user_id=str(current_user.id),
            query_embedding=query_embedding,
            k=15,
        )

        if ranked_ids:
            # Fetch chunks from DB in ranked order, applying optional file filter
            chunk_uuids = [uuid.UUID(cid) for cid in ranked_ids]

            chunk_q = (
                select(DocChunk)
                .join(UserLibraryItem, DocChunk.library_item_id == UserLibraryItem.id)
                .where(
                    DocChunk.id.in_(chunk_uuids),
                    UserLibraryItem.user_id == current_user.id,
                )
            )
            if payload.file_ids:
                file_uuids = [uuid.UUID(fid) for fid in payload.file_ids]
                chunk_q = chunk_q.where(UserLibraryItem.id.in_(file_uuids))

            result = await db.execute(chunk_q)
            db_chunks_by_id = {str(c.id): c for c in result.scalars().all()}

            # Preserve FAISS ranking order
            chunks = [db_chunks_by_id[cid] for cid in ranked_ids if cid in db_chunks_by_id]

    # --- Fallback: recency-based retrieval ---
    if not chunks:
        file_filter = []
        if payload.file_ids:
            file_filter = [UserLibraryItem.id.in_([uuid.UUID(fid) for fid in payload.file_ids])]

        fallback_q = (
            select(DocChunk)
            .join(UserLibraryItem, DocChunk.library_item_id == UserLibraryItem.id)
            .where(
                UserLibraryItem.user_id == current_user.id,
                *file_filter,
            )
            .order_by(DocChunk.chunk_order)
            .limit(20)
        )
        result = await db.execute(fallback_q)
        chunks = result.scalars().all()

    if not chunks:
        return VaultQueryResponse(
            answer="No documents found in your vault. Please upload documents first.",
            sources=[],
            points_used=3,
        )

    context_text = "\n\n".join(c.chunk_text for c in chunks)
    answer = await ai.ask_document(
        query=payload.query,
        context=context_text,
        ai_context=payload.context,
    )

    return VaultQueryResponse(answer=answer, sources=[], points_used=3)
