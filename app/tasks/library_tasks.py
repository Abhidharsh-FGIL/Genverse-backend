"""
Celery tasks for embedding processing (Public Library & Knowledge Vault).

These run in a separate worker process, so we use synchronous DB access
and asyncio.run() for the async AI service calls.
"""
import asyncio
import json
import logging
import uuid

import redis as sync_redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.celery_app import celery
from app.config import settings
from app.models.public_library import PublicFile, PublicFileChunk
from app.models.content import UserLibraryItem, DocChunk
from app.services.ai_service import AIService
from app.services.faiss_service import FAISSService

logger = logging.getLogger(__name__)


def _vault_channel(user_id: str) -> str:
    """Redis pub/sub channel for a user's Knowledge Vault status events.
    Subscribed by the SSE endpoint in app/routers/library.py."""
    return f"vault:events:{user_id}"


def _publish_vault_event(user_id: str, event: dict) -> None:
    """Publish a Knowledge Vault status event to Redis pub/sub.

    Failure here is non-fatal — the SSE stream is best-effort, and the
    frontend has a one-shot reconciliation refetch on reconnect.
    """
    try:
        r = sync_redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        r.publish(_vault_channel(user_id), json.dumps(event))
    except Exception as exc:
        logger.warning("[Vault] Failed to publish SSE event for user=%s: %s", user_id, exc)

FAISS_KEY = "public_library"

# Sync engine for Celery worker (separate from FastAPI's async engine)
_sync_engine = create_engine(
    settings.sync_database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
SyncSessionLocal = sessionmaker(bind=_sync_engine, expire_on_commit=False)


def _run_async(coro):
    """Run an async coroutine from sync Celery context."""
    return asyncio.run(coro)


@celery.task(bind=True, name="process_library_file_embeddings", max_retries=2)
def process_library_file_embeddings(self, file_id: str):
    """
    Extract text, chunk it, generate embeddings, and store in FAISS.

    Called after a file is uploaded to the public library.
    The file record already exists in DB with is_processed=False.
    """
    db: Session = SyncSessionLocal()
    try:
        pub_file = db.get(PublicFile, uuid.UUID(file_id))
        if not pub_file:
            return {"status": "error", "detail": "File not found"}

        if pub_file.is_processed:
            return {"status": "skipped", "detail": "Already processed"}

        ai = AIService()
        faiss_svc = FAISSService(settings.STORAGE_ROOT)

        # 1. Extract text from the stored file
        extracted_text = _run_async(ai.extract_text_from_file(pub_file.storage_path))
        if not extracted_text:
            pub_file.is_processed = False
            db.commit()
            return {"status": "error", "detail": "No text extracted"}

        # 2. Semantic chunking
        chunks = ai.semantic_chunk_text(extracted_text)

        # 3. Create chunk records + generate embeddings
        chunk_ids: list[str] = []
        embeddings: list[list[float]] = []

        for i, chunk_text in enumerate(chunks):
            doc_chunk = PublicFileChunk(
                file_id=pub_file.id,
                chunk_text=chunk_text,
                chunk_order=i,
            )
            db.add(doc_chunk)
            db.flush()  # get the assigned id

            embedding = _run_async(ai.generate_embedding(chunk_text))
            if embedding:
                chunk_ids.append(str(doc_chunk.id))
                embeddings.append(embedding)

        # 4. Batch-add to FAISS index
        if chunk_ids:
            faiss_svc.add_batch(
                user_id=FAISS_KEY,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
            )

        # 5. Mark file as processed and store embedding count
        pub_file.is_processed = True
        pub_file.chunks_embedded = len(chunk_ids)
        db.commit()

        return {
            "status": "success",
            "file_id": file_id,
            "chunks_embedded": len(chunk_ids),
        }

    except Exception as exc:
        db.rollback()
        # Retry with exponential backoff (10s, 30s)
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))
    finally:
        db.close()


@celery.task(bind=True, name="delete_vault_embeddings", max_retries=2)
def delete_vault_embeddings(self, user_id: str, chunk_ids: list[str]):
    """
    Remove the given chunk IDs from the user's FAISS index.

    Called asynchronously after a Knowledge Vault item is deleted via
    DELETE /library/{item_id}. The UserLibraryItem and its DocChunk rows
    have already been removed from Postgres by the request handler;
    this task only cleans up the vector store.
    """
    if not chunk_ids:
        return {"status": "skipped", "detail": "No chunk IDs provided"}

    logger.info(
        "[Vault] Removing %d embedding(s) from FAISS for user=%s",
        len(chunk_ids), user_id,
    )
    try:
        faiss_svc = FAISSService(settings.STORAGE_ROOT)
        faiss_svc.remove_chunks(user_id=user_id, chunk_ids=set(chunk_ids))
        logger.info(
            "[Vault] Removed %d embedding(s) from FAISS for user=%s",
            len(chunk_ids), user_id,
        )
        return {"status": "success", "removed": len(chunk_ids)}
    except Exception as exc:
        logger.exception(
            "[Vault] Failed to remove embeddings from FAISS for user=%s", user_id,
        )
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))


@celery.task(bind=True, name="process_vault_embeddings", max_retries=2)
def process_vault_embeddings(self, item_id: str, user_id: str, filename: str = ""):
    """
    Extract text from an uploaded Knowledge Vault file, chunk it,
    generate embeddings, and store them in the user's FAISS index.

    Called asynchronously after a file is uploaded via POST /library/upload.
    The UserLibraryItem already exists with processing_status='processing'.
    """
    from sqlalchemy import select as _sa_select, delete as _sa_delete

    print(f"[Celery] Starting processing for '{filename}' (item={item_id}, user={user_id})")
    db: Session = SyncSessionLocal()
    try:
        item = db.get(UserLibraryItem, uuid.UUID(item_id))
        if not item:
            print(f"[Celery] ERROR: Item {item_id} ('{filename}') not found in DB")
            return {"status": "error", "detail": "Item not found"}

        if item.processing_status == "ready":
            print(f"[Celery] '{filename}' already processed, skipping")
            return {"status": "skipped", "detail": "Already processed"}

        ai = AIService()
        faiss_svc = FAISSService(settings.STORAGE_ROOT)

        # Remove any stale chunks from a previous failed/partial attempt
        old_ids = [str(r[0]) for r in db.execute(
            _sa_select(DocChunk.id).where(DocChunk.library_item_id == item.id)
        ).all()]
        if old_ids:
            print(f"[Celery] Removing {len(old_ids)} stale chunk(s) before reprocessing")
            faiss_svc.remove_chunks(user_id=user_id, chunk_ids=set(old_ids))
            db.execute(_sa_delete(DocChunk).where(DocChunk.library_item_id == item.id))
            db.flush()

        # 1. Extract text from the stored file
        print(f"[Celery] Extracting text from '{filename}' (path: {item.storage_path})")
        extracted_text = _run_async(ai.extract_text_from_file(item.storage_path))
        if not extracted_text:
            item.processing_status = "failed"
            db.commit()
            print(f"[Celery] WARNING: No text extracted from '{filename}' — marked as failed")
            _publish_vault_event(user_id, {
                "type": "item_status",
                "item_id": item_id,
                "processing_status": "failed",
            })
            return {"status": "error", "detail": "No text extracted"}

        print(f"[Celery] Extracted {len(extracted_text)} chars from '{filename}'")

        # 2. Semantic chunking
        chunks = ai.semantic_chunk_text(extracted_text)
        print(f"[Celery] '{filename}' → {len(chunks)} chunks")

        # 3. Create chunk records + generate embeddings
        chunk_ids: list[str] = []
        embeddings: list[list[float]] = []

        for i, chunk_text in enumerate(chunks):
            doc_chunk = DocChunk(
                library_item_id=item.id,
                chunk_text=chunk_text,
                chunk_order=i,
            )
            db.add(doc_chunk)
            db.flush()

            embedding = _run_async(ai.generate_embedding(chunk_text))
            if embedding:
                chunk_ids.append(str(doc_chunk.id))
                embeddings.append(embedding)

        # 4. Batch-add to user's FAISS index
        if chunk_ids:
            faiss_svc.add_batch(
                user_id=user_id,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
            )

        # 5. Mark item as ready
        item.is_processed = True
        item.extracted_text_ref = "processed"
        item.processing_status = "ready"
        db.commit()

        print(f"[Celery] '{filename}' done — {len(chunk_ids)} embeddings stored")
        _publish_vault_event(user_id, {
            "type": "item_status",
            "item_id": item_id,
            "processing_status": "ready",
            "is_processed": True,
            "extracted_text_ref": "processed",
            "chunks_embedded": len(chunk_ids),
        })
        return {
            "status": "success",
            "item_id": item_id,
            "chunks_embedded": len(chunk_ids),
        }

    except Exception as exc:
        db.rollback()
        print(f"[Celery] ERROR processing '{filename}' (item={item_id}): {exc}")
        # Mark as failed on final retry
        if self.request.retries >= self.max_retries:
            try:
                item = db.get(UserLibraryItem, uuid.UUID(item_id))
                if item:
                    item.processing_status = "failed"
                    db.commit()
                    _publish_vault_event(user_id, {
                        "type": "item_status",
                        "item_id": item_id,
                        "processing_status": "failed",
                    })
            except Exception:
                db.rollback()
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))
    finally:
        db.close()
