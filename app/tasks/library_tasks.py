"""
Celery tasks for Public Library embedding processing.

These run in a separate worker process, so we use synchronous DB access
and asyncio.run() for the async AI service calls.
"""
import asyncio
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.celery_app import celery
from app.config import settings
from app.models.public_library import PublicFile, PublicFileChunk
from app.services.ai_service import AIService
from app.services.faiss_service import FAISSService

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
