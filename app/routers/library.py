import uuid
from typing import Optional
from fastapi import APIRouter, status, UploadFile, File, Form, Query
from sqlalchemy import select, delete as sql_delete

from app.dependencies import DBSession, CurrentUser
from app.models.content import UserLibraryItem, DocChunk
from app.schemas.content import LibraryItemResponse, LibraryItemUpdate, VaultQueryRequest, VaultQueryResponse
from app.core.exceptions import NotFoundException
from app.services.storage_service import StorageService
from app.services.ai_service import AIService
from app.services.faiss_service import FAISSService
from app.services.points_service import PointsService
from app.config import settings

router = APIRouter()


def _faiss() -> FAISSService:
    return FAISSService(settings.STORAGE_ROOT)


@router.post("/upload", response_model=LibraryItemResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    folder: str = Form(None),
    tags: str = Form(None),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Upload a document to the knowledge vault. Text extraction + embedding happens inline."""
    storage = StorageService()
    file_info = await storage.upload_file(
        file=file,
        bucket="user-library",
        prefix=str(current_user.id),
    )

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    item = UserLibraryItem(
        user_id=current_user.id,
        title=file.filename or "Untitled",
        file_type=file_info.get("type"),
        storage_path=file_info.get("path"),
        file_size_mb=file_info.get("size_mb"),
        folder=folder,
        tags=tag_list,
        is_processed=False,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)

    # Award XP for document upload
    current_user.xp = (current_user.xp or 0) + 5

    # Text extraction → semantic chunking → FAISS embedding
    ai = AIService()
    try:
        extracted_text = await ai.extract_text_from_file(file_info["path"])
        if extracted_text:
            chunks = ai.semantic_chunk_text(extracted_text)
            chunk_ids: list[str] = []
            embeddings: list[list[float]] = []

            for i, chunk_text in enumerate(chunks):
                doc_chunk = DocChunk(
                    library_item_id=item.id,
                    chunk_text=chunk_text,
                    chunk_order=i,
                )
                db.add(doc_chunk)
                await db.flush()  # assigns doc_chunk.id

                embedding = await ai.generate_embedding(chunk_text)
                if embedding:
                    chunk_ids.append(str(doc_chunk.id))
                    embeddings.append(embedding)

            if chunk_ids:
                _faiss().add_batch(
                    user_id=str(current_user.id),
                    chunk_ids=chunk_ids,
                    embeddings=embeddings,
                )

            item.is_processed = True
            item.extracted_text_ref = "processed"
    except Exception:
        pass  # Non-blocking — extraction can be retried

    await db.commit()
    await db.refresh(item)
    return item


@router.get("/", response_model=list[LibraryItemResponse])
async def list_library_items(
    current_user: CurrentUser,
    db: DBSession,
    folder: str | None = Query(None),
):
    q = select(UserLibraryItem).where(UserLibraryItem.user_id == current_user.id)
    if folder:
        # Return only items in the requested folder
        q = q.where(UserLibraryItem.folder == folder)
    else:
        # Default: exclude OCR extractions — they belong to the Extract OCR tool, not the vault
        q = q.where(
            (UserLibraryItem.folder != "ocr") | (UserLibraryItem.folder.is_(None))
        )
    q = q.order_by(UserLibraryItem.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.post("/extract-text-inline")
async def extract_text_inline(
    file: UploadFile = File(...),
    current_user: CurrentUser = None,
):
    """Extract text from an uploaded file without saving it anywhere.

    Used by the AI assistant 'From Device' attachment feature so users can ask
    questions about a document without it being stored in the Knowledge Vault.
    """
    import tempfile
    import os
    from pathlib import Path as _Path

    ai = AIService()
    suffix = _Path(file.filename or "file").suffix.lower() or ".tmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = await ai.extract_text_from_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    word_count = len(text.split()) if text else 0
    return {
        "text": text or "",
        "word_count": word_count,
        "filename": file.filename,
    }


@router.get("/signed-url")
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
):
    """Alias for GET / that accepts an optional ?fields= param (ignored server-side)."""
    q = select(UserLibraryItem).where(UserLibraryItem.user_id == current_user.id)
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
            ai = AIService()
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

    # Collect chunk IDs so we can remove them from the FAISS index
    chunks_result = await db.execute(
        select(DocChunk.id).where(DocChunk.library_item_id == item_id)
    )
    chunk_ids = {str(row[0]) for row in chunks_result.all()}
    if chunk_ids:
        _faiss().remove_chunks(user_id=str(current_user.id), chunk_ids=chunk_ids)

    storage = StorageService()
    if item.storage_path:
        await storage.delete_file(item.storage_path)

    await db.delete(item)
    await db.commit()


@router.post("/vault/query", response_model=VaultQueryResponse)
async def query_vault(payload: VaultQueryRequest, current_user: CurrentUser, db: DBSession):
    """RAG query against the user's uploaded documents (Knowledge Vault).

    Uses FAISS cosine-similarity search to find the most relevant chunks, then
    feeds them to the AI as context.  Falls back to recency-based retrieval when
    the user has no FAISS index yet.
    """
    # Deduct points
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="rag_query", db=db)

    ai = AIService()
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
