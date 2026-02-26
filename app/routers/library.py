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
from app.services.points_service import PointsService

router = APIRouter()


@router.post("/upload", response_model=LibraryItemResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    folder: str = Form(None),
    tags: str = Form(None),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Upload a document to the knowledge vault. Text extraction happens asynchronously."""
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

    # Trigger text extraction (async)
    # In production, use Celery. For now, do it inline for small files.
    ai = AIService()
    try:
        extracted_text = await ai.extract_text_from_file(file_info["path"])
        if extracted_text:
            chunks = ai.chunk_text(extracted_text)
            for i, chunk in enumerate(chunks):
                doc_chunk = DocChunk(
                    library_item_id=item.id,
                    chunk_text=chunk,
                    chunk_order=i,
                )
                db.add(doc_chunk)
            item.is_processed = True
    except Exception:
        pass  # Non-blocking - extraction can be retried

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
        q = q.where(UserLibraryItem.folder == folder)
    q = q.order_by(UserLibraryItem.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


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

    # Re-chunk updated extracted text
    if extracted_text is not None:
        await db.execute(sql_delete(DocChunk).where(DocChunk.library_item_id == item_id))
        if extracted_text:
            ai = AIService()
            for i, chunk in enumerate(ai.chunk_text(extracted_text)):
                db.add(DocChunk(library_item_id=item_id, chunk_text=chunk, chunk_order=i))
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

    storage = StorageService()
    if item.storage_path:
        await storage.delete_file(item.storage_path)

    await db.delete(item)
    await db.commit()


@router.post("/vault/query", response_model=VaultQueryResponse)
async def query_vault(payload: VaultQueryRequest, current_user: CurrentUser, db: DBSession):
    """RAG query against the user's uploaded documents (Knowledge Vault)."""
    # Deduct points
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="rag_query", db=db)

    # Get chunks from specified files or all user files
    if payload.file_ids:
        result = await db.execute(
            select(DocChunk)
            .join(UserLibraryItem, DocChunk.library_item_id == UserLibraryItem.id)
            .where(
                UserLibraryItem.user_id == current_user.id,
                UserLibraryItem.id.in_([uuid.UUID(fid) for fid in payload.file_ids]),
            )
            .order_by(DocChunk.chunk_order)
        )
    else:
        result = await db.execute(
            select(DocChunk)
            .join(UserLibraryItem, DocChunk.library_item_id == UserLibraryItem.id)
            .where(UserLibraryItem.user_id == current_user.id)
            .order_by(DocChunk.chunk_order)
            .limit(100)
        )
    chunks = result.scalars().all()

    if not chunks:
        return VaultQueryResponse(
            answer="No documents found in your vault. Please upload documents first.",
            sources=[],
            points_used=3,
        )

    ai = AIService()
    context_text = "\n\n".join([c.chunk_text for c in chunks[:20]])
    answer = await ai.ask_document(
        query=payload.query,
        context=context_text,
        ai_context=payload.context,
    )

    return VaultQueryResponse(answer=answer, sources=[], points_used=3)
