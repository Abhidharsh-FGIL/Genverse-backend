"""
Public Library — no authentication required.

Two-level folder structure: Board → Grade.
Files (one per subject) are uploaded into grade folders with a subject tag.
Text extraction, semantic chunking, and FAISS embedding are processed in the
background via Celery.

All FAISS vectors are stored under a single shared index key: "public_library".
"""
import uuid
from typing import Optional

from fastapi import APIRouter, status, UploadFile, File, Form, Query, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.public_library import PublicFolder, PublicFile, PublicFileChunk
from app.services.storage_service import StorageService
from app.services.ai_service import AIService, get_ai_service
from app.services.faiss_service import FAISSService
from app.config import settings
from app.tasks.library_tasks import process_library_file_embeddings

router = APIRouter()

FAISS_KEY = "public_library"  # single shared FAISS index for all public files


def _faiss() -> FAISSService:
    return FAISSService(settings.STORAGE_ROOT)


# ── Folders (Board → Grade) ─────────────────────────────────────────────────

@router.post("/folders", status_code=status.HTTP_201_CREATED)
async def create_folder(
    name: str = Query(..., description="Folder name (e.g. 'CBSE', 'Grade 10')"),
    parent_id: Optional[str] = Query(None, description="Parent folder ID. Omit for board (root), provide board ID for grade."),
    folder_type: Optional[str] = Query(None, description="'board' for root folders, 'grade' for grade folders inside a board"),
    db: AsyncSession = Depends(get_db),
):
    """Create a Board or Grade folder. Only two levels allowed: Board (root) → Grade (child)."""
    parsed_parent = uuid.UUID(parent_id) if parent_id else None

    # Enforce two-level hierarchy
    if parsed_parent:
        parent = await db.get(PublicFolder, parsed_parent)
        if not parent:
            raise HTTPException(status_code=404, detail="Parent folder not found")
        if parent.parent_id is not None:
            raise HTTPException(status_code=400, detail="Only two levels allowed: Board → Grade. Cannot nest deeper.")

    # Check duplicate name under same parent
    existing = await db.execute(
        select(PublicFolder).where(
            PublicFolder.parent_id == parsed_parent,
            PublicFolder.name == name,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Folder '{name}' already exists in this location")

    # Auto-detect folder_type if not provided
    if not folder_type:
        folder_type = "grade" if parsed_parent else "board"

    folder = PublicFolder(
        name=name,
        parent_id=parsed_parent,
        folder_type=folder_type,
    )
    db.add(folder)
    await db.commit()
    await db.refresh(folder)

    return {
        "id": str(folder.id),
        "name": folder.name,
        "parent_id": str(folder.parent_id) if folder.parent_id else None,
        "folder_type": folder.folder_type,
        "created_at": folder.created_at.isoformat() if folder.created_at else None,
    }


@router.get("/folders")
async def list_folders(
    parent_id: Optional[str] = Query(None, description="Parent folder ID. Omit to list all boards."),
    db: AsyncSession = Depends(get_db),
):
    """List folders. Omit parent_id for boards, provide a board ID to list its grades."""
    parsed_parent = uuid.UUID(parent_id) if parent_id else None

    q = select(PublicFolder).where(PublicFolder.parent_id == parsed_parent).order_by(PublicFolder.name)
    result = await db.execute(q)
    folders = result.scalars().all()

    return [
        {
            "id": str(f.id),
            "name": f.name,
            "parent_id": str(f.parent_id) if f.parent_id else None,
            "folder_type": f.folder_type,
            "created_at": f.created_at.isoformat() if f.created_at else None,
        }
        for f in folders
    ]


@router.get("/folders/{folder_id}")
async def get_folder(
    folder_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a folder with its children (grades) and files (subject books)."""
    fid = uuid.UUID(folder_id)
    result = await db.execute(
        select(PublicFolder)
        .options(selectinload(PublicFolder.children), selectinload(PublicFolder.files))
        .where(PublicFolder.id == fid)
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    # Build breadcrumb (walk up parents)
    breadcrumb = []
    current = folder
    while current:
        breadcrumb.insert(0, {"id": str(current.id), "name": current.name})
        if current.parent_id:
            current = await db.get(PublicFolder, current.parent_id)
        else:
            current = None

    return {
        "id": str(folder.id),
        "name": folder.name,
        "parent_id": str(folder.parent_id) if folder.parent_id else None,
        "folder_type": folder.folder_type,
        "breadcrumb": breadcrumb,
        "children": [
            {
                "id": str(c.id),
                "name": c.name,
                "folder_type": c.folder_type,
            }
            for c in sorted(folder.children, key=lambda x: x.name)
        ],
        "files": [
            {
                "id": str(f.id),
                "title": f.title,
                "subject": f.subject,
                "file_type": f.file_type,
                "file_size_mb": f.file_size_mb,
                "storage_path": f.storage_path,
                "is_processed": f.is_processed,
                "chunks_embedded": f.chunks_embedded,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in folder.files if f.is_processed
        ],
    }


@router.delete("/folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    folder_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a folder and all its contents (child grades, files, chunks, FAISS vectors)."""
    fid = uuid.UUID(folder_id)
    folder = await db.get(PublicFolder, fid)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    # Collect all file chunk IDs recursively for FAISS cleanup
    chunk_ids = await _collect_chunk_ids_recursive(db, fid)
    if chunk_ids:
        _faiss().remove_chunks(FAISS_KEY, set(chunk_ids))

    await db.delete(folder)
    await db.commit()


async def _collect_chunk_ids_recursive(db: AsyncSession, folder_id: uuid.UUID) -> list[str]:
    """Recursively collect all chunk IDs under a folder tree."""
    chunk_ids: list[str] = []

    # Get files in this folder
    files_result = await db.execute(
        select(PublicFile).where(PublicFile.folder_id == folder_id)
    )
    for f in files_result.scalars().all():
        chunks_result = await db.execute(
            select(PublicFileChunk.id).where(PublicFileChunk.file_id == f.id)
        )
        chunk_ids.extend(str(cid) for cid in chunks_result.scalars().all())

    # Recurse into child folders
    children_result = await db.execute(
        select(PublicFolder.id).where(PublicFolder.parent_id == folder_id)
    )
    for child_id in children_result.scalars().all():
        chunk_ids.extend(await _collect_chunk_ids_recursive(db, child_id))

    return chunk_ids


# ── Files (one per subject inside a grade folder) ───────────────────────────

@router.post("/files", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    folder_id: str = Form(..., description="Grade folder ID to upload into"),
    subject: str = Form(..., description="Subject name (e.g. 'Mathematics', 'Physics')"),
    title: Optional[str] = Form(None, description="Display title (defaults to filename)"),
    db: AsyncSession = Depends(get_db),
):
    """Upload a subject book into a grade folder. Saves the file immediately and dispatches embedding processing to Celery."""
    fid = uuid.UUID(folder_id)
    folder = await db.get(PublicFolder, fid)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    # Store file
    storage = StorageService()
    file_info = await storage.upload_file(
        file=file,
        bucket="public-library",
        prefix=str(fid),
    )

    pub_file = PublicFile(
        folder_id=fid,
        title=title or file.filename or "Untitled",
        subject=subject,
        file_type=file_info.get("type"),
        storage_path=file_info.get("path"),
        file_size_mb=file_info.get("size_mb"),
        is_processed=False,
    )
    db.add(pub_file)
    await db.commit()
    await db.refresh(pub_file)

    # Dispatch embedding processing to Celery background worker
    task = process_library_file_embeddings.delay(str(pub_file.id))

    return {
        "id": str(pub_file.id),
        "folder_id": str(pub_file.folder_id),
        "title": pub_file.title,
        "subject": pub_file.subject,
        "file_type": pub_file.file_type,
        "file_size_mb": pub_file.file_size_mb,
        "is_processed": False,
        "task_id": task.id,
        "created_at": pub_file.created_at.isoformat() if pub_file.created_at else None,
    }


@router.get("/files/{file_id}")
async def get_file(
    file_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get file details including its chunks."""
    result = await db.execute(
        select(PublicFile)
        .options(selectinload(PublicFile.chunks))
        .where(PublicFile.id == uuid.UUID(file_id))
    )
    pub_file = result.scalar_one_or_none()
    if not pub_file:
        raise HTTPException(status_code=404, detail="File not found")
    if not pub_file.is_processed:
        raise HTTPException(status_code=202, detail="File is still being processed")

    return {
        "id": str(pub_file.id),
        "folder_id": str(pub_file.folder_id),
        "title": pub_file.title,
        "subject": pub_file.subject,
        "file_type": pub_file.file_type,
        "file_size_mb": pub_file.file_size_mb,
        "is_processed": pub_file.is_processed,
        "chunks_embedded": pub_file.chunks_embedded,
        "created_at": pub_file.created_at.isoformat() if pub_file.created_at else None,
        "chunks": [
            {"id": str(c.id), "chunk_order": c.chunk_order, "chunk_text": c.chunk_text[:200]}
            for c in sorted(pub_file.chunks, key=lambda x: x.chunk_order)
        ],
    }


@router.delete("/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a file and remove its chunks from FAISS."""
    fid = uuid.UUID(file_id)
    result = await db.execute(
        select(PublicFile).options(selectinload(PublicFile.chunks)).where(PublicFile.id == fid)
    )
    pub_file = result.scalar_one_or_none()
    if not pub_file:
        raise HTTPException(status_code=404, detail="File not found")

    # Remove from FAISS
    chunk_ids = {str(c.id) for c in pub_file.chunks}
    if chunk_ids:
        _faiss().remove_chunks(FAISS_KEY, chunk_ids)

    await db.delete(pub_file)
    await db.commit()


# ── Search ───────────────────────────────────────────────────────────────────

@router.get("/search")
async def search_public_library(
    q: str = Query(..., min_length=2, description="Search query"),
    k: int = Query(10, ge=1, le=50, description="Number of results"),
    db: AsyncSession = Depends(get_db),
):
    """Semantic search across all public library files using FAISS."""
    ai = get_ai_service()
    query_embedding = await ai.generate_query_embedding(q)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")

    chunk_ids = _faiss().search(user_id=FAISS_KEY, query_embedding=query_embedding, k=k)
    if not chunk_ids:
        return {"results": []}

    # Fetch chunks with their file and folder info
    result = await db.execute(
        select(PublicFileChunk)
        .options(selectinload(PublicFileChunk.file).selectinload(PublicFile.folder))
        .where(PublicFileChunk.id.in_([uuid.UUID(cid) for cid in chunk_ids]))
    )
    chunks = {str(c.id): c for c in result.scalars().all()}

    # Maintain FAISS ranking order
    results = []
    for cid in chunk_ids:
        chunk = chunks.get(cid)
        if not chunk:
            continue
        results.append({
            "chunk_id": cid,
            "chunk_text": chunk.chunk_text,
            "chunk_order": chunk.chunk_order,
            "file": {
                "id": str(chunk.file.id),
                "title": chunk.file.title,
                "subject": chunk.file.subject,
                "file_type": chunk.file.file_type,
            },
            "folder": {
                "id": str(chunk.file.folder.id),
                "name": chunk.file.folder.name,
                "folder_type": chunk.file.folder.folder_type,
            } if chunk.file.folder else None,
        })

    return {"results": results}


# ── Task Status ───────────────────────────────────────────────────────────────

@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of a Celery embedding task."""
    from app.celery_app import celery as celery_app

    result = celery_app.AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": result.status,  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
    }
    if result.ready():
        response["result"] = result.result if result.successful() else str(result.result)
    return response


@router.get("/files/{file_id}/status")
async def get_file_processing_status(
    file_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Check whether a file's embeddings have been processed."""
    pub_file = await db.get(PublicFile, uuid.UUID(file_id))
    if not pub_file:
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "id": str(pub_file.id),
        "title": pub_file.title,
        "is_processed": pub_file.is_processed,
        "chunks_embedded": pub_file.chunks_embedded,
    }


# ── Signed URL (no auth) ────────────────────────────────────────────────────

@router.get("/signed-url", include_in_schema=True)
@router.get("/signed-url/", include_in_schema=False)
async def get_signed_url(
    path: str = Query(..., description="Storage path of the file"),
):
    """Convert an absolute storage path to a serving URL (no auth required)."""
    from pathlib import Path as _Path
    try:
        storage_root = _Path(settings.STORAGE_ROOT).resolve()
        abs_path = _Path(path).resolve()
        rel = abs_path.relative_to(storage_root)
        url = f"/uploads/{rel.as_posix()}"
    except Exception:
        url = None
    return {"url": url}
