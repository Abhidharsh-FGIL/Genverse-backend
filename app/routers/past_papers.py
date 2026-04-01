import uuid
from fastapi import APIRouter, status, UploadFile, File, Form, Query
from sqlalchemy import select, and_

from app.dependencies import DBSession, CurrentUser
from app.models.content import PastPaper
from app.schemas.content import PastPaperResponse, PastPaperListResponse
from app.core.exceptions import NotFoundException
from app.services.storage_service import StorageService

router = APIRouter()


@router.get("/", response_model=PastPaperListResponse)
async def list_past_papers(
    current_user: CurrentUser,
    db: DBSession,
    board: str | None = Query(None),
    grade: int | None = Query(None),
    subject: str | None = Query(None),
    year: int | None = Query(None),
    exam_type: str | None = Query(None),
    limit: int = Query(12, ge=1, le=50),
    cursor: str | None = Query(None, description="ISO datetime cursor for pagination"),
):
    from datetime import datetime as dt
    from sqlalchemy import func as sa_func

    base = select(PastPaper).where(PastPaper.is_public == True)
    if board:
        base = base.where(PastPaper.board == board)
    if grade:
        base = base.where(PastPaper.grade == grade)
    if subject:
        base = base.where(PastPaper.subject == subject)
    if year:
        base = base.where(PastPaper.year == year)
    if exam_type:
        base = base.where(PastPaper.exam_type == exam_type)

    count_q = select(sa_func.count()).select_from(base.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    q = base
    if cursor:
        try:
            cursor_dt = dt.fromisoformat(cursor)
            q = q.where(PastPaper.created_at < cursor_dt)
        except (ValueError, TypeError):
            pass

    q = q.order_by(PastPaper.created_at.desc()).limit(limit + 1)
    result = await db.execute(q)
    rows = result.scalars().all()

    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].created_at.isoformat() if has_more and items else None

    return PastPaperListResponse(items=items, next_cursor=next_cursor, has_more=has_more, total=total)


@router.get("/{paper_id}/signed-url")
async def get_past_paper_signed_url(paper_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    """Return a serving URL for viewing/downloading a past paper file."""
    from pathlib import Path as _Path
    from app.config import settings as _settings

    result = await db.execute(select(PastPaper).where(PastPaper.id == paper_id))
    paper = result.scalar_one_or_none()
    if not paper or (not paper.is_public and paper.uploaded_by != current_user.id):
        raise NotFoundException("Past paper not found")

    if not paper.storage_path:
        return {"url": paper.file_url}

    try:
        storage_root = _Path(_settings.STORAGE_ROOT).resolve()
        abs_path = _Path(paper.storage_path).resolve()
        rel = abs_path.relative_to(storage_root)
        url = f"/uploads/{rel.as_posix()}"
    except Exception:
        url = paper.file_url
    return {"url": url}


@router.get("/{paper_id}", response_model=PastPaperResponse)
async def get_past_paper(paper_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(PastPaper).where(PastPaper.id == paper_id))
    paper = result.scalar_one_or_none()
    if not paper or (not paper.is_public and paper.uploaded_by != current_user.id):
        raise NotFoundException("Past paper not found")
    return paper


@router.post("/upload", response_model=PastPaperResponse, status_code=status.HTTP_201_CREATED)
async def upload_past_paper(
    file: UploadFile = File(...),
    title: str = Form(...),
    board: str = Form(None),
    grade: int = Form(None),
    subject: str = Form(None),
    year: int = Form(None),
    exam_type: str = Form(None),
    is_public: bool = Form(True),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    storage = StorageService()
    file_info = await storage.upload_file(
        file=file,
        bucket="past-papers",
        prefix=str(current_user.id),
    )

    paper = PastPaper(
        title=title,
        board=board,
        grade=grade,
        subject=subject,
        year=year,
        exam_type=exam_type,
        storage_path=file_info.get("path"),
        file_url=file_info.get("url"),
        uploaded_by=current_user.id,
        is_public=is_public,
    )
    db.add(paper)
    await db.commit()
    await db.refresh(paper)
    return paper
