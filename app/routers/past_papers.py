import uuid
from fastapi import APIRouter, status, UploadFile, File, Form, Query
from sqlalchemy import select, and_

from app.dependencies import DBSession, CurrentUser
from app.models.content import PastPaper
from app.schemas.content import PastPaperResponse
from app.core.exceptions import NotFoundException
from app.services.storage_service import StorageService

router = APIRouter()


@router.get("/", response_model=list[PastPaperResponse])
async def list_past_papers(
    current_user: CurrentUser,
    db: DBSession,
    board: str | None = Query(None),
    grade: int | None = Query(None),
    subject: str | None = Query(None),
    year: int | None = Query(None),
    exam_type: str | None = Query(None),
    limit: int = Query(50, le=200),
):
    q = select(PastPaper).where(PastPaper.is_public == True)
    if board:
        q = q.where(PastPaper.board == board)
    if grade:
        q = q.where(PastPaper.grade == grade)
    if subject:
        q = q.where(PastPaper.subject == subject)
    if year:
        q = q.where(PastPaper.year == year)
    if exam_type:
        q = q.where(PastPaper.exam_type == exam_type)
    q = q.order_by(PastPaper.year.desc()).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


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
