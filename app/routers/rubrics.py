import uuid
from fastapi import APIRouter, status, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Rubric
from app.schemas.classes import RubricCreate, RubricUpdate, RubricResponse, GenerateRubricRequest
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.ai_service import AIService

router = APIRouter()


@router.post("/", response_model=RubricResponse, status_code=status.HTTP_201_CREATED)
async def create_rubric(payload: RubricCreate, current_user: CurrentUser, db: DBSession):
    rubric = Rubric(
        title=payload.title,
        board=payload.board,
        grade=payload.grade,
        subject=payload.subject,
        criteria=payload.criteria,
        created_by=current_user.id,
        is_ai_generated=False,
    )
    db.add(rubric)
    await db.commit()
    await db.refresh(rubric)
    return rubric


@router.get("/", response_model=list[RubricResponse])
async def list_rubrics(
    current_user: CurrentUser,
    db: DBSession,
    board: str | None = Query(None),
    grade: int | None = Query(None),
    subject: str | None = Query(None),
):
    q = select(Rubric).where(Rubric.created_by == current_user.id)
    if board:
        q = q.where(Rubric.board == board)
    if grade:
        q = q.where(Rubric.grade == grade)
    if subject:
        q = q.where(Rubric.subject == subject)
    q = q.order_by(Rubric.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{rubric_id}", response_model=RubricResponse)
async def get_rubric(rubric_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Rubric).where(Rubric.id == rubric_id))
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise NotFoundException("Rubric not found")
    return rubric


@router.patch("/{rubric_id}", response_model=RubricResponse)
async def update_rubric(
    rubric_id: uuid.UUID, payload: RubricUpdate, current_user: CurrentUser, db: DBSession
):
    result = await db.execute(select(Rubric).where(Rubric.id == rubric_id))
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise NotFoundException("Rubric not found")
    if rubric.created_by != current_user.id:
        raise ForbiddenException("Not your rubric")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(rubric, key, value)
    await db.commit()
    await db.refresh(rubric)
    return rubric


@router.delete("/{rubric_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rubric(rubric_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Rubric).where(Rubric.id == rubric_id))
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise NotFoundException("Rubric not found")
    if rubric.created_by != current_user.id:
        raise ForbiddenException("Not your rubric")
    await db.delete(rubric)
    await db.commit()


@router.post("/generate", response_model=RubricResponse)
async def generate_rubric_ai(payload: GenerateRubricRequest, current_user: CurrentUser, db: DBSession):
    """Use AI to auto-generate a rubric based on board, grade, subject and topic."""
    ai = AIService()
    criteria = await ai.generate_rubric(
        board=payload.board,
        grade=payload.grade,
        subject=payload.subject,
        topic=payload.topic,
        criteria_count=payload.criteria_count,
    )
    rubric = Rubric(
        title=f"{payload.subject} Rubric - {payload.topic}",
        board=payload.board,
        grade=payload.grade,
        subject=payload.subject,
        criteria=criteria,
        created_by=current_user.id,
        is_ai_generated=True,
    )
    db.add(rubric)
    await db.commit()
    await db.refresh(rubric)
    return rubric
