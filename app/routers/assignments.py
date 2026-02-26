import uuid
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Assignment, Class
from app.schemas.classes import AssignmentCreate, AssignmentUpdate, AssignmentResponse, SuggestQuestionsRequest
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.ai_service import AIService

router = APIRouter()


@router.post("/", response_model=AssignmentResponse, status_code=status.HTTP_201_CREATED)
async def create_assignment(payload: AssignmentCreate, current_user: CurrentUser, db: DBSession):
    class_result = await db.execute(select(Class).where(Class.id == uuid.UUID(payload.class_id)))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

    assignment = Assignment(
        class_id=uuid.UUID(payload.class_id),
        title=payload.title,
        topic=payload.topic,
        instructions=payload.instructions,
        due_date=payload.due_date,
        points=payload.points,
        rubric_id=uuid.UUID(payload.rubric_id) if payload.rubric_id else None,
        status=payload.status,
        questions=payload.questions,
        attachments=payload.attachments,
        target_student_ids=payload.target_student_ids,
        created_by=current_user.id,
    )
    db.add(assignment)
    await db.commit()
    await db.refresh(assignment)
    return assignment


@router.get("/", response_model=list[AssignmentResponse])
async def list_assignments(
    current_user: CurrentUser,
    db: DBSession,
    class_id: str | None = Query(None),
    status: str | None = Query(None),
):
    q = select(Assignment)
    if class_id:
        q = q.where(Assignment.class_id == uuid.UUID(class_id))
    if status:
        q = q.where(Assignment.status == status)
    q = q.order_by(Assignment.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{assignment_id}", response_model=AssignmentResponse)
async def get_assignment(assignment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Assignment).where(Assignment.id == assignment_id))
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise NotFoundException("Assignment not found")
    return assignment


@router.patch("/{assignment_id}", response_model=AssignmentResponse)
async def update_assignment(
    assignment_id: uuid.UUID, payload: AssignmentUpdate, current_user: CurrentUser, db: DBSession
):
    result = await db.execute(select(Assignment).where(Assignment.id == assignment_id))
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise NotFoundException("Assignment not found")
    if assignment.created_by != current_user.id:
        raise ForbiddenException("Only the assignment creator can modify it")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(assignment, key, value)
    await db.commit()
    await db.refresh(assignment)
    return assignment


@router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assignment(assignment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(Assignment).where(Assignment.id == assignment_id))
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise NotFoundException("Assignment not found")
    if assignment.created_by != current_user.id:
        raise ForbiddenException("Only the assignment creator can delete it")
    await db.delete(assignment)
    await db.commit()


@router.post("/suggest-questions")
async def suggest_questions(payload: SuggestQuestionsRequest, current_user: CurrentUser, db: DBSession):
    """Use AI to suggest questions for an assignment based on topic."""
    ai = AIService()
    suggestions = await ai.suggest_questions(
        class_id=payload.class_id,
        topic=payload.topic,
        question_types=payload.question_types,
        count=payload.count,
        db=db,
    )
    return {"suggestions": suggestions}
