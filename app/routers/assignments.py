import uuid
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Assignment, Class
from app.schemas.classes import AssignmentCreate, AssignmentUpdate, AssignmentResponse, SuggestQuestionsRequest
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.ai_service import AIService
from app.models.classes import ClassStudent
from app.models.notification import NotificationType
from app.services.notification_service import create_notification_for_many

router = APIRouter()


@router.post("/", response_model=AssignmentResponse, status_code=status.HTTP_201_CREATED)
async def create_assignment(payload: AssignmentCreate, current_user: CurrentUser, db: DBSession):
    class_result = await db.execute(select(Class).where(Class.id == uuid.UUID(payload.class_id)))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")
    if not class_.is_active:
        raise HTTPException(status_code=400, detail="Cannot create assignments in an archived class")

    assignment = Assignment(
        class_id=uuid.UUID(payload.class_id),
        title=payload.title,
        topic=payload.topic,
        instructions=payload.instructions,
        due_date=payload.due_date,
        points=payload.points,
        rubric_id=uuid.UUID(payload.rubric_id) if payload.rubric_id else None,
        lesson_plan_id=uuid.UUID(payload.lesson_plan_id) if payload.lesson_plan_id else None,
        status=payload.status,
        questions=payload.questions,
        attachments=payload.attachments,
        target_student_ids=payload.target_student_ids,
        assignment_type=payload.assignment_type,
        created_by=current_user.id,
    )
    db.add(assignment)
    await db.commit()
    await db.refresh(assignment)

    # Notify students about the new assignment
    if payload.target_student_ids:
        student_ids = [uuid.UUID(sid) for sid in payload.target_student_ids]
    else:
        rows = (await db.execute(
            select(ClassStudent.student_id).where(ClassStudent.class_id == assignment.class_id)
        )).scalars().all()
        student_ids = list(rows)
    if student_ids:
        await create_notification_for_many(
            db,
            user_ids=student_ids,
            notification_type=NotificationType.ASSIGNMENT_POSTED,
            title=f"New Assignment: {assignment.title}",
            body=f"A new assignment has been posted in {class_.name}.",
            icon="file-text",
            org_id=class_.org_id if hasattr(class_, 'org_id') else None,
            data_json={"assignment_id": str(assignment.id), "class_id": str(class_.id), "link": f"/student/class/{class_.id}"},
        )
        await db.commit()

    return assignment


@router.get("/", response_model=list[AssignmentResponse])
async def list_assignments(
    current_user: CurrentUser,
    db: DBSession,
    class_id: str | None = Query(None),
    classIds: str | None = Query(None),
    status: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    q = select(Assignment)

    # Filter by class_id (singular) or classIds (comma-separated)
    if classIds:
        class_id_list = [uuid.UUID(cid.strip()) for cid in classIds.split(",") if cid.strip()]
        if class_id_list:
            q = q.where(Assignment.class_id.in_(class_id_list))
        else:
            return []
    elif class_id:
        q = q.where(Assignment.class_id == uuid.UUID(class_id))
    else:
        # No class filter — scope to classes the user has access to (enrolled or teaches)
        from app.models.classes import ClassStudent, ClassTeacher
        enrolled_q = select(ClassStudent.class_id).where(ClassStudent.student_id == current_user.id)
        teaching_q = select(Class.id).where(Class.teacher_id == current_user.id)
        co_teaching_q = select(ClassTeacher.class_id).where(ClassTeacher.teacher_id == current_user.id)
        enrolled_result = await db.execute(enrolled_q)
        teaching_result = await db.execute(teaching_q)
        co_teaching_result = await db.execute(co_teaching_q)
        allowed_ids = set(
            [r[0] for r in enrolled_result.all()] +
            [r[0] for r in teaching_result.all()] +
            [r[0] for r in co_teaching_result.all()]
        )
        if not allowed_ids:
            return []
        q = q.where(Assignment.class_id.in_(list(allowed_ids)))

    if status:
        q = q.where(Assignment.status == status)
    q = q.order_by(Assignment.created_at.desc()).offset(skip).limit(limit)
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

    updates = payload.model_dump(exclude_unset=True)
    # Convert string UUIDs to proper UUID objects
    for uuid_field in ('rubric_id', 'lesson_plan_id'):
        if uuid_field in updates:
            val = updates[uuid_field]
            updates[uuid_field] = uuid.UUID(val) if val else None
    for key, value in updates.items():
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


@router.post("/files/upload")
async def upload_assignment_file(
    file: UploadFile = File(...),
    path: str = Form(""),
    current_user: CurrentUser = None,
):
    """Upload a file (image, document) for assignment questions or documents."""
    from app.services.storage_service import StorageService

    storage = StorageService()
    prefix = path.rsplit("/", 1)[0] if "/" in path else "general"
    file_info = await storage.upload_file(file=file, bucket="assignment-files", prefix=prefix)
    return {"url": file_info["url"]}
