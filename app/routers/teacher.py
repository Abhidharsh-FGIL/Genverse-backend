import uuid
from typing import Optional, Any, List
from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.classes import Assignment, Class, ClassTeacher, Submission
from app.models.user import User
from app.core.exceptions import NotFoundException

router = APIRouter()


class RubricCreateRequest(BaseModel):
    title: str
    board: str = "CBSE"
    grade: int = 10
    subject: str = ""
    class_id: Optional[str] = None
    criteria: Optional[List[Any]] = None
    difficulty_level: Optional[str] = None


@router.get("/assignments")
async def list_teacher_assignments(
    current_user: CurrentUser,
    db: DBSession,
    class_ids: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """Return assignments for the teacher's classes, optionally filtered by class_ids."""
    # Resolve which class IDs to query
    if class_ids:
        requested_ids = [uuid.UUID(cid.strip()) for cid in class_ids.split(",") if cid.strip()]
    else:
        # All classes where user is teacher or co-teacher
        own_result = await db.execute(
            select(Class.id).where(Class.teacher_id == current_user.id, Class.is_active == True)
        )
        co_result = await db.execute(
            select(ClassTeacher.class_id).where(ClassTeacher.teacher_id == current_user.id)
        )
        requested_ids = [r[0] for r in own_result.all()] + [r[0] for r in co_result.all()]

    if not requested_ids:
        return []

    q = select(Assignment).where(Assignment.class_id.in_(requested_ids))
    if status:
        q = q.where(Assignment.status == status)
    q = q.order_by(Assignment.created_at.desc())
    result = await db.execute(q)
    assignments = result.scalars().all()

    # Return flat dicts that include class_id directly (frontend uses a.class_id)
    return [
        {
            "id": str(a.id),
            "class_id": str(a.class_id),
            "title": a.title,
            "status": a.status,
            "due_date": a.due_date.isoformat() if a.due_date else None,
            "points": a.points,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in assignments
    ]


@router.get("/submissions/{submission_id}")
async def get_teacher_submission(
    submission_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
):
    """Get a single submission with full nested data for the grading page."""
    result = await db.execute(
        select(Submission, User)
        .join(User, Submission.student_id == User.id)
        .where(Submission.id == submission_id)
    )
    row = result.one_or_none()
    if not row:
        raise NotFoundException("Submission not found")

    sub, student = row

    # Fetch assignment + class
    assign_result = await db.execute(
        select(Assignment, Class)
        .join(Class, Assignment.class_id == Class.id)
        .where(Assignment.id == sub.assignment_id)
    )
    assign_row = assign_result.one_or_none()
    assignment = None
    class_ = None
    if assign_row:
        assignment, class_ = assign_row

    # Fetch rubric if linked
    rubric_data = None
    if assignment and assignment.rubric_id:
        from app.models.classes import Rubric
        rubric_result = await db.execute(
            select(Rubric).where(Rubric.id == assignment.rubric_id)
        )
        rubric = rubric_result.scalar_one_or_none()
        if rubric:
            rubric_data = {
                "id": str(rubric.id),
                "title": rubric.title,
                "criteria": rubric.criteria,
            }

    return {
        "id": str(sub.id),
        "assignment_id": str(sub.assignment_id),
        "student_id": str(sub.student_id),
        "status": sub.status,
        "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None,
        "text_response": sub.text_response,
        "files": sub.files,
        "grade": sub.grade,
        "ai_grade_suggestion": sub.ai_grade_suggestion,
        "remediation_plan": sub.remediation_plan,
        "graded_at": sub.graded_at.isoformat() if sub.graded_at else None,
        "student": {
            "id": str(student.id),
            "name": student.name,
            "email": student.email,
            "avatar": getattr(student, "avatar", None),
        },
        "assignment": {
            "id": str(assignment.id) if assignment else None,
            "title": assignment.title if assignment else None,
            "instructions": assignment.instructions if assignment else None,
            "points": assignment.points if assignment else None,
            "due_date": assignment.due_date.isoformat() if assignment and assignment.due_date else None,
            "attachments": assignment.attachments if assignment else None,
            "questions": assignment.questions if assignment else None,
            "class_id": str(assignment.class_id) if assignment else None,
            "classes": {
                "id": str(class_.id),
                "name": class_.name,
                "color": class_.color,
            } if class_ else None,
        } if assignment else None,
        "rubric": rubric_data,
    }


@router.get("/submissions")
async def list_teacher_submissions(
    current_user: CurrentUser,
    db: DBSession,
    status: Optional[str] = Query(None),
    class_ids: Optional[str] = Query(None),
    limit: Optional[int] = Query(None),
):
    """List submissions for the teacher's classes with nested assignment, class, and student data."""
    # Resolve class IDs the teacher owns or co-teaches
    own_result = await db.execute(
        select(Class.id).where(Class.teacher_id == current_user.id, Class.is_active == True)
    )
    co_result = await db.execute(
        select(ClassTeacher.class_id).where(ClassTeacher.teacher_id == current_user.id)
    )
    teacher_class_ids = set(
        [r[0] for r in own_result.all()] + [r[0] for r in co_result.all()]
    )

    if not teacher_class_ids:
        return []

    # Further filter by requested class_ids if provided
    if class_ids:
        requested = {uuid.UUID(cid.strip()) for cid in class_ids.split(",") if cid.strip()}
        teacher_class_ids = teacher_class_ids & requested

    if not teacher_class_ids:
        return []

    # Fetch submissions joined with assignment (to filter by class) and student
    q = (
        select(Submission, Assignment, Class, User)
        .join(Assignment, Submission.assignment_id == Assignment.id)
        .join(Class, Assignment.class_id == Class.id)
        .join(User, Submission.student_id == User.id)
        .where(Assignment.class_id.in_(list(teacher_class_ids)))
    )
    if status:
        q = q.where(Submission.status == status)
    q = q.order_by(Submission.submitted_at.desc())
    if limit:
        q = q.limit(limit)

    result = await db.execute(q)
    rows = result.all()

    return [
        {
            "id": str(sub.id),
            "status": sub.status,
            "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None,
            "grade": sub.grade,
            "student": {
                "id": str(student.id),
                "name": student.name,
                "email": student.email,
            },
            "assignment": {
                "id": str(assignment.id),
                "title": assignment.title,
                "class_id": str(assignment.class_id),
                "classes": {
                    "id": str(class_.id),
                    "name": class_.name,
                },
            },
        }
        for sub, assignment, class_, student in rows
    ]


@router.get("/lesson-plans")
async def list_teacher_lesson_plans(
    current_user: CurrentUser,
    db: DBSession,
    class_id: Optional[str] = Query(None),
):
    """List lesson plans created by the teacher, optionally filtered by class."""
    from app.models.classes import LessonPlan
    q = select(LessonPlan).where(LessonPlan.created_by == current_user.id)
    if class_id:
        q = q.where(LessonPlan.class_id == uuid.UUID(class_id))
    q = q.order_by(LessonPlan.created_at.desc())
    result = await db.execute(q)
    plans = result.scalars().all()
    return [
        {
            "id": str(p.id),
            "class_id": str(p.class_id),
            "title": p.title,
            "topic": p.topic,
            "objectives": p.objectives,
            "time_estimate": p.time_estimate,
            "steps": p.steps,
            "practice_tasks": p.practice_tasks,
            "formative_check": p.formative_check,
            "homework": p.homework,
            "differentiation": p.differentiation,
            "status": p.status,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in plans
    ]


@router.get("/rubrics")
async def list_teacher_rubrics(
    current_user: CurrentUser,
    db: DBSession,
    class_id: Optional[str] = Query(None),
):
    """List rubrics created by the teacher, optionally filtered by class_id stored in __meta."""
    from app.models.classes import Rubric
    q = select(Rubric).where(Rubric.created_by == current_user.id)
    q = q.order_by(Rubric.created_at.desc())
    result = await db.execute(q)
    rubrics = result.scalars().all()

    def get_meta(criteria) -> dict:
        for c in (criteria or []):
            if isinstance(c, dict) and c.get("__meta"):
                return c
        return {}

    # When class_id is provided, only return rubrics whose __meta.class_id matches
    if class_id:
        rubrics = [
            r for r in rubrics
            if str(get_meta(r.criteria).get("class_id") or "") == class_id
        ]

    return [
        {
            "id": str(r.id),
            "title": r.title,
            "subject": r.subject,
            "board": r.board,
            "grade": r.grade,
            "criteria": r.criteria,
            "is_ai_generated": r.is_ai_generated,
            "difficulty_level": get_meta(r.criteria).get("difficulty_level"),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rubrics
    ]


@router.get("/pending-submissions")
async def get_teacher_pending_submissions(
    current_user: CurrentUser,
    db: DBSession,
):
    """Return submissions awaiting grading (status 'submitted' or 'late') for the teacher's classes."""
    own_result = await db.execute(
        select(Class.id).where(Class.teacher_id == current_user.id, Class.is_active == True)
    )
    co_result = await db.execute(
        select(ClassTeacher.class_id).where(ClassTeacher.teacher_id == current_user.id)
    )
    teacher_class_ids = list(
        set([r[0] for r in own_result.all()] + [r[0] for r in co_result.all()])
    )

    if not teacher_class_ids:
        return {"total": 0, "items": []}

    q = (
        select(Submission, Assignment, Class, User)
        .join(Assignment, Submission.assignment_id == Assignment.id)
        .join(Class, Assignment.class_id == Class.id)
        .join(User, Submission.student_id == User.id)
        .where(
            Assignment.class_id.in_(teacher_class_ids),
            Submission.status.in_(["submitted", "late"]),
        )
        .order_by(Submission.submitted_at.desc())
        .limit(50)
    )
    result = await db.execute(q)
    rows = result.all()

    items = [
        {
            "submission_id": str(sub.id),
            "student_name": student.name,
            "assignment_title": assignment.title,
            "class_name": class_.name,
            "class_id": str(class_.id),
            "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None,
            "is_late": sub.status == "late",
        }
        for sub, assignment, class_, student in rows
    ]
    return {"total": len(items), "items": items}


@router.post("/rubrics", status_code=201)
async def create_teacher_rubric(
    payload: RubricCreateRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Create a rubric from the teacher's rubric builder (class_id/difficulty_level are frontend-only)."""
    from app.models.classes import Rubric
    rubric = Rubric(
        title=payload.title,
        board=payload.board,
        grade=payload.grade,
        subject=payload.subject,
        criteria=payload.criteria or [],
        created_by=current_user.id,
        is_ai_generated=False,
    )
    db.add(rubric)
    await db.commit()
    await db.refresh(rubric)
    return {
        "id": str(rubric.id),
        "title": rubric.title,
        "subject": rubric.subject,
        "board": rubric.board,
        "grade": rubric.grade,
        "criteria": rubric.criteria,
        "difficulty_level": None,
        "created_at": rubric.created_at.isoformat() if rubric.created_at else None,
    }
