import uuid
from fastapi import APIRouter, status, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.classes import LessonPlan, Class
from app.schemas.classes import LessonPlanRequest, LessonPlanResponse
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.ai_service import AIService

router = APIRouter()


@router.post("/generate", response_model=LessonPlanResponse, status_code=status.HTTP_201_CREATED)
async def generate_lesson_plan(payload: LessonPlanRequest, current_user: CurrentUser, db: DBSession):
    """Use AI to generate a structured lesson plan for a class topic."""
    class_result = await db.execute(select(Class).where(Class.id == uuid.UUID(payload.class_id)))
    class_ = class_result.scalar_one_or_none()
    if not class_:
        raise NotFoundException("Class not found")

    ai = AIService()
    plan_data = await ai.generate_lesson_plan(
        class_id=payload.class_id,
        topic=payload.topic,
        board=class_.board,
        grade=class_.grade,
        subject=class_.subject,
        additional_context=payload.additional_context,
    )

    lesson_plan = LessonPlan(
        class_id=class_.id,
        created_by=current_user.id,
        title=plan_data.get("title", f"Lesson Plan: {payload.topic}"),
        topic=payload.topic,
        objectives=plan_data.get("objectives"),
        time_estimate=plan_data.get("timeEstimate"),
        steps=plan_data.get("steps"),
        practice_tasks=plan_data.get("practiceTasks"),
        formative_check=plan_data.get("formativeCheck"),
        homework=plan_data.get("homework"),
        differentiation=plan_data.get("differentiation"),
        status="draft",
    )
    db.add(lesson_plan)
    await db.commit()
    await db.refresh(lesson_plan)
    return lesson_plan


@router.get("/", response_model=list[LessonPlanResponse])
async def list_lesson_plans(
    current_user: CurrentUser,
    db: DBSession,
    class_id: str | None = Query(None),
):
    q = select(LessonPlan).where(LessonPlan.created_by == current_user.id)
    if class_id:
        q = q.where(LessonPlan.class_id == uuid.UUID(class_id))
    q = q.order_by(LessonPlan.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{plan_id}", response_model=LessonPlanResponse)
async def get_lesson_plan(plan_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(LessonPlan).where(LessonPlan.id == plan_id))
    plan = result.scalar_one_or_none()
    if not plan:
        raise NotFoundException("Lesson plan not found")
    return plan


@router.patch("/{plan_id}/publish")
async def publish_lesson_plan(plan_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(LessonPlan).where(LessonPlan.id == plan_id))
    plan = result.scalar_one_or_none()
    if not plan:
        raise NotFoundException("Lesson plan not found")
    if plan.created_by != current_user.id:
        raise ForbiddenException("Not your lesson plan")
    plan.status = "published"
    await db.commit()
    return {"message": "Lesson plan published"}


@router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lesson_plan(plan_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(select(LessonPlan).where(LessonPlan.id == plan_id))
    plan = result.scalar_one_or_none()
    if not plan:
        raise NotFoundException("Lesson plan not found")
    if plan.created_by != current_user.id:
        raise ForbiddenException("Not your lesson plan")
    await db.delete(plan)
    await db.commit()
