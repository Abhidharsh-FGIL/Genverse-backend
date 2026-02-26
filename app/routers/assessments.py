import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, status, Query
from sqlalchemy import select, func

from app.dependencies import DBSession, CurrentUser
from app.models.assessment import PracticeAssessment, AssessmentAttempt, TopicMastery, IntegrityLog
from app.schemas.assessment import (
    AssessmentCreate,
    AssessmentResponse,
    AttemptStartResponse,
    AttemptSubmitRequest,
    AttemptResponse,
    TopicMasteryResponse,
    TopicMasteryUpsert,
    IntegrityEventRequest,
    GenerateAssessmentRequest,
    AssessmentSaveRequest,
)
from app.core.exceptions import NotFoundException, ConflictException
from fastapi import HTTPException
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()


@router.post("/generate", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
async def generate_assessment(payload: GenerateAssessmentRequest, current_user: CurrentUser, db: DBSession):
    """Use AI to generate practice assessment questions and save them."""
    # Deduct points: 2 pts per 10 questions
    points_service = PointsService()
    await points_service.deduct(
        user_id=current_user.id,
        action="generate_assessment",
        db=db,
    )

    ai = AIService()
    questions = await ai.generate_practice_assessment(
        subject=payload.subject,
        topics=payload.topics,
        grade=payload.grade,
        board=payload.board,
        difficulty=payload.difficulty,
        question_count=payload.question_count,
        question_types=payload.question_types,
        mode=payload.mode,
    )

    assessment = PracticeAssessment(
        created_by=current_user.id,
        title=payload.title,
        subject=payload.subject,
        board=payload.board,
        grade=payload.grade,
        topics=payload.topics,
        difficulty=payload.difficulty,
        mode=payload.mode,
        question_count=payload.question_count,
        question_types=payload.question_types,
        question_json=questions,
        time_limit=payload.time_limit,
        is_adaptive=payload.is_adaptive,
        negative_marking=payload.negative_marking,
        negative_mark_value=payload.negative_mark_value,
    )
    db.add(assessment)
    await db.commit()
    await db.refresh(assessment)
    return assessment


@router.post("/", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
async def save_assessment(payload: AssessmentSaveRequest, current_user: CurrentUser, db: DBSession):
    """Save a reviewed/edited set of questions to the library (no AI generation)."""
    assessment = PracticeAssessment(
        created_by=current_user.id,
        title=payload.title,
        subject=payload.subject or "",
        board=payload.board,
        grade=payload.grade,
        topics=payload.topics,
        difficulty=payload.difficulty,
        mode=payload.mode,
        question_count=len(payload.questions),
        question_json=payload.questions,
        time_limit=payload.time_limit,
        negative_marking=payload.negative_marking,
        negative_mark_value=payload.negative_mark_value,
    )
    db.add(assessment)
    await db.commit()
    await db.refresh(assessment)
    return assessment


@router.get("/", response_model=list[AssessmentResponse])
async def list_assessments(
    current_user: CurrentUser,
    db: DBSession,
    subject: str | None = Query(None),
):
    q = select(PracticeAssessment).where(PracticeAssessment.created_by == current_user.id)
    if subject:
        q = q.where(PracticeAssessment.subject == subject)
    q = q.order_by(PracticeAssessment.created_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


# ── Static routes MUST be defined before /{assessment_id} ──────────────────


@router.post("/complete")
async def complete_assessment(payload: dict, current_user: CurrentUser, db: DBSession):
    """Compute enriched post-assessment data: improvement index, mastery snapshot, past scores, recommendations."""
    attempt_id = payload.get("attempt_id")
    assessment_id = payload.get("assessment_id")
    percentage = float(payload.get("percentage") or 0)
    subject = payload.get("subject") or ""

    # Past scores for this assessment (excluding current attempt)
    past_result = await db.execute(
        select(AssessmentAttempt.percentage)
        .where(
            AssessmentAttempt.assessment_id == assessment_id,
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
            AssessmentAttempt.id != attempt_id,
        )
        .order_by(AssessmentAttempt.submitted_at.desc())
        .limit(10)
    )
    past_scores = [float(s) for s in past_result.scalars().all() if s is not None]

    avg = sum(past_scores) / len(past_scores) if past_scores else None
    improvement_index = round(percentage - avg, 1) if avg is not None else 0

    # Bonus XP for high scores (awarded in addition to base XP from submit)
    bonus_xp = 10 if percentage >= 90 else 5 if percentage >= 75 else 0
    if bonus_xp:
        current_user.xp = (current_user.xp or 0) + bonus_xp
        await db.commit()

    # Topic mastery snapshot for this subject
    mastery_result = await db.execute(
        select(TopicMastery).where(
            TopicMastery.user_id == current_user.id,
            TopicMastery.subject == subject,
        )
    )
    mastery_records = mastery_result.scalars().all()
    mastery_snapshot = {m.topic: round(m.mastery_level / 10, 1) for m in mastery_records}

    # Recommendations based on score
    recommendations = []
    if percentage < 50:
        recommendations.append({"type": "retry", "message": f"Score below 50% — review {subject or 'this topic'} and retry."})
        recommendations.append({"type": "weak_topic", "message": "Focus on the questions you got wrong before attempting again."})
    elif percentage >= 80:
        recommendations.append({"type": "difficulty_upgrade", "message": "Excellent! Try a harder difficulty to keep challenging yourself."})
        recommendations.append({"type": "strength", "message": f"You have a strong grasp of {subject or 'this subject'}. Keep it up!"})
    else:
        recommendations.append({"type": "improvement", "message": f"Good effort! Review your incorrect answers to improve further."})
        recommendations.append({"type": "practice_more", "message": "Practice a few more assessments to solidify your understanding."})

    return {
        "bonus_xp": bonus_xp,
        "improvement_index": improvement_index,
        "mastery_snapshot": mastery_snapshot,
        "past_scores": past_scores[:5],
        "recommendations": recommendations,
    }


@router.get("/history", response_model=list[AttemptResponse])
async def get_attempt_history(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(50, le=200),
):
    """Return only evaluated (submitted) attempts by the current user, newest first."""
    result = await db.execute(
        select(AssessmentAttempt)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
        .order_by(AssessmentAttempt.started_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/trends")
async def get_assessment_trends(
    current_user: CurrentUser,
    db: DBSession,
    subject: str | None = Query(None),
):
    """Return per-subject score trend aggregates for the current user."""
    q = (
        select(
            PracticeAssessment.subject,
            func.count(AssessmentAttempt.id).label("attempt_count"),
            func.avg(AssessmentAttempt.percentage).label("average_score"),
            func.max(AssessmentAttempt.percentage).label("best_score"),
        )
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
        .group_by(PracticeAssessment.subject)
    )
    if subject:
        q = q.where(PracticeAssessment.subject == subject)
    result = await db.execute(q)
    rows = result.all()
    return [
        {
            "subject": r.subject,
            "attempt_count": r.attempt_count,
            "average_score": round(float(r.average_score or 0), 2),
            "best_score": round(float(r.best_score or 0), 2),
        }
        for r in rows
    ]


@router.get("/mastery", response_model=list[TopicMasteryResponse])
async def get_mastery(
    current_user: CurrentUser,
    db: DBSession,
    subject: str | None = Query(None),
):
    """Return topic mastery data for the current user."""
    q = select(TopicMastery).where(TopicMastery.user_id == current_user.id)
    if subject:
        q = q.where(TopicMastery.subject == subject)
    result = await db.execute(q)
    return result.scalars().all()


@router.post("/mastery", response_model=TopicMasteryResponse, status_code=status.HTTP_201_CREATED)
async def upsert_mastery(payload: TopicMasteryUpsert, current_user: CurrentUser, db: DBSession):
    """Create or update a topic mastery record."""
    result = await db.execute(
        select(TopicMastery).where(
            TopicMastery.user_id == current_user.id,
            TopicMastery.subject == payload.subject,
            TopicMastery.topic == payload.topic,
        )
    )
    mastery = result.scalar_one_or_none()
    if mastery:
        mastery.mastery_level = payload.mastery_level
        mastery.attempts_count = payload.total_attempts
        mastery.correct_count = payload.correct_count
    else:
        mastery = TopicMastery(
            user_id=current_user.id,
            subject=payload.subject,
            topic=payload.topic,
            mastery_level=payload.mastery_level,
            attempts_count=payload.total_attempts,
            correct_count=payload.correct_count,
        )
        db.add(mastery)
    await db.commit()
    await db.refresh(mastery)
    return mastery


@router.get("/attempts", response_model=list[AttemptResponse])
async def get_all_attempts(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(50, le=200),
):
    """Return all assessment attempts by the current user."""
    result = await db.execute(
        select(AssessmentAttempt)
        .where(AssessmentAttempt.user_id == current_user.id)
        .order_by(AssessmentAttempt.started_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.get("/attempts/{attempt_id}", response_model=AttemptResponse)
async def get_attempt_by_id(
    attempt_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
):
    """Return a single attempt by its ID."""
    result = await db.execute(
        select(AssessmentAttempt).where(
            AssessmentAttempt.id == attempt_id,
            AssessmentAttempt.user_id == current_user.id,
        )
    )
    attempt = result.scalar_one_or_none()
    if not attempt:
        raise NotFoundException("Attempt not found")
    return attempt


@router.post("/attempts/{attempt_id}/submit", response_model=AttemptResponse)
async def submit_attempt_by_id(
    attempt_id: uuid.UUID,
    payload: AttemptSubmitRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Submit an attempt by attempt ID only (no assessment_id needed in path)."""
    attempt_result = await db.execute(
        select(AssessmentAttempt).where(
            AssessmentAttempt.id == attempt_id,
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "in_progress",
        )
    )
    attempt = attempt_result.scalar_one_or_none()
    if not attempt:
        raise NotFoundException("Attempt not found or already submitted")

    # If no questions were answered, delete the abandoned attempt and reject
    has_answers = payload.responses and any(v.strip() for v in payload.responses.values() if isinstance(v, str))
    if not has_answers:
        await db.delete(attempt)
        await db.commit()
        raise HTTPException(status_code=400, detail="No questions were answered. Attempt discarded.")

    assessment_result = await db.execute(
        select(PracticeAssessment).where(PracticeAssessment.id == attempt.assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")

    ai = AIService()
    evaluation = await ai.auto_evaluate_attempt(
        questions=assessment.question_json,
        responses=payload.responses,
    )

    attempt.responses_json = payload.responses
    attempt.score = evaluation.get("score", 0)
    attempt.max_score = evaluation.get("max_score", assessment.question_count)

    # Apply negative marking: deduct per wrong-but-answered question
    if assessment.negative_marking and assessment.negative_mark_value:
        feedback = evaluation.get("feedback", {})
        wrong_count = sum(
            1 for qId, fb in feedback.items()
            if not fb.get("correct") and str(payload.responses.get(qId, "")).strip()
        )
        attempt.score = max(0.0, attempt.score - wrong_count * assessment.negative_mark_value)

    attempt.percentage = round((attempt.score / attempt.max_score * 100), 2) if attempt.max_score else 0
    attempt.feedback_json = evaluation.get("feedback")
    attempt.submitted_at = datetime.now(timezone.utc)
    attempt.status = "evaluated"
    attempt.xp_earned = 20

    await _update_topic_mastery(current_user.id, assessment, evaluation, db)
    current_user.xp = (current_user.xp or 0) + attempt.xp_earned

    await db.commit()
    await db.refresh(attempt)
    return attempt


# ── Path-param routes (must come after all static routes) ──────────────────


@router.get("/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment(assessment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(PracticeAssessment).where(
            PracticeAssessment.id == assessment_id,
            PracticeAssessment.created_by == current_user.id,
        )
    )
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")
    return assessment


@router.delete("/{assessment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assessment(assessment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(PracticeAssessment).where(
            PracticeAssessment.id == assessment_id,
            PracticeAssessment.created_by == current_user.id,
        )
    )
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")
    await db.delete(assessment)
    await db.commit()


@router.post("/{assessment_id}/start", response_model=AttemptStartResponse)
async def start_attempt(assessment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(PracticeAssessment).where(PracticeAssessment.id == assessment_id)
    )
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")

    attempt = AssessmentAttempt(
        assessment_id=assessment_id,
        user_id=current_user.id,
        status="in_progress",
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)
    return attempt


@router.post("/{assessment_id}/attempts/{attempt_id}/submit", response_model=AttemptResponse)
async def submit_attempt(
    assessment_id: uuid.UUID,
    attempt_id: uuid.UUID,
    payload: AttemptSubmitRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    attempt_result = await db.execute(
        select(AssessmentAttempt).where(
            AssessmentAttempt.id == attempt_id,
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "in_progress",
        )
    )
    attempt = attempt_result.scalar_one_or_none()
    if not attempt:
        raise NotFoundException("Attempt not found or already submitted")

    # If no questions were answered, delete the abandoned attempt and reject
    has_answers = payload.responses and any(v.strip() for v in payload.responses.values() if isinstance(v, str))
    if not has_answers:
        await db.delete(attempt)
        await db.commit()
        raise HTTPException(status_code=400, detail="No questions were answered. Attempt discarded.")

    assessment_result = await db.execute(
        select(PracticeAssessment).where(PracticeAssessment.id == assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()

    # Auto-evaluate using AI
    ai = AIService()
    evaluation = await ai.auto_evaluate_attempt(
        questions=assessment.question_json,
        responses=payload.responses,
    )

    attempt.responses_json = payload.responses
    attempt.score = evaluation.get("score", 0)
    attempt.max_score = evaluation.get("max_score", assessment.question_count)

    # Apply negative marking: deduct per wrong-but-answered question
    if assessment.negative_marking and assessment.negative_mark_value:
        feedback = evaluation.get("feedback", {})
        wrong_count = sum(
            1 for qId, fb in feedback.items()
            if not fb.get("correct") and str(payload.responses.get(qId, "")).strip()
        )
        attempt.score = max(0.0, attempt.score - wrong_count * assessment.negative_mark_value)

    attempt.percentage = round((attempt.score / attempt.max_score * 100), 2) if attempt.max_score else 0
    attempt.feedback_json = evaluation.get("feedback")
    attempt.submitted_at = datetime.now(timezone.utc)
    attempt.status = "evaluated"
    attempt.xp_earned = 20

    # Update topic mastery
    await _update_topic_mastery(current_user.id, assessment, evaluation, db)

    # Award XP to user
    current_user.xp = (current_user.xp or 0) + attempt.xp_earned

    await db.commit()
    await db.refresh(attempt)
    return attempt


async def _update_topic_mastery(user_id, assessment, evaluation, db):
    topics = assessment.topics or [assessment.subject]
    for topic in topics:
        result = await db.execute(
            select(TopicMastery).where(
                TopicMastery.user_id == user_id,
                TopicMastery.subject == assessment.subject,
                TopicMastery.topic == topic,
            )
        )
        mastery = result.scalar_one_or_none()
        percentage = evaluation.get("percentage", 0)
        if not mastery:
            mastery = TopicMastery(
                user_id=user_id,
                subject=assessment.subject,
                topic=topic,
                mastery_level=percentage,
                attempts_count=1,
                correct_count=1 if percentage >= 50 else 0,
            )
            db.add(mastery)
        else:
            old = mastery.mastery_level
            mastery.mastery_level = (mastery.mastery_level + percentage) / 2
            mastery.attempts_count += 1
            if percentage >= 50:
                mastery.correct_count += 1
            mastery.trend = "improving" if mastery.mastery_level > old else "declining" if mastery.mastery_level < old else "stable"
            mastery.last_attempted_at = datetime.now(timezone.utc)


@router.get("/{assessment_id}/attempts", response_model=list[AttemptResponse])
async def list_attempts(assessment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(AssessmentAttempt).where(
            AssessmentAttempt.assessment_id == assessment_id,
            AssessmentAttempt.user_id == current_user.id,
        ).order_by(AssessmentAttempt.started_at.desc())
    )
    return result.scalars().all()


@router.get("/mastery/topics", response_model=list[TopicMasteryResponse])
async def get_topic_mastery(
    current_user: CurrentUser,
    db: DBSession,
    subject: str | None = Query(None),
):
    q = select(TopicMastery).where(TopicMastery.user_id == current_user.id)
    if subject:
        q = q.where(TopicMastery.subject == subject)
    result = await db.execute(q)
    return result.scalars().all()


@router.post("/integrity/log")
async def log_integrity_event(payload: IntegrityEventRequest, current_user: CurrentUser, db: DBSession):
    log = IntegrityLog(
        user_id=current_user.id,
        attempt_id=uuid.UUID(payload.attempt_id),
        event_type=payload.event_type,
        event_data=payload.event_data,
    )
    db.add(log)
    await db.commit()
    return {"message": "Integrity event logged"}
