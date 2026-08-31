import uuid
from datetime import datetime, timezone, timedelta, date as date_cls
from fastapi import APIRouter, status, Query
from sqlalchemy import select, func

from app.dependencies import DBSession, CurrentUser
from app.models.assessment import PracticeAssessment, AssessmentAttempt, TopicMastery, IntegrityLog
from app.schemas.assessment import (
    AssessmentCreate,
    AssessmentResponse,
    AssessmentListResponse,
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
from app.services.ai_service import AIService, get_ai_service
from app.services.points_service import PointsService

router = APIRouter()


def _parse_org_id(org_id: str | None) -> uuid.UUID | None:
    """Convert org_id string to UUID. Returns None for personal workspace."""
    if not org_id or org_id == "personal":
        return None
    try:
        return uuid.UUID(org_id)
    except ValueError:
        return None


def _apply_org_filter(q, column, org_id_param: str | None):
    """Apply workspace filter to a SQLAlchemy query.
    org_id_param='personal' or None → filter column IS NULL (personal workspace).
    org_id_param=UUID string → filter column == UUID (org workspace).
    """
    parsed = _parse_org_id(org_id_param)
    if parsed is None:
        return q.where(column.is_(None))
    return q.where(column == parsed)


@router.post("/generate", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
async def generate_assessment(payload: GenerateAssessmentRequest, current_user: CurrentUser, db: DBSession):
    """Use AI to generate practice assessment questions and save them."""

    # For daily challenges, reuse today's existing one if available — but only
    # if it actually has questions. If a prior generation produced an empty
    # assessment (AI failure), drop it so the user can regenerate cleanly.
    is_daily = payload.title and payload.title.lower().startswith("daily challenge")
    if is_daily:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        existing_rows = (await db.execute(
            select(PracticeAssessment).where(
                PracticeAssessment.created_by == current_user.id,
                func.lower(PracticeAssessment.title).like("daily challenge%"),
                PracticeAssessment.created_at >= today_start,
            ).order_by(PracticeAssessment.created_at.desc())
        )).scalars().all()
        usable = next(
            (r for r in existing_rows if r.question_json and (not isinstance(r.question_json, list) or len(r.question_json) > 0)),
            None,
        )
        if usable:
            return usable
        # Delete any empty/broken rows so they don't pollute history/status checks
        for row in existing_rows:
            await db.delete(row)
        if existing_rows:
            await db.commit()

    # Check feature access + usage, then deduct points
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="create_assessment", db=db, org_id=_parse_org_id(payload.org_id))
    await points_service.deduct(
        user_id=current_user.id,
        action="generate_assessment",
        db=db,
        org_id=_parse_org_id(payload.org_id),
    )

    ai = get_ai_service()
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

    if not questions or (isinstance(questions, list) and len(questions) == 0):
        raise HTTPException(
            status_code=502,
            detail="The AI couldn't generate questions right now. Please try again in a moment.",
        )

    assessment = PracticeAssessment(
        created_by=current_user.id,
        org_id=_parse_org_id(payload.org_id),
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


@router.get("/daily-challenge-status")
async def daily_challenge_status(current_user: CurrentUser, db: DBSession):
    """Check if user has a daily challenge today and its completion status."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Find today's daily challenge
    result = await db.execute(
        select(PracticeAssessment).where(
            PracticeAssessment.created_by == current_user.id,
            func.lower(PracticeAssessment.title).like("daily challenge%"),
            PracticeAssessment.created_at >= today_start,
        ).order_by(PracticeAssessment.created_at.desc()).limit(1)
    )
    assessment = result.scalar_one_or_none()
    if not assessment:
        return {"status": "not_started", "quiz_id": None}

    # If the assessment row exists but has no usable questions, treat it as
    # not_started so the user can generate a fresh one instead of getting stuck.
    qjson = assessment.question_json
    if not qjson or (isinstance(qjson, list) and len(qjson) == 0):
        return {"status": "not_started", "quiz_id": None}

    # Check if there's a submitted attempt
    attempt_result = await db.execute(
        select(AssessmentAttempt).where(
            AssessmentAttempt.assessment_id == assessment.id,
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        ).limit(1)
    )
    attempt = attempt_result.scalar_one_or_none()
    if attempt:
        return {"status": "completed", "quiz_id": str(assessment.id), "score": attempt.percentage, "xp_earned": attempt.xp_earned}

    return {"status": "in_progress", "quiz_id": str(assessment.id)}


@router.get("/daily-challenge-history")
async def daily_challenge_history(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
    days: int = Query(60, ge=1, le=365),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None, alias="to"),
):
    """Return daily-challenge attempts for the current user with missed days synthesized.

    Window defaults to the last `days` days but can be overridden via from/to (YYYY-MM-DD).
    Used by the student "Daily Challenge History" page.
    """
    # Resolve window
    today = datetime.now(timezone.utc).date()
    try:
        end_date = date_cls.fromisoformat(to) if to else today
        start_date = date_cls.fromisoformat(from_) if from_ else (end_date - timedelta(days=days - 1))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid from/to date; expected YYYY-MM-DD")
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)

    # Pull all daily challenge assessments + attempts in window for this user
    q = (
        select(PracticeAssessment, AssessmentAttempt)
        .join(AssessmentAttempt, AssessmentAttempt.assessment_id == PracticeAssessment.id, isouter=True)
        .where(
            PracticeAssessment.created_by == current_user.id,
            func.lower(PracticeAssessment.title).like("daily challenge%"),
            PracticeAssessment.created_at >= start_dt,
            PracticeAssessment.created_at <= end_dt,
        )
    )
    if org_id is not None:
        q = _apply_org_filter(q, PracticeAssessment.org_id, org_id)
    q = q.order_by(PracticeAssessment.created_at.asc())
    result = await db.execute(q)
    rows = result.all()

    # Bucket by date — keep the latest evaluated attempt per day, else any in_progress
    by_date: dict[str, dict] = {}
    for assessment, attempt in rows:
        d = assessment.created_at.astimezone(timezone.utc).date().isoformat()
        existing = by_date.get(d)
        is_eval = bool(attempt and attempt.status == "evaluated")
        if existing and existing.get("is_eval") and not is_eval:
            continue
        questions_payload = []
        if attempt and attempt.status == "evaluated":
            qjson = assessment.question_json or []
            responses = attempt.responses_json or {}
            feedback = attempt.feedback_json or {}
            if isinstance(qjson, list):
                for q_obj in qjson:
                    qid = str(q_obj.get("id", ""))
                    fb = feedback.get(qid) or {}
                    is_correct = bool(fb.get("correct", False))
                    questions_payload.append({
                        "id": qid,
                        "prompt": q_obj.get("text") or q_obj.get("question") or q_obj.get("prompt") or "",
                        "topic": q_obj.get("topic"),
                        "studentAnswer": responses.get(qid),
                        "correctAnswer": q_obj.get("correctAnswer") or q_obj.get("correct_answer") or q_obj.get("answer") or "",
                        "isCorrect": is_correct,
                        "explanation": fb.get("explanation") or q_obj.get("explanation"),
                    })
        by_date[d] = {
            "is_eval": is_eval,
            "payload": {
                "date": d,
                "status": "completed" if is_eval else ("in_progress" if attempt else "in_progress"),
                "assessmentId": str(assessment.id),
                "attemptId": str(attempt.id) if attempt else None,
                "score": float(attempt.percentage) if (attempt and attempt.percentage is not None) else None,
                "maxScore": float(attempt.max_score) if (attempt and attempt.max_score is not None) else None,
                "xpEarned": int(attempt.xp_earned) if attempt else 0,
                "subject": assessment.subject,
                "topics": assessment.topics if isinstance(assessment.topics, list) else None,
                "questions": questions_payload,
            },
        }

    # Synthesize the full date range with missed/future
    attempts_out: list[dict] = []
    cursor = start_date
    while cursor <= end_date:
        ds = cursor.isoformat()
        if ds in by_date:
            attempts_out.append(by_date[ds]["payload"])
        else:
            attempts_out.append({
                "date": ds,
                "status": "future" if cursor > today else "missed",
                "assessmentId": None,
                "attemptId": None,
                "score": None,
                "xpEarned": 0,
                "questions": [],
            })
        cursor += timedelta(days=1)

    # Stats
    completed = [a for a in attempts_out if a["status"] == "completed"]
    past_days = [a for a in attempts_out if a["status"] != "future"]
    missed = [a for a in past_days if a["status"] == "missed"]
    total_past = len(past_days)
    completion_rate = round((len(completed) / total_past) * 100) if total_past else 0
    avg_score = round(sum((a.get("score") or 0) for a in completed) / len(completed)) if completed else 0
    total_xp = sum(int(a.get("xpEarned") or 0) for a in completed)

    # Streaks (computed over past_days only, chronological)
    best = cur = 0
    for a in past_days:
        if a["status"] == "completed":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    # Current streak counts back from today (or most recent past day)
    current_streak = 0
    for a in reversed(past_days):
        if a["status"] == "completed":
            current_streak += 1
        else:
            break

    # Topic breakdown across question-level data
    topic_agg: dict[str, dict] = {}
    for a in completed:
        for q_item in a.get("questions") or []:
            topic = (q_item.get("topic") or "Other").strip() or "Other"
            t = topic_agg.setdefault(topic, {"correct": 0, "total": 0})
            t["total"] += 1
            if q_item.get("isCorrect"):
                t["correct"] += 1
    topic_breakdown = [
        {
            "topic": k,
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": round((v["correct"] / v["total"]) * 100) if v["total"] else 0,
        }
        for k, v in sorted(topic_agg.items(), key=lambda kv: kv[1]["total"], reverse=True)
    ]

    return {
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "attempts": attempts_out,
        "stats": {
            "totalDays": total_past,
            "completedDays": len(completed),
            "missedDays": len(missed),
            "completionRate": completion_rate,
            "currentStreak": current_streak,
            "bestStreak": best,
            "avgScore": avg_score,
            "totalXp": total_xp,
        },
        "topicBreakdown": topic_breakdown,
    }


@router.post("/", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
async def save_assessment(payload: AssessmentSaveRequest, current_user: CurrentUser, db: DBSession):
    """Save a reviewed/edited set of questions to the library (no AI generation)."""
    assessment = PracticeAssessment(
        created_by=current_user.id,
        org_id=_parse_org_id(payload.org_id),
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


@router.get("/", response_model=AssessmentListResponse)
async def list_assessments(
    current_user: CurrentUser,
    db: DBSession,
    subject: str | None = Query(None),
    org_id: str | None = Query(None, description="'personal' or org UUID for workspace filtering"),
    limit: int = Query(12, ge=1, le=50),
    cursor: str | None = Query(None, description="ISO datetime cursor for pagination"),
):
    from datetime import datetime as dt
    from sqlalchemy import func as sa_func

    base = select(PracticeAssessment).where(PracticeAssessment.created_by == current_user.id)
    if org_id is not None:
        base = _apply_org_filter(base, PracticeAssessment.org_id, org_id)
    if subject:
        base = base.where(PracticeAssessment.subject == subject)

    count_q = select(sa_func.count()).select_from(base.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    q = base
    if cursor:
        try:
            cursor_dt = dt.fromisoformat(cursor)
            q = q.where(PracticeAssessment.created_at < cursor_dt)
        except (ValueError, TypeError):
            pass

    q = q.order_by(PracticeAssessment.created_at.desc()).limit(limit + 1)
    result = await db.execute(q)
    rows = result.scalars().all()

    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].created_at.isoformat() if has_more and items else None

    return AssessmentListResponse(items=items, next_cursor=next_cursor, has_more=has_more, total=total)


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
    org_id: str | None = Query(None),
):
    """Return only evaluated (submitted) attempts by the current user, newest first.
    Joins PracticeAssessment to include title, subject, difficulty, and topics."""
    q = (
        select(AssessmentAttempt)
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
    )
    if org_id is not None:
        q = _apply_org_filter(q, PracticeAssessment.org_id, org_id)
    q = q.order_by(AssessmentAttempt.started_at.desc()).limit(limit)
    result = await db.execute(q)
    attempts = result.scalars().all()

    # Enrich each attempt with assessment metadata
    assessment_ids = list({a.assessment_id for a in attempts})
    assessments_result = await db.execute(
        select(PracticeAssessment).where(PracticeAssessment.id.in_(assessment_ids))
    )
    assessments_map = {a.id: a for a in assessments_result.scalars().all()}

    enriched = []
    for attempt in attempts:
        assessment = assessments_map.get(attempt.assessment_id)
        data = AttemptResponse.model_validate(attempt)
        if assessment:
            data.title = assessment.title
            data.subject = assessment.subject
            data.difficulty = assessment.difficulty
            data.topics = assessment.topics
        enriched.append(data)

    return enriched


@router.get("/trends")
async def get_assessment_trends(
    current_user: CurrentUser,
    db: DBSession,
    subject: str | None = Query(None),
    org_id: str | None = Query(None),
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
    if org_id is not None:
        q = _apply_org_filter(q, PracticeAssessment.org_id, org_id)
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
    org_id: str | None = Query(None),
):
    """Return topic mastery data for the current user."""
    q = select(TopicMastery).where(TopicMastery.user_id == current_user.id)
    if org_id is not None:
        q = _apply_org_filter(q, TopicMastery.org_id, org_id)
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
    org_id: str | None = Query(None),
):
    """Return all assessment attempts by the current user."""
    q = select(AssessmentAttempt).where(AssessmentAttempt.user_id == current_user.id)
    if org_id is not None:
        q = q.join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        q = _apply_org_filter(q, PracticeAssessment.org_id, org_id)
    q = q.order_by(AssessmentAttempt.started_at.desc()).limit(limit)
    result = await db.execute(q)
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

    ai = get_ai_service()
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
    is_daily = assessment.title and assessment.title.lower().startswith("daily challenge")
    attempt.xp_earned = 5 if is_daily else 20

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


@router.patch("/{assessment_id}", response_model=AssessmentResponse)
async def update_assessment(assessment_id: uuid.UUID, payload: dict, current_user: CurrentUser, db: DBSession):
    """Update assessment fields (question_json, question_count, title, etc.)."""
    result = await db.execute(
        select(PracticeAssessment).where(
            PracticeAssessment.id == assessment_id,
            PracticeAssessment.created_by == current_user.id,
        )
    )
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")

    allowed_fields = {
        "title", "subject", "difficulty", "mode", "question_json",
        "question_count", "answer_key_json", "time_limit",
        "negative_marking", "negative_mark_value", "topics",
    }
    for key, value in payload.items():
        if key in allowed_fields and hasattr(assessment, key):
            setattr(assessment, key, value)

    # If question_json updated but question_count not provided, sync it
    if "question_json" in payload and "question_count" not in payload:
        qj = payload["question_json"]
        if isinstance(qj, list):
            assessment.question_count = len(qj)

    await db.commit()
    await db.refresh(assessment)
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
    ai = get_ai_service()
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
    is_daily = assessment.title and assessment.title.lower().startswith("daily challenge")
    attempt.xp_earned = 5 if is_daily else 20

    # Update topic mastery
    await _update_topic_mastery(current_user.id, assessment, evaluation, db)

    # Award XP to user
    current_user.xp = (current_user.xp or 0) + attempt.xp_earned

    await db.commit()
    await db.refresh(attempt)
    return attempt


async def _update_topic_mastery(user_id, assessment, evaluation, db):
    topics = assessment.topics or [assessment.subject]
    assessment_org_id = getattr(assessment, "org_id", None)
    for topic in topics:
        result = await db.execute(
            select(TopicMastery).where(
                TopicMastery.user_id == user_id,
                TopicMastery.subject == assessment.subject,
                TopicMastery.topic == topic,
                TopicMastery.org_id == assessment_org_id if assessment_org_id else TopicMastery.org_id.is_(None),
            )
        )
        mastery = result.scalar_one_or_none()
        percentage = evaluation.get("percentage", 0)
        if not mastery:
            mastery = TopicMastery(
                user_id=user_id,
                org_id=assessment_org_id,
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
    org_id: str | None = Query(None),
):
    q = select(TopicMastery).where(TopicMastery.user_id == current_user.id)
    if org_id is not None:
        q = _apply_org_filter(q, TopicMastery.org_id, org_id)
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
