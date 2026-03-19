import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, status, Query
from sqlalchemy import select, func

from app.dependencies import DBSession, CurrentUser
from app.models.insights import UserInsight, InsightArticle, Recommendation
from app.models.ai import IntelligenceCache
from app.models.assessment import PracticeAssessment
from app.models.gamification import WorkspaceGamification
from app.schemas.insights import (
    UserInsightResponse,
    InsightArticleResponse,
    GenerateInsightsRequest,
    IntelligenceRequest,
    IntelligenceResponse,
    RecommendationResponse,
    LearningCurveResponse,
    ClassRecommendationRequest,
    ClassRecommendationItem,
)
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.points_service import PointsService
from app.services.news_service import NewsService

router = APIRouter()


def _parse_org_id(org_id: str | None):
    if not org_id or org_id == "personal":
        return None
    try:
        return uuid.UUID(org_id)
    except ValueError:
        return None


def _apply_org_filter(q, column, org_id_param: str | None):
    parsed = _parse_org_id(org_id_param)
    if parsed is None:
        return q.where(column.is_(None))
    return q.where(column == parsed)


@router.post("/generate", response_model=list[UserInsightResponse])
async def generate_insights(
    payload: GenerateInsightsRequest,
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Generate personalized learning insights using AI."""
    parsed_oid = _parse_org_id(org_id)

    if not payload.force_refresh:
        # Check for recent insights
        q = select(UserInsight).where(UserInsight.user_id == current_user.id)
        if org_id is not None:
            q = _apply_org_filter(q, UserInsight.org_id, org_id)
        q = q.order_by(UserInsight.created_at.desc()).limit(5)
        result = await db.execute(q)
        existing = result.scalars().all()
        if existing:
            return existing
    else:
        # Delete existing unread insights so regeneration replaces rather than accumulates
        from sqlalchemy import delete as sql_delete
        del_q = sql_delete(UserInsight).where(
            UserInsight.user_id == current_user.id,
            UserInsight.is_read == False,
        )
        if org_id is not None:
            if parsed_oid:
                del_q = del_q.where(UserInsight.org_id == parsed_oid)
            else:
                del_q = del_q.where(UserInsight.org_id.is_(None))
        await db.execute(del_q)
        await db.commit()

    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="insights", db=db, org_id=parsed_oid)
    await points_service.deduct(user_id=current_user.id, action="generate_insights", db=db)

    ai = AIService()
    insights_data = await ai.generate_insights(user_id=str(current_user.id), db=db)

    new_insights = []
    for item in insights_data:
        insight = UserInsight(
            user_id=current_user.id,
            org_id=parsed_oid,
            insight_type=item.get("type", "general"),
            title=item.get("title"),
            content=item.get("content"),
            data_json=item.get("data"),
        )
        db.add(insight)
        new_insights.append(insight)

    await db.commit()
    for insight in new_insights:
        await db.refresh(insight)
    return new_insights


@router.get("/", response_model=list[UserInsightResponse])
async def list_insights(
    current_user: CurrentUser,
    db: DBSession,
    unread_only: bool = Query(False),
    limit: int = Query(20, le=100),
    org_id: str | None = Query(None),
):
    from app.models.assessment import AssessmentAttempt

    parsed_oid = _parse_org_id(org_id)

    q = select(UserInsight).where(UserInsight.user_id == current_user.id)
    if org_id is not None:
        q = _apply_org_filter(q, UserInsight.org_id, org_id)
    if unread_only:
        q = q.where(UserInsight.is_read == False)
    q = q.order_by(UserInsight.created_at.desc()).limit(limit)
    result = await db.execute(q)
    insights = result.scalars().all()

    # Auto-generate insights on first load if none exist and the user has real assessment data
    if not insights and not unread_only:
        attempt_count = await db.execute(
            select(func.count(AssessmentAttempt.id)).where(
                AssessmentAttempt.user_id == current_user.id,
                AssessmentAttempt.status == "evaluated",
            )
        )
        if attempt_count.scalar_one() > 0:
            ai = AIService()
            insights_data = await ai.generate_insights(user_id=str(current_user.id), db=db)
            new_insights = []
            for item in insights_data:
                insight = UserInsight(
                    user_id=current_user.id,
                    org_id=parsed_oid,
                    insight_type=item.get("type", "general"),
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    data_json=item.get("data"),
                )
                db.add(insight)
                new_insights.append(insight)
            if new_insights:
                await db.commit()
                for ins in new_insights:
                    await db.refresh(ins)
            insights = new_insights

    return insights


@router.patch("/{insight_id}/read")
async def mark_insight_read(insight_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(UserInsight).where(UserInsight.id == insight_id, UserInsight.user_id == current_user.id)
    )
    insight = result.scalar_one_or_none()
    if not insight:
        raise NotFoundException("Insight not found")
    insight.is_read = True
    await db.commit()
    return {"message": "Insight marked as read"}


@router.patch("/{insight_id}")
async def dismiss_insight(insight_id: uuid.UUID, payload: dict, current_user: CurrentUser, db: DBSession):
    """Mark an insight as read/dismissed."""
    result = await db.execute(
        select(UserInsight).where(UserInsight.id == insight_id, UserInsight.user_id == current_user.id)
    )
    insight = result.scalar_one_or_none()
    if not insight:
        raise NotFoundException("Insight not found")
    insight.is_read = True
    await db.commit()
    return {"message": "Insight dismissed"}


@router.get("/feed", response_model=list[InsightArticleResponse])
async def get_insight_feed(
    current_user: CurrentUser,
    db: DBSession,
    subject: str | None = Query(None),
    limit: int = Query(10, le=50),
    org_id: str | None = Query(None),
):
    parsed_oid = _parse_org_id(org_id)

    q = select(InsightArticle).where(InsightArticle.user_id == current_user.id)
    if org_id is not None:
        q = _apply_org_filter(q, InsightArticle.org_id, org_id)
    if subject:
        q = q.where(InsightArticle.subject == subject)
    q = q.order_by(InsightArticle.created_at.desc()).limit(limit)
    result = await db.execute(q)
    articles = result.scalars().all()
    if not articles:
        # Generate new feed
        ai = AIService()
        feed_data = await ai.generate_insight_feed(
            user_id=str(current_user.id),
            subject=subject,
            db=db,
        )
        for item in feed_data:
            article = InsightArticle(
                user_id=current_user.id,
                org_id=parsed_oid,
                title=item.get("title"),
                summary=item.get("summary"),
                content=item.get("content"),
                subject=item.get("subject"),
                tags=item.get("tags"),
                reading_time_minutes=item.get("reading_time_minutes"),
            )
            db.add(article)
            articles.append(article)
        await db.commit()
    return articles


@router.get("/news/common")
async def get_common_news(
    current_user: CurrentUser,
    db: DBSession,
    category: str = Query("all"),
    language: str = Query("en"),
    max_results: int = Query(10, le=30),
):
    """Fetch real-time Google News articles by category via RapidAPI."""
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="news_feed", db=db)

    svc = NewsService()
    return await svc.get_common_news(category=category, language=language, max_results=max_results)


@router.get("/news/personal")
async def get_personal_news(
    current_user: CurrentUser,
    db: DBSession,
    language: str = Query("en"),
    max_results: int = Query(10, le=30),
):
    """Fetch real-time news personalised to the user's learning context via RapidAPI."""
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="news_feed", db=db)

    from app.models.ai import AiContextSession, AiInteractionHistory
    from app.models.assessment import TopicMastery

    context: dict = {
        "subjects": [], "topics": [], "weak_topics": [],
        "grade": None, "board": None, "recent_queries": [],
    }

    # AI context session — subject, grade, board from most recent session
    ctx = (await db.execute(
        select(AiContextSession)
        .where(AiContextSession.user_id == current_user.id)
        .order_by(AiContextSession.updated_at.desc())
        .limit(1)
    )).scalar_one_or_none()
    if ctx:
        if ctx.subject: context["subjects"].append(ctx.subject)
        if ctx.grade:   context["grade"] = ctx.grade
        if ctx.board:   context["board"] = ctx.board

    # User profile fallbacks for grade and subjects
    if not context["grade"] and getattr(current_user, "grade", None):
        context["grade"] = current_user.grade
    if getattr(current_user, "subjects", None):
        context["subjects"].extend(current_user.subjects[:2])

    # Topics the user is actively mastering (most recent)
    for m in (await db.execute(
        select(TopicMastery)
        .where(TopicMastery.user_id == current_user.id)
        .order_by(TopicMastery.updated_at.desc())
        .limit(3)
    )).scalars().all():
        if m.topic:
            context["topics"].append(m.topic)

    # Weak topics — mastery_level < 50 (areas that need improvement)
    for m in (await db.execute(
        select(TopicMastery)
        .where(
            TopicMastery.user_id == current_user.id,
            TopicMastery.mastery_level < 50,
        )
        .order_by(TopicMastery.mastery_level.asc())
        .limit(3)
    )).scalars().all():
        if m.topic:
            context["weak_topics"].append(m.topic)

    # Recent queries across ALL AI services (assistant, ebook, ask-doc, etc.)
    for h in (await db.execute(
        select(AiInteractionHistory)
        .where(AiInteractionHistory.user_id == current_user.id)
        .order_by(AiInteractionHistory.created_at.desc())
        .limit(5)
    )).scalars().all():
        if getattr(h, "query", None):
            context["recent_queries"].append(h.query[:50])

    svc = NewsService()
    return await svc.get_personal_news(user_context=context, language=language, max_results=max_results)


@router.post("/intelligence", response_model=IntelligenceResponse)
async def get_learning_intelligence(
    payload: IntelligenceRequest,
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Get aggregated learning intelligence (dashboard snapshots, Bloom profiles, recommendations)."""
    from datetime import timedelta
    workspace_tag = org_id or "personal"
    cache_key = f"intelligence:{current_user.id}:{workspace_tag}:{','.join(sorted(payload.modules or []))}"

    # Try reading from cache — skip gracefully if the table doesn't exist yet
    if not payload.force_refresh:
        try:
            result = await db.execute(
                select(IntelligenceCache).where(
                    IntelligenceCache.user_id == current_user.id,
                    IntelligenceCache.cache_key == cache_key,
                    IntelligenceCache.expires_at > datetime.now(timezone.utc),
                )
            )
            cached = result.scalar_one_or_none()
            if cached:
                return IntelligenceResponse(**cached.payload, cached=True)
        except Exception:
            pass

    ai = AIService()
    intelligence = await ai.get_learning_intelligence(
        user_id=str(current_user.id),
        modules=payload.modules,
        db=db,
    )

    # Try caching the result — skip gracefully if the table doesn't exist yet
    try:
        expires = datetime.now(timezone.utc) + timedelta(minutes=15)
        cache = IntelligenceCache(
            user_id=current_user.id,
            cache_key=cache_key,
            payload=intelligence,
            expires_at=expires,
        )
        db.add(cache)
        await db.commit()
    except Exception:
        await db.rollback()

    return IntelligenceResponse(**intelligence, cached=False)


@router.get("/assessment-summary")
async def get_assessment_summary(
    current_user: CurrentUser,
    db: DBSession,
    force_refresh: bool = Query(False),
    org_id: str | None = Query(None),
):
    """
    Returns an AI-generated coach-style summary of the user's Assessment Hub usage:
    narrative summary, momentum, strengths, weak areas, and personalised goals.
    Cached for 30 minutes per user+workspace via IntelligenceCache.
    """
    from datetime import timedelta

    workspace_tag = org_id or "personal"
    cache_key = f"assessment-summary:{current_user.id}:{workspace_tag}"

    if not force_refresh:
        try:
            result = await db.execute(
                select(IntelligenceCache).where(
                    IntelligenceCache.user_id == current_user.id,
                    IntelligenceCache.cache_key == cache_key,
                    IntelligenceCache.expires_at > datetime.now(timezone.utc),
                )
            )
            cached = result.scalar_one_or_none()
            if cached:
                return {**cached.payload, "cached": True}
        except Exception:
            pass

    ai = AIService()
    summary = await ai.generate_assessment_summary(user_id=str(current_user.id), db=db)

    # Cache for 30 minutes
    try:
        # Delete old cache entry for this key if any
        old = await db.execute(
            select(IntelligenceCache).where(
                IntelligenceCache.user_id == current_user.id,
                IntelligenceCache.cache_key == cache_key,
            )
        )
        old_row = old.scalar_one_or_none()
        if old_row:
            await db.delete(old_row)

        expires = datetime.now(timezone.utc) + timedelta(minutes=30)
        cache = IntelligenceCache(
            user_id=current_user.id,
            cache_key=cache_key,
            payload=summary,
            expires_at=expires,
        )
        db.add(cache)
        await db.commit()
    except Exception:
        await db.rollback()

    return {**summary, "cached": False}


@router.get("/recommendations", response_model=list[RecommendationResponse])
async def get_recommendations(
    current_user: CurrentUser,
    db: DBSession,
    rec_type: str | None = Query(None),
    org_id: str | None = Query(None),
):
    parsed_oid = _parse_org_id(org_id)

    q = select(Recommendation).where(
        Recommendation.user_id == current_user.id,
        Recommendation.is_acted_on == False,
    )
    if rec_type:
        q = q.where(Recommendation.rec_type == rec_type)
    # Filter by workspace: recommendations store org_id in metadata_json
    if org_id is not None:
        from sqlalchemy import text, cast, String
        if parsed_oid:
            q = q.where(
                cast(Recommendation.metadata_json["org_id"], String) == f'"{parsed_oid}"'
            )
        else:
            # Personal workspace: org_id is absent or null in metadata
            q = q.where(
                ~Recommendation.metadata_json.has_key("org_id")
            )
    q = q.order_by(Recommendation.created_at.desc()).limit(20)
    result = await db.execute(q)
    recs = result.scalars().all()

    if not recs:
        # Auto-generate on first load — same logic as /recommendations/generate
        ai = AIService()
        recs_data = await ai.generate_assessment_recommendations(
            user_id=str(current_user.id), db=db
        )
        new_recs = []
        for item in recs_data:
            metadata = {
                "subject": item.get("subject"),
                "topic": item.get("topic"),
                "priority_score": item.get("priority_score", 50),
                "href": "/u/assessments",
            }
            if parsed_oid:
                metadata["org_id"] = str(parsed_oid)
            rec = Recommendation(
                user_id=current_user.id,
                rec_type=item.get("type", "topic"),
                title=item.get("title", ""),
                description=item.get("description"),
                reason=item.get("reason"),
                metadata_json=metadata,
            )
            db.add(rec)
            new_recs.append(rec)
        if new_recs:
            await db.commit()
            for rec in new_recs:
                await db.refresh(rec)
        return new_recs

    return recs


@router.post("/recommendations/generate", response_model=list[RecommendationResponse])
async def generate_recommendations(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Re-generate assessment-based recommendations and replace old ones."""
    from sqlalchemy import delete as sql_delete

    parsed_oid = _parse_org_id(org_id)

    # Deduct 1 point for regenerating recommendations
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="view_recommendations", db=db)

    # Remove old pending recommendations for this workspace
    del_q = sql_delete(Recommendation).where(
        Recommendation.user_id == current_user.id,
        Recommendation.is_acted_on == False,
    )
    if org_id is not None:
        if parsed_oid:
            from sqlalchemy import cast, String
            del_q = del_q.where(
                cast(Recommendation.metadata_json["org_id"], String) == f'"{parsed_oid}"'
            )
        else:
            del_q = del_q.where(~Recommendation.metadata_json.has_key("org_id"))
    await db.execute(del_q)
    await db.commit()

    ai = AIService()
    recs_data = await ai.generate_assessment_recommendations(
        user_id=str(current_user.id), db=db
    )

    new_recs = []
    for item in recs_data:
        metadata = {
            "subject": item.get("subject"),
            "topic": item.get("topic"),
            "priority_score": item.get("priority_score", 50),
            "href": "/u/assessments",
        }
        if parsed_oid:
            metadata["org_id"] = str(parsed_oid)
        rec = Recommendation(
            user_id=current_user.id,
            rec_type=item.get("type", "topic"),
            title=item.get("title", ""),
            description=item.get("description"),
            reason=item.get("reason"),
            metadata_json=metadata,
        )
        db.add(rec)
        new_recs.append(rec)

    if new_recs:
        await db.commit()
        for rec in new_recs:
            await db.refresh(rec)
    return new_recs


@router.get("/learning-curve", response_model=LearningCurveResponse)
async def get_learning_curve(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Get learning progress trends and history for the user."""
    from app.models.assessment import AssessmentAttempt, TopicMastery

    parsed_oid = _parse_org_id(org_id)

    # Assessment scores (filtered by workspace)
    attempts_q = (
        select(AssessmentAttempt)
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
    )
    if org_id is not None:
        attempts_q = attempts_q.where(
            PracticeAssessment.org_id == parsed_oid if parsed_oid else PracticeAssessment.org_id.is_(None)
        )
    attempts_q = attempts_q.order_by(AssessmentAttempt.submitted_at.asc()).limit(50)
    attempts_result = await db.execute(attempts_q)
    attempts = attempts_result.scalars().all()

    assessment_scores = [
        {
            "date": a.submitted_at.isoformat() if a.submitted_at else None,
            "score": a.score,
            "max_score": a.max_score,
            "percentage": a.percentage,
        }
        for a in attempts
    ]

    # Topic mastery (filtered by workspace)
    mastery_q = (
        select(TopicMastery)
        .where(TopicMastery.user_id == current_user.id)
    )
    if org_id is not None:
        mastery_q = mastery_q.where(
            TopicMastery.org_id == parsed_oid if parsed_oid else TopicMastery.org_id.is_(None)
        )
    mastery_q = mastery_q.order_by(TopicMastery.mastery_level.desc()).limit(20)
    mastery_result = await db.execute(mastery_q)
    mastery_data = mastery_result.scalars().all()

    # Use workspace-specific XP/streak
    if parsed_oid is None:
        ws_q = select(WorkspaceGamification).where(
            WorkspaceGamification.user_id == current_user.id,
            WorkspaceGamification.org_id.is_(None),
        )
    else:
        ws_q = select(WorkspaceGamification).where(
            WorkspaceGamification.user_id == current_user.id,
            WorkspaceGamification.org_id == parsed_oid,
        )
    ws_result = await db.execute(ws_q)
    ws_gam = ws_result.scalar_one_or_none()
    ws_xp = ws_gam.xp if ws_gam else 0
    ws_streak = ws_gam.streak if ws_gam else 0

    return LearningCurveResponse(
        user_id=str(current_user.id),
        assessment_scores=assessment_scores,
        topic_mastery_trend=[
            {"topic": m.topic, "subject": m.subject, "mastery_level": m.mastery_level, "trend": m.trend}
            for m in mastery_data
        ],
        xp_trend=[{"xp": ws_xp, "date": datetime.now(timezone.utc).isoformat()}],
        streak_history=[ws_streak],
        weekly_activity={"current_streak": ws_streak, "total_xp": ws_xp},
    )


@router.post("/class-recommendations", response_model=list[ClassRecommendationItem])
async def generate_class_recommendations(
    payload: ClassRecommendationRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Generate AI-powered teaching recommendations for a teacher based on
    their class grading data (criterion averages, weak areas, student performance).
    """
    from app.models.classes import Class
    import uuid as _uuid

    # Enrich class context from DB if not provided in payload
    class_name = payload.class_name
    board = payload.board
    grade = payload.grade
    subject = payload.subject

    if not all([class_name, board, grade, subject]):
        try:
            result = await db.execute(
                select(Class).where(Class.id == _uuid.UUID(payload.class_id))
            )
            cls = result.scalar_one_or_none()
            if cls:
                class_name = class_name or cls.name
                board = board or cls.board
                grade = grade or cls.grade
                subject = subject or cls.subject
        except Exception:
            pass

    ai = AIService()
    recommendations = await ai.generate_class_recommendations({
        "class_id": payload.class_id,
        "class_name": class_name,
        "board": board,
        "grade": grade,
        "subject": subject,
        "total_students": payload.total_students,
        "submissions_graded": payload.submissions_graded,
        "class_average": payload.class_average,
        "students_needing_help": payload.students_needing_help,
        "criterion_averages": [c.model_dump() for c in payload.criterion_averages],
        "weak_outcomes": [w.model_dump() for w in payload.weak_outcomes],
    })
    return recommendations
