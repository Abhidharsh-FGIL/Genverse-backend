import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, status, Query
from sqlalchemy import select, func

from app.dependencies import DBSession, CurrentUser
from app.models.insights import UserInsight, InsightArticle, Recommendation
from app.models.ai import IntelligenceCache
from app.schemas.insights import (
    UserInsightResponse,
    InsightArticleResponse,
    GenerateInsightsRequest,
    IntelligenceRequest,
    IntelligenceResponse,
    RecommendationResponse,
    LearningCurveResponse,
)
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()


@router.post("/generate", response_model=list[UserInsightResponse])
async def generate_insights(payload: GenerateInsightsRequest, current_user: CurrentUser, db: DBSession):
    """Generate personalized learning insights using AI."""
    if not payload.force_refresh:
        # Check for recent insights
        result = await db.execute(
            select(UserInsight)
            .where(UserInsight.user_id == current_user.id)
            .order_by(UserInsight.created_at.desc())
            .limit(5)
        )
        existing = result.scalars().all()
        if existing:
            return existing

    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="generate_insights", db=db)

    ai = AIService()
    insights_data = await ai.generate_insights(user_id=str(current_user.id), db=db)

    new_insights = []
    for item in insights_data:
        insight = UserInsight(
            user_id=current_user.id,
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
):
    from app.models.assessment import AssessmentAttempt

    q = select(UserInsight).where(UserInsight.user_id == current_user.id)
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
):
    q = select(InsightArticle).where(InsightArticle.user_id == current_user.id)
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


@router.post("/intelligence", response_model=IntelligenceResponse)
async def get_learning_intelligence(
    payload: IntelligenceRequest, current_user: CurrentUser, db: DBSession
):
    """Get aggregated learning intelligence (dashboard snapshots, Bloom profiles, recommendations)."""
    from datetime import timedelta
    cache_key = f"intelligence:{current_user.id}:{','.join(sorted(payload.modules or []))}"

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
):
    """
    Returns an AI-generated coach-style summary of the user's Assessment Hub usage:
    narrative summary, momentum, strengths, weak areas, and personalised goals.
    Cached for 30 minutes per user via IntelligenceCache.
    """
    from datetime import timedelta

    cache_key = f"assessment-summary:{current_user.id}"

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
):
    q = select(Recommendation).where(
        Recommendation.user_id == current_user.id,
        Recommendation.is_acted_on == False,
    )
    if rec_type:
        q = q.where(Recommendation.rec_type == rec_type)
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
            rec = Recommendation(
                user_id=current_user.id,
                rec_type=item.get("type", "topic"),
                title=item.get("title", ""),
                description=item.get("description"),
                reason=item.get("reason"),
                metadata_json={
                    "subject": item.get("subject"),
                    "topic": item.get("topic"),
                    "priority_score": item.get("priority_score", 50),
                    "href": "/u/assessments",
                },
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
async def generate_recommendations(current_user: CurrentUser, db: DBSession):
    """Re-generate assessment-based recommendations and replace old ones."""
    from sqlalchemy import delete as sql_delete

    # Remove old pending recommendations so fresh ones replace them
    await db.execute(
        sql_delete(Recommendation).where(
            Recommendation.user_id == current_user.id,
            Recommendation.is_acted_on == False,
        )
    )
    await db.commit()

    ai = AIService()
    recs_data = await ai.generate_assessment_recommendations(
        user_id=str(current_user.id), db=db
    )

    new_recs = []
    for item in recs_data:
        rec = Recommendation(
            user_id=current_user.id,
            rec_type=item.get("type", "topic"),
            title=item.get("title", ""),
            description=item.get("description"),
            reason=item.get("reason"),
            metadata_json={
                "subject": item.get("subject"),
                "topic": item.get("topic"),
                "priority_score": item.get("priority_score", 50),
                "href": "/u/assessments",
            },
        )
        db.add(rec)
        new_recs.append(rec)

    if new_recs:
        await db.commit()
        for rec in new_recs:
            await db.refresh(rec)
    return new_recs


@router.get("/learning-curve", response_model=LearningCurveResponse)
async def get_learning_curve(current_user: CurrentUser, db: DBSession):
    """Get learning progress trends and history for the user."""
    from app.models.assessment import AssessmentAttempt, TopicMastery
    # Assessment scores
    attempts_result = await db.execute(
        select(AssessmentAttempt)
        .where(
            AssessmentAttempt.user_id == current_user.id,
            AssessmentAttempt.status == "evaluated",
        )
        .order_by(AssessmentAttempt.submitted_at.asc())
        .limit(50)
    )
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

    # Topic mastery
    mastery_result = await db.execute(
        select(TopicMastery)
        .where(TopicMastery.user_id == current_user.id)
        .order_by(TopicMastery.mastery_level.desc())
        .limit(20)
    )
    mastery_data = mastery_result.scalars().all()

    return LearningCurveResponse(
        user_id=str(current_user.id),
        assessment_scores=assessment_scores,
        topic_mastery_trend=[
            {"topic": m.topic, "subject": m.subject, "mastery_level": m.mastery_level, "trend": m.trend}
            for m in mastery_data
        ],
        xp_trend=[{"xp": current_user.xp, "date": datetime.now(timezone.utc).isoformat()}],
        streak_history=[current_user.streak],
        weekly_activity={"current_streak": current_user.streak, "total_xp": current_user.xp},
    )
