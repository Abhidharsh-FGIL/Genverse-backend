"""
platform_stats_service.py
Computes and caches platform-wide usage averages (score, study time, streak,
topics mastered) so Personal Insights news cards can annotate a user's own
numbers against a "platform average" value.

No Celery Beat exists in this deployment (all tasks are on-demand only), so
refresh is lazy: the cached row is reused until it goes stale, and whichever
request notices the staleness recomputes and upserts it. This avoids requiring
a new always-on worker process.
"""
import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.insights import PlatformStats
from app.models.assessment import AssessmentAttempt, TopicMastery
from app.models.study_time import StudyTimeDaily
from app.models.gamification import WorkspaceGamification

STALE_AFTER = timedelta(hours=6)
LOOKBACK_DAYS = 30
MASTERY_THRESHOLD = 70.0  # mirrors the existing "weak topic" cutoff (< 50) on the high side


async def get_platform_stats(db: AsyncSession) -> dict:
    """Return cached platform-wide averages, recomputing if stale or missing."""
    row = (await db.execute(
        select(PlatformStats).order_by(PlatformStats.computed_at.desc()).limit(1)
    )).scalar_one_or_none()

    if row:
        computed_at = row.computed_at
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - computed_at < STALE_AFTER:
            return row.stats_json

    stats = await _compute_platform_stats(db)
    db.add(PlatformStats(id=uuid.uuid4(), stats_json=stats))
    await db.commit()
    return stats


async def _compute_platform_stats(db: AsyncSession) -> dict:
    since_dt = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    since_date = since_dt.date()

    avg_score = (await db.execute(
        select(func.avg(AssessmentAttempt.percentage)).where(
            AssessmentAttempt.status == "evaluated",
            AssessmentAttempt.submitted_at >= since_dt,
        )
    )).scalar()

    total_minutes, active_users = (await db.execute(
        select(
            func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0),
            func.count(func.distinct(StudyTimeDaily.user_id)),
        ).where(StudyTimeDaily.date >= since_date)
    )).first()
    avg_daily_minutes = (
        round(total_minutes / active_users / LOOKBACK_DAYS, 1) if active_users else None
    )

    avg_streak = (await db.execute(
        select(func.avg(WorkspaceGamification.streak)).where(
            WorkspaceGamification.org_id.is_(None)
        )
    )).scalar()

    mastered_count, learners = (await db.execute(
        select(
            func.count(TopicMastery.id).filter(TopicMastery.mastery_level >= MASTERY_THRESHOLD),
            func.count(func.distinct(TopicMastery.user_id)),
        )
    )).first()
    avg_topics_mastered = round(mastered_count / learners, 1) if learners else None

    return {
        "avg_score_percent": round(avg_score, 1) if avg_score is not None else None,
        "avg_daily_study_minutes": avg_daily_minutes,
        "avg_streak": round(float(avg_streak), 1) if avg_streak is not None else None,
        "avg_topics_mastered": avg_topics_mastered,
    }


async def get_user_usage_metrics(db: AsyncSession, user_id: uuid.UUID) -> dict:
    """Same four metrics as get_platform_stats, computed for a single user."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date()

    avg_score = (await db.execute(
        select(func.avg(AssessmentAttempt.percentage)).where(
            AssessmentAttempt.user_id == user_id,
            AssessmentAttempt.status == "evaluated",
        )
    )).scalar()

    total_minutes = (await db.execute(
        select(func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0))
        .where(StudyTimeDaily.user_id == user_id, StudyTimeDaily.date >= since_date)
    )).scalar()

    # .first() (not scalar_one_or_none()) — some users have more than one
    # personal-workspace gamification row in the data; take the most recently
    # active one rather than crashing on MultipleResultsFound.
    ws_gam = (await db.execute(
        select(WorkspaceGamification)
        .where(
            WorkspaceGamification.user_id == user_id,
            WorkspaceGamification.org_id.is_(None),
        )
        .order_by(WorkspaceGamification.updated_at.desc())
        .limit(1)
    )).scalars().first()

    mastered_count = (await db.execute(
        select(func.count(TopicMastery.id)).where(
            TopicMastery.user_id == user_id,
            TopicMastery.mastery_level >= MASTERY_THRESHOLD,
        )
    )).scalar()

    return {
        "score_percent": round(avg_score, 1) if avg_score is not None else None,
        "daily_study_minutes": round((total_minutes or 0) / LOOKBACK_DAYS, 1),
        "streak": ws_gam.streak if ws_gam else 0,
        "topics_mastered": mastered_count or 0,
    }


_METRIC_ROTATION = ["score", "study_time", "streak", "topics_mastered"]


def _build_annotation(metric: str, user_stats: dict, platform_stats: dict) -> dict | None:
    if metric == "score":
        u, p = user_stats.get("score_percent"), platform_stats.get("avg_score_percent")
        if u is None or p is None:
            return None
        return {"metric": metric, "label": "Avg. score", "your_value": f"{u:.0f}%", "platform_value": f"{p:.0f}%", "higher_is_better": True}
    if metric == "study_time":
        u, p = user_stats.get("daily_study_minutes"), platform_stats.get("avg_daily_study_minutes")
        if u is None or p is None:
            return None
        return {"metric": metric, "label": "Daily study time", "your_value": f"{u:.0f}m", "platform_value": f"{p:.0f}m", "higher_is_better": True}
    if metric == "streak":
        u, p = user_stats.get("streak"), platform_stats.get("avg_streak")
        if u is None or p is None:
            return None
        return {"metric": metric, "label": "Streak", "your_value": f"{u} days", "platform_value": f"{p:.0f} days", "higher_is_better": True}
    if metric == "topics_mastered":
        u, p = user_stats.get("topics_mastered"), platform_stats.get("avg_topics_mastered")
        if u is None or p is None:
            return None
        return {"metric": metric, "label": "Topics mastered", "your_value": str(u), "platform_value": f"{p:.1f}", "higher_is_better": True}
    return None


def annotate_personal_articles(articles: list[dict], user_stats: dict, platform_stats: dict) -> list[dict]:
    """Attach a rotating usage_annotation (your value vs platform average) to each article."""
    for i, article in enumerate(articles):
        metric = _METRIC_ROTATION[i % len(_METRIC_ROTATION)]
        article["usage_annotation"] = _build_annotation(metric, user_stats, platform_stats)
    return articles
