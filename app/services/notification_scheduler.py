"""
Background scheduler for generating personalized & system notifications.

Two loops:
- Every 4 hours: personalized checks (weak topics, streaks, XP, study, features)
- Every 30 minutes: time-sensitive system checks (plan expiry, points low)
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models.user import User
from app.models.notification import NotificationType
from app.models.subscription import Subscription
from app.models.assessment import TopicMastery, AssessmentAttempt
from app.models.gamification import WorkspaceGamification
from app.models.ai import AiInteractionHistory
from app.services.notification_service import create_notification

logger = logging.getLogger(__name__)

# ── Streak milestones to celebrate ──────────────────────────────────────────
STREAK_MILESTONES = {7, 14, 30, 50, 100, 200, 365}
XP_MILESTONES = {100, 500, 1000, 2500, 5000, 10000}

# Features to nudge users about (service names in AiInteractionHistory)
DISCOVERABLE_FEATURES = {
    "playground": {"label": "Playground", "icon": "gamepad-2", "link": "/u/playground"},
    "generate-ebook": {"label": "eBook Generator", "icon": "book-open", "link": "/u/ebooks"},
    "generate-mindmap": {"label": "Mind Maps", "icon": "brain", "link": "/u/mindmaps"},
    "career_guidance": {"label": "Career Guidance", "icon": "compass", "link": "/u/career"},
    "practice_assessment": {"label": "Practice Assessments", "icon": "clipboard-check", "link": "/u/assessments"},
}


# ── Personalized Checks (run every 4 hours) ─────────────────────────────────

async def _check_weak_topics(user: User, db: AsyncSession):
    """Alert for topics with mastery < 50 that haven't been attempted in 7+ days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    weak = (await db.execute(
        select(TopicMastery)
        .where(
            TopicMastery.user_id == user.id,
            TopicMastery.mastery_level < 50,
            or_(
                TopicMastery.last_attempted_at < cutoff,
                TopicMastery.last_attempted_at == None,  # noqa: E711
            ),
        )
        .order_by(TopicMastery.mastery_level.asc())
        .limit(3)
    )).scalars().all()

    for tm in weak:
        await create_notification(
            db,
            user_id=user.id,
            notification_type=NotificationType.WEAK_TOPIC_ALERT,
            category="personalized",
            title=f"Review: {tm.topic}",
            body=f"Your mastery in {tm.topic} ({tm.subject}) is at {tm.mastery_level}%. A quick assessment could help strengthen it!",
            icon="alert-triangle",
            data_json={"subject": tm.subject, "topic": tm.topic, "mastery": tm.mastery_level, "link": "/u/assessments"},
            dedup_hours=48,
        )


async def _check_streak_milestones(user: User, db: AsyncSession):
    """Celebrate streak milestones (7, 14, 30, 50, 100...)."""
    records = (await db.execute(
        select(WorkspaceGamification).where(WorkspaceGamification.user_id == user.id)
    )).scalars().all()

    for wg in records:
        if wg.streak in STREAK_MILESTONES:
            await create_notification(
                db,
                user_id=user.id,
                notification_type=NotificationType.STREAK_MILESTONE,
                category="personalized",
                title=f"{wg.streak}-Day Streak!",
                body=f"Incredible! You've been learning for {wg.streak} days in a row. Keep the momentum going!",
                icon="flame",
                org_id=wg.org_id,
                data_json={"streak": wg.streak},
                priority="high" if wg.streak >= 30 else "normal",
                dedup_hours=72,
            )


async def _check_streak_at_risk(user: User, db: AsyncSession):
    """Warn if user has a streak but hasn't been active today."""
    today = datetime.now(timezone.utc).date()
    records = (await db.execute(
        select(WorkspaceGamification).where(
            WorkspaceGamification.user_id == user.id,
            WorkspaceGamification.streak > 0,
        )
    )).scalars().all()

    for wg in records:
        if wg.last_activity_date and wg.last_activity_date.date() < today:
            await create_notification(
                db,
                user_id=user.id,
                notification_type=NotificationType.STREAK_AT_RISK,
                category="personalized",
                title="Your streak is at risk!",
                body=f"You have a {wg.streak}-day streak. Complete any activity today to keep it alive!",
                icon="flame",
                org_id=wg.org_id,
                data_json={"streak": wg.streak, "link": "/u/playground"},
                priority="high",
                dedup_hours=12,
            )


async def _check_xp_milestones(user: User, db: AsyncSession):
    """Celebrate XP milestones (100, 500, 1000, 5000...)."""
    records = (await db.execute(
        select(WorkspaceGamification).where(WorkspaceGamification.user_id == user.id)
    )).scalars().all()

    for wg in records:
        for milestone in sorted(XP_MILESTONES):
            if wg.xp >= milestone:
                # Only notify for the highest reached milestone
                pass
            else:
                break
        else:
            milestone = max(XP_MILESTONES)

        # Find the highest milestone the user has crossed
        reached = [m for m in XP_MILESTONES if wg.xp >= m]
        if reached:
            top = max(reached)
            await create_notification(
                db,
                user_id=user.id,
                notification_type=NotificationType.XP_MILESTONE,
                category="personalized",
                title=f"{top} XP Reached!",
                body=f"You've earned {wg.xp} XP. Amazing progress — keep learning!",
                icon="star",
                org_id=wg.org_id,
                data_json={"xp": wg.xp, "milestone": top},
                dedup_hours=168,  # once per week per milestone
            )


async def _check_study_continuation(user: User, db: AsyncSession):
    """Remind user to continue studying a topic they were recently working on."""
    cutoff_recent = datetime.now(timezone.utc) - timedelta(days=2)
    cutoff_old = datetime.now(timezone.utc) - timedelta(days=14)

    last_interaction = (await db.execute(
        select(AiInteractionHistory)
        .where(
            AiInteractionHistory.user_id == user.id,
            AiInteractionHistory.created_at < cutoff_recent,
            AiInteractionHistory.created_at > cutoff_old,
        )
        .order_by(AiInteractionHistory.created_at.desc())
        .limit(1)
    )).scalar_one_or_none()

    if not last_interaction:
        return

    # Extract topic from context snapshot
    ctx = last_interaction.context_snapshot or {}
    subject = ctx.get("subject") or "your studies"
    query = last_interaction.query or ""
    topic_hint = query[:60] + "…" if len(query) > 60 else query

    await create_notification(
        db,
        user_id=user.id,
        notification_type=NotificationType.STUDY_CONTINUATION,
        category="personalized",
        title=f"Continue with {subject}",
        body=f"You were studying {subject} recently. Pick up where you left off!{(' Topic: ' + topic_hint) if topic_hint else ''}",
        icon="book-open",
        data_json={"subject": subject, "link": "/u/ask"},
        dedup_hours=48,
    )


async def _check_feature_discovery(user: User, db: AsyncSession):
    """Nudge users to try features they haven't used yet (after 7+ days on platform)."""
    if not user.created_at:
        return
    if (datetime.now(timezone.utc) - user.created_at.replace(tzinfo=timezone.utc)).days < 7:
        return

    # Find which services the user has used
    used_services = set((await db.execute(
        select(AiInteractionHistory.service).where(
            AiInteractionHistory.user_id == user.id,
        ).distinct()
    )).scalars().all())

    for service_key, meta in DISCOVERABLE_FEATURES.items():
        if service_key not in used_services:
            await create_notification(
                db,
                user_id=user.id,
                notification_type=NotificationType.FEATURE_DISCOVERY,
                category="personalized",
                title=f"Have you tried {meta['label']}?",
                body=f"Explore {meta['label']} — it can help you learn in a whole new way!",
                icon=meta["icon"],
                data_json={"feature": service_key, "link": meta["link"]},
                dedup_hours=168,  # once per week
            )
            break  # Only one feature nudge at a time


async def _check_score_improvement(user: User, db: AsyncSession):
    """Congratulate when latest assessment score improved over previous."""
    attempts = (await db.execute(
        select(AssessmentAttempt)
        .where(
            AssessmentAttempt.user_id == user.id,
            AssessmentAttempt.status == "evaluated",
            AssessmentAttempt.percentage != None,  # noqa: E711
        )
        .order_by(AssessmentAttempt.submitted_at.desc())
        .limit(2)
    )).scalars().all()

    if len(attempts) < 2:
        return
    latest, previous = attempts[0], attempts[1]
    if latest.percentage and previous.percentage and latest.percentage > previous.percentage:
        improvement = round(latest.percentage - previous.percentage, 1)
        await create_notification(
            db,
            user_id=user.id,
            notification_type=NotificationType.SCORE_IMPROVEMENT,
            category="personalized",
            title="Score Improved!",
            body=f"Your latest assessment score improved by {improvement}%! You scored {round(latest.percentage, 1)}%. Great work!",
            icon="trending-up",
            data_json={"score": latest.percentage, "improvement": improvement, "link": "/u/learning-curve"},
            dedup_hours=24,
        )


# ── Study Time Checks ──────────────────────────────────────────────────────

STUDY_TIME_MILESTONES = [30, 60, 120, 180]
STUDY_CONSISTENCY_MILESTONES = {5, 10, 20, 30, 50, 100}


async def _check_study_time_milestone(user: User, db: AsyncSession):
    """Celebrate when today's study time crosses 30/60/120/180 minutes."""
    from app.models.study_time import StudyTimeDaily
    today = datetime.now(timezone.utc).date()

    result = await db.execute(
        select(func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0)).where(
            StudyTimeDaily.user_id == user.id,
            StudyTimeDaily.date == today,
        )
    )
    total_today = result.scalar_one()
    if total_today <= 0:
        return

    # Find highest crossed milestone
    reached = [m for m in STUDY_TIME_MILESTONES if total_today >= m]
    if not reached:
        return

    top = max(reached)
    hours = top // 60
    mins = top % 60
    label = f"{hours}h {mins}m" if hours else f"{mins} min"

    await create_notification(
        db,
        user_id=user.id,
        notification_type=NotificationType.STUDY_TIME_MILESTONE,
        category="personalized",
        title=f"You studied {label} today!",
        body=f"Great focus! You've logged {total_today} minutes of study time today. Keep it up!",
        icon="clock",
        data_json={"minutes_today": total_today, "milestone": top, "link": "/u/learning-curve"},
        dedup_hours=24,
    )


async def _check_study_time_inactive(user: User, db: AsyncSession):
    """Alert students who haven't studied for 3+ consecutive days."""
    from app.models.study_time import StudyTimeDaily
    today = datetime.now(timezone.utc).date()
    three_days_ago = today - timedelta(days=3)

    # Check if user has any study time in the last 3 days
    recent = await db.execute(
        select(func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0)).where(
            StudyTimeDaily.user_id == user.id,
            StudyTimeDaily.date >= three_days_ago,
            StudyTimeDaily.date <= today,
        )
    )
    recent_minutes = recent.scalar_one()
    if recent_minutes > 0:
        return

    # Only alert if user was previously active (at least 5 days of recorded study time)
    history_count = (await db.execute(
        select(func.count(func.distinct(StudyTimeDaily.date))).where(
            StudyTimeDaily.user_id == user.id,
        )
    )).scalar_one()
    if history_count < 5:
        return

    await create_notification(
        db,
        user_id=user.id,
        notification_type=NotificationType.STUDY_TIME_INACTIVE,
        category="personalized",
        title="We miss you!",
        body="You haven't studied in 3 days. Even 15 minutes can make a difference — jump back in!",
        icon="book-open",
        data_json={"days_inactive": 3, "link": "/u/ask"},
        priority="high",
        dedup_hours=72,
    )


async def _check_study_time_weekly_summary(user: User, db: AsyncSession):
    """Send a weekly study time summary on Sundays."""
    from app.models.study_time import StudyTimeDaily
    today = datetime.now(timezone.utc).date()

    # Only run on Sundays (weekday 6)
    if today.weekday() != 6:
        return

    week_start = today - timedelta(days=7)
    prev_week_start = today - timedelta(days=14)

    # This week
    this_week = (await db.execute(
        select(func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0)).where(
            StudyTimeDaily.user_id == user.id,
            StudyTimeDaily.date >= week_start,
            StudyTimeDaily.date < today,
        )
    )).scalar_one()

    if this_week <= 0:
        return

    # Previous week
    prev_week = (await db.execute(
        select(func.coalesce(func.sum(StudyTimeDaily.total_minutes), 0)).where(
            StudyTimeDaily.user_id == user.id,
            StudyTimeDaily.date >= prev_week_start,
            StudyTimeDaily.date < week_start,
        )
    )).scalar_one()

    # Top subject this week
    top_subject_row = (await db.execute(
        select(StudyTimeDaily.subject, func.sum(StudyTimeDaily.total_minutes).label("mins"))
        .where(
            StudyTimeDaily.user_id == user.id,
            StudyTimeDaily.date >= week_start,
            StudyTimeDaily.date < today,
        )
        .group_by(StudyTimeDaily.subject)
        .order_by(func.sum(StudyTimeDaily.total_minutes).desc())
        .limit(1)
    )).first()
    top_subject = top_subject_row[0] if top_subject_row else "General"

    hours = this_week // 60
    mins = this_week % 60
    time_str = f"{hours}h {mins}m" if hours else f"{mins}m"

    trend = ""
    if prev_week > 0:
        change = round(((this_week - prev_week) / prev_week) * 100)
        trend = f" {'📈' if change >= 0 else '📉'} {abs(change)}% vs last week."

    await create_notification(
        db,
        user_id=user.id,
        notification_type=NotificationType.STUDY_TIME_WEEKLY_SUMMARY,
        category="personalized",
        title=f"Weekly Study Recap: {time_str}",
        body=f"You studied {time_str} this week. Top subject: {top_subject}.{trend}",
        icon="bar-chart-2",
        data_json={
            "this_week_minutes": this_week,
            "prev_week_minutes": prev_week,
            "top_subject": top_subject,
            "link": "/u/learning-curve",
        },
        dedup_hours=168,
    )


async def _check_study_time_consistency(user: User, db: AsyncSession):
    """Celebrate consecutive days of studying (5, 10, 20, 30, 50, 100)."""
    from app.models.study_time import StudyTimeDaily
    today = datetime.now(timezone.utc).date()

    # Get distinct study dates in descending order (minimum 15 min to count)
    study_dates = (await db.execute(
        select(StudyTimeDaily.date, func.sum(StudyTimeDaily.total_minutes).label("mins"))
        .where(StudyTimeDaily.user_id == user.id)
        .group_by(StudyTimeDaily.date)
        .having(func.sum(StudyTimeDaily.total_minutes) >= 15)
        .order_by(StudyTimeDaily.date.desc())
        .limit(110)  # enough to detect up to 100-day streak
    )).all()

    if not study_dates:
        return

    # Count consecutive days ending at the most recent study day
    consecutive = 0
    expected_date = study_dates[0][0]
    for row_date, _ in study_dates:
        if row_date == expected_date:
            consecutive += 1
            expected_date = row_date - timedelta(days=1)
        else:
            break

    # Find the highest milestone reached
    reached = [m for m in STUDY_CONSISTENCY_MILESTONES if consecutive >= m]
    if not reached:
        return

    top = max(reached)
    await create_notification(
        db,
        user_id=user.id,
        notification_type=NotificationType.STUDY_TIME_CONSISTENCY,
        category="personalized",
        title=f"{top}-Day Study Streak!",
        body=f"You've studied for {consecutive} consecutive days. Consistency is the key to mastery!",
        icon="flame",
        data_json={"consecutive_days": consecutive, "milestone": top, "link": "/u/learning-curve"},
        priority="high" if top >= 30 else "normal",
        dedup_hours=72,
    )


# ── System Checks (run every 30 minutes) ────────────────────────────────────

async def _check_plan_expiry(db: AsyncSession):
    """Warn users whose plan expires in 3 days or 1 day."""
    now = datetime.now(timezone.utc)
    for days_ahead, urgency in [(3, "normal"), (1, "high")]:
        window_start = now + timedelta(days=days_ahead - 0.5)
        window_end = now + timedelta(days=days_ahead + 0.5)
        subs = (await db.execute(
            select(Subscription).where(
                Subscription.status == "active",
                Subscription.current_period_end >= window_start,
                Subscription.current_period_end < window_end,
                Subscription.plan != "free",
            )
        )).scalars().all()

        for sub in subs:
            await create_notification(
                db,
                user_id=sub.user_id,
                notification_type=NotificationType.PLAN_EXPIRY_APPROACHING,
                category="system",
                title=f"Plan expires in {days_ahead} day{'s' if days_ahead > 1 else ''}",
                body=f"Your {sub.plan.replace('_', ' ').title()} plan expires soon. Renew to keep all your features!",
                icon="credit-card",
                org_id=sub.org_id,
                data_json={"plan": sub.plan, "expires_at": sub.current_period_end.isoformat(), "link": "/u/plans"},
                priority=urgency,
                dedup_hours=20,
            )


async def _check_points_low(db: AsyncSession):
    """Notify when points balance is low or exhausted."""
    # Points exhausted
    exhausted = (await db.execute(
        select(Subscription).where(
            Subscription.status == "active",
            Subscription.points_balance <= 0,
        )
    )).scalars().all()

    for sub in exhausted:
        await create_notification(
            db,
            user_id=sub.user_id,
            notification_type=NotificationType.POINTS_EXHAUSTED,
            category="system",
            title="No AI points remaining",
            body="You've used all your AI points. Upgrade your plan or wait for the monthly reset to continue using AI features.",
            icon="alert-circle",
            org_id=sub.org_id,
            data_json={"balance": sub.points_balance, "link": "/u/plans"},
            priority="high",
            dedup_hours=24,
        )

    # Points low (< 10 but > 0)
    low = (await db.execute(
        select(Subscription).where(
            Subscription.status == "active",
            Subscription.points_balance > 0,
            Subscription.points_balance < 10,
        )
    )).scalars().all()

    for sub in low:
        await create_notification(
            db,
            user_id=sub.user_id,
            notification_type=NotificationType.POINTS_LOW,
            category="system",
            title=f"Only {sub.points_balance} AI points left",
            body=f"You have {sub.points_balance} AI points remaining. Consider upgrading to get more!",
            icon="alert-circle",
            org_id=sub.org_id,
            data_json={"balance": sub.points_balance, "link": "/u/plans"},
            priority="normal",
            dedup_hours=24,
        )


async def _cleanup_expired(db: AsyncSession):
    """Delete notifications past their expires_at or older than 30 days."""
    from app.models.notification import Notification
    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)

    await db.execute(
        select(Notification).where(
            or_(
                and_(Notification.expires_at != None, Notification.expires_at < now),  # noqa: E711
                Notification.created_at < thirty_days_ago,
            )
        ).with_for_update()
    )
    # Actually delete them
    from sqlalchemy import delete
    await db.execute(
        delete(Notification).where(
            or_(
                and_(Notification.expires_at != None, Notification.expires_at < now),  # noqa: E711
                Notification.created_at < thirty_days_ago,
            )
        )
    )


# ── Main Scheduler Loops ────────────────────────────────────────────────────

async def _run_personalized_checks():
    """Run personalized notification checks for all active users."""
    async with AsyncSessionLocal() as db:
        try:
            # Aggregate yesterday's study time before running checks
            from app.services.study_time_aggregator import aggregate_all_users_yesterday
            await aggregate_all_users_yesterday(db)
            await db.commit()
        except Exception as e:
            await db.rollback()
            print(f"[NotificationScheduler] Study time aggregation error: {e}", flush=True)

    async with AsyncSessionLocal() as db:
        try:
            users = (await db.execute(
                select(User).where(User.is_active == True)  # noqa: E712
            )).scalars().all()

            for user in users:
                try:
                    await _check_weak_topics(user, db)
                    await _check_streak_milestones(user, db)
                    await _check_streak_at_risk(user, db)
                    await _check_xp_milestones(user, db)
                    await _check_study_continuation(user, db)
                    await _check_feature_discovery(user, db)
                    await _check_score_improvement(user, db)
                    await _check_study_time_milestone(user, db)
                    await _check_study_time_inactive(user, db)
                    await _check_study_time_weekly_summary(user, db)
                    await _check_study_time_consistency(user, db)
                except Exception as e:
                    logger.warning("Notification check failed for user %s: %s", user.id, e)

            await db.commit()
            print(f"[NotificationScheduler] Personalized checks done for {len(users)} users", flush=True)
        except Exception as e:
            await db.rollback()
            print(f"[NotificationScheduler] Personalized checks error: {e}", flush=True)


async def _run_system_checks():
    """Run time-sensitive system notification checks."""
    async with AsyncSessionLocal() as db:
        try:
            await _check_plan_expiry(db)
            await _check_points_low(db)
            await _cleanup_expired(db)
            await db.commit()
            print("[NotificationScheduler] System checks done", flush=True)
        except Exception as e:
            await db.rollback()
            print(f"[NotificationScheduler] System checks error: {e}", flush=True)


async def run_notification_scheduler():
    """Background loop with two cadences: 30min (system) and 4h (personalized)."""
    print("[NotificationScheduler] Started background notification scheduler", flush=True)

    SYSTEM_INTERVAL = 30 * 60        # 30 minutes
    PERSONALIZED_INTERVAL = 4 * 3600  # 4 hours
    last_personalized = 0.0

    while True:
        try:
            await _run_system_checks()

            import time
            now = time.time()
            if now - last_personalized >= PERSONALIZED_INTERVAL:
                await _run_personalized_checks()
                last_personalized = now

        except Exception as e:
            print(f"[NotificationScheduler] Error: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

        await asyncio.sleep(SYSTEM_INTERVAL)
