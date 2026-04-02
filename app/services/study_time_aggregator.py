"""
Study Time Aggregator — derives daily study time from AiInteractionHistory
and AssessmentAttempt timestamps.

Algorithm:
1. Collect all timestamped events (AI interactions + assessment attempts) for a user/date.
2. Sort by timestamp and cluster events within 30-minute gaps into sessions.
3. Session duration = time between first and last event in cluster + 5 min for the tail.
4. Assessment duration = submitted_at - started_at (capped at time_limit or 120 min).
5. Group by subject and upsert into study_time_daily.
"""

import logging
from datetime import datetime, timezone, timedelta, date as date_type
from collections import defaultdict

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.ai import AiInteractionHistory
from app.models.assessment import AssessmentAttempt, PracticeAssessment
from app.models.study_time import StudyTimeDaily
from app.models.organization import OrgMember

logger = logging.getLogger(__name__)

# Maximum gap between events to consider them part of the same session
SESSION_GAP_MINUTES = 30
# Fixed minutes added for the last event in a session cluster
TAIL_MINUTES = 5
# Maximum assessment duration cap (minutes)
MAX_ASSESSMENT_MINUTES = 120


async def aggregate_user_study_time(
    user_id,
    target_date: date_type,
    db: AsyncSession,
    org_id=None,
):
    """Aggregate study time for a single user on a specific date."""
    day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    # 1. Fetch AI interactions for this user on this date
    ai_rows = (await db.execute(
        select(AiInteractionHistory).where(
            AiInteractionHistory.user_id == user_id,
            AiInteractionHistory.created_at >= day_start,
            AiInteractionHistory.created_at < day_end,
        ).order_by(AiInteractionHistory.created_at)
    )).scalars().all()

    # 2. Fetch assessment attempts that started or were submitted on this date
    attempt_rows = (await db.execute(
        select(AssessmentAttempt, PracticeAssessment.subject, PracticeAssessment.time_limit)
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == user_id,
            AssessmentAttempt.started_at >= day_start,
            AssessmentAttempt.started_at < day_end,
        )
    )).all()

    # 3. Build events list: (timestamp, subject, type, duration_override_minutes)
    events = []
    for row in ai_rows:
        ctx = row.context_snapshot or {}
        subject = ctx.get("subject") or "General"
        events.append((row.created_at, subject, "interaction", None))

    # Track assessment-specific minutes per subject
    assessment_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "minutes": 0})

    for attempt, subject, time_limit in attempt_rows:
        subj = subject or "General"
        assessment_stats[subj]["count"] += 1

        # Calculate assessment duration
        if attempt.submitted_at and attempt.started_at:
            duration = (attempt.submitted_at - attempt.started_at).total_seconds() / 60
            cap = min(time_limit or MAX_ASSESSMENT_MINUTES, MAX_ASSESSMENT_MINUTES)
            duration = min(duration, cap)
            assessment_stats[subj]["minutes"] += int(duration)
            events.append((attempt.started_at, subj, "assessment", duration))
        else:
            events.append((attempt.started_at, subj, "assessment", TAIL_MINUTES))

    if not events:
        return

    # 4. Sort all events by timestamp
    events.sort(key=lambda e: e[0])

    # 5. Cluster events into sessions per subject and calculate time
    subject_minutes: dict[str, int] = defaultdict(int)
    subject_interactions: dict[str, int] = defaultdict(int)
    hour_minutes: dict[int, int] = defaultdict(int)

    # For session clustering, we cluster ALL events regardless of subject
    # then attribute time to the subject of each event
    clusters = []
    current_cluster = [events[0]]

    for i in range(1, len(events)):
        prev_ts = events[i - 1][0]
        curr_ts = events[i][0]
        if prev_ts.tzinfo is None:
            prev_ts = prev_ts.replace(tzinfo=timezone.utc)
        if curr_ts.tzinfo is None:
            curr_ts = curr_ts.replace(tzinfo=timezone.utc)
        gap = (curr_ts - prev_ts).total_seconds() / 60
        if gap <= SESSION_GAP_MINUTES:
            current_cluster.append(events[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [events[i]]
    clusters.append(current_cluster)

    # 6. Calculate minutes per cluster
    for cluster in clusters:
        if len(cluster) == 1:
            ts, subj, etype, dur_override = cluster[0]
            minutes = dur_override if dur_override else TAIL_MINUTES
            subject_minutes[subj] += int(minutes)
            if etype == "interaction":
                subject_interactions[subj] += 1
            ts_tz = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
            hour_minutes[ts_tz.hour] += int(minutes)
        else:
            # Distribute time between consecutive events to the subject of the earlier event
            for j in range(len(cluster) - 1):
                ts1 = cluster[j][0]
                ts2 = cluster[j + 1][0]
                if ts1.tzinfo is None:
                    ts1 = ts1.replace(tzinfo=timezone.utc)
                if ts2.tzinfo is None:
                    ts2 = ts2.replace(tzinfo=timezone.utc)
                gap_min = (ts2 - ts1).total_seconds() / 60
                subj = cluster[j][1]
                subject_minutes[subj] += int(gap_min)
                if cluster[j][2] == "interaction":
                    subject_interactions[subj] += 1
                hour_minutes[ts1.hour] += int(gap_min)

            # Add tail for the last event in the cluster
            last_ts, last_subj, last_type, last_dur = cluster[-1]
            tail = last_dur if last_dur else TAIL_MINUTES
            subject_minutes[last_subj] += int(tail)
            if last_type == "interaction":
                subject_interactions[last_subj] += 1
            last_ts_tz = last_ts.replace(tzinfo=timezone.utc) if last_ts.tzinfo is None else last_ts
            hour_minutes[last_ts_tz.hour] += int(tail)

    # 7. Find peak hour
    peak_hour = max(hour_minutes, key=hour_minutes.get) if hour_minutes else None

    # 8. Upsert into study_time_daily
    for subj, minutes in subject_minutes.items():
        if minutes <= 0:
            continue

        a_stats = assessment_stats.get(subj, {"count": 0, "minutes": 0})

        stmt = pg_insert(StudyTimeDaily).values(
            user_id=user_id,
            org_id=org_id,
            date=target_date,
            subject=subj,
            total_minutes=minutes,
            interaction_count=subject_interactions.get(subj, 0),
            assessment_count=a_stats["count"],
            assessment_minutes=a_stats["minutes"],
            peak_hour=peak_hour,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_study_time_daily",
            set_={
                "total_minutes": stmt.excluded.total_minutes,
                "interaction_count": stmt.excluded.interaction_count,
                "assessment_count": stmt.excluded.assessment_count,
                "assessment_minutes": stmt.excluded.assessment_minutes,
                "peak_hour": stmt.excluded.peak_hour,
                "updated_at": func.now(),
            },
        )
        await db.execute(stmt)


async def aggregate_all_users_yesterday(db: AsyncSession):
    """Aggregate study time for all users who had activity yesterday. Called from notification scheduler."""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    # Find users who had AI interactions yesterday
    user_rows = (await db.execute(
        select(AiInteractionHistory.user_id).where(
            AiInteractionHistory.created_at >= datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc),
            AiInteractionHistory.created_at < datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc) + timedelta(days=1),
        ).distinct()
    )).scalars().all()

    # Also find users who had assessment attempts yesterday
    attempt_users = (await db.execute(
        select(AssessmentAttempt.user_id).where(
            AssessmentAttempt.started_at >= datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc),
            AssessmentAttempt.started_at < datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc) + timedelta(days=1),
        ).distinct()
    )).scalars().all()

    all_users = set(user_rows) | set(attempt_users)

    # Look up org memberships for all active users so we can store org_id
    org_memberships: dict = {}
    if all_users:
        om_rows = (await db.execute(
            select(OrgMember.user_id, OrgMember.org_id).where(
                OrgMember.user_id.in_(list(all_users)),
                OrgMember.status == "active",
            )
        )).all()
        for uid, oid in om_rows:
            org_memberships.setdefault(uid, []).append(oid)

    for uid in all_users:
        try:
            org_ids = org_memberships.get(uid, [None])
            for oid in org_ids:
                await aggregate_user_study_time(uid, yesterday, db, org_id=oid)
        except Exception as e:
            logger.warning("Study time aggregation failed for user %s: %s", uid, e)

    logger.info("[StudyTimeAggregator] Aggregated study time for %d users for %s", len(all_users), yesterday)


async def backfill_study_time(db: AsyncSession, days_back: int = 90):
    """Backfill study time for all users over the last N days. Run once after deployment."""
    today = datetime.now(timezone.utc).date()

    # Find all distinct dates with AI activity in the window
    start = today - timedelta(days=days_back)
    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)

    # Get all user/date combos that have interactions
    user_date_rows = (await db.execute(
        select(
            AiInteractionHistory.user_id,
            func.date(AiInteractionHistory.created_at).label("day"),
        ).where(
            AiInteractionHistory.created_at >= start_dt,
        ).group_by(
            AiInteractionHistory.user_id,
            func.date(AiInteractionHistory.created_at),
        )
    )).all()

    # Also assessment user/date combos
    attempt_user_dates = (await db.execute(
        select(
            AssessmentAttempt.user_id,
            func.date(AssessmentAttempt.started_at).label("day"),
        ).where(
            AssessmentAttempt.started_at >= start_dt,
        ).group_by(
            AssessmentAttempt.user_id,
            func.date(AssessmentAttempt.started_at),
        )
    )).all()

    # Merge into a set of (user_id, date) pairs
    pairs = set()
    for uid, day in user_date_rows:
        if day:
            pairs.add((uid, day))
    for uid, day in attempt_user_dates:
        if day:
            pairs.add((uid, day))

    # Look up org memberships for all users
    all_user_ids = {uid for uid, _ in pairs}
    org_memberships: dict = {}
    if all_user_ids:
        om_rows = (await db.execute(
            select(OrgMember.user_id, OrgMember.org_id).where(
                OrgMember.user_id.in_(list(all_user_ids)),
                OrgMember.status == "active",
            )
        )).all()
        for uid, oid in om_rows:
            org_memberships.setdefault(uid, []).append(oid)

    count = 0
    for uid, day in pairs:
        try:
            org_ids = org_memberships.get(uid, [None])
            for oid in org_ids:
                await aggregate_user_study_time(uid, day, db, org_id=oid)
            count += 1
        except Exception as e:
            logger.warning("Backfill failed for user %s date %s: %s", uid, day, e)

    print(f"[StudyTimeAggregator] Backfill complete: {count} user-days processed over {days_back} days", flush=True)
    return count


async def compute_realtime_study_minutes(user_id, target_date: date_type, db: AsyncSession) -> dict:
    """Compute study time for a given date on the fly (not stored). Used for today's data."""
    day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    ai_rows = (await db.execute(
        select(AiInteractionHistory).where(
            AiInteractionHistory.user_id == user_id,
            AiInteractionHistory.created_at >= day_start,
            AiInteractionHistory.created_at < day_end,
        ).order_by(AiInteractionHistory.created_at)
    )).scalars().all()

    attempt_rows = (await db.execute(
        select(AssessmentAttempt, PracticeAssessment.subject, PracticeAssessment.time_limit)
        .join(PracticeAssessment, AssessmentAttempt.assessment_id == PracticeAssessment.id)
        .where(
            AssessmentAttempt.user_id == user_id,
            AssessmentAttempt.started_at >= day_start,
            AssessmentAttempt.started_at < day_end,
        )
    )).all()

    if not ai_rows and not attempt_rows:
        return {"total_minutes": 0, "subjects": {}}

    # Build events
    events = []
    for row in ai_rows:
        ctx = row.context_snapshot or {}
        subject = ctx.get("subject") or "General"
        events.append((row.created_at, subject))

    for attempt, subject, time_limit in attempt_rows:
        subj = subject or "General"
        if attempt.submitted_at and attempt.started_at:
            duration = (attempt.submitted_at - attempt.started_at).total_seconds() / 60
            cap = min(time_limit or MAX_ASSESSMENT_MINUTES, MAX_ASSESSMENT_MINUTES)
            duration = min(duration, cap)
        else:
            duration = TAIL_MINUTES
        events.append((attempt.started_at, subj))

    events.sort(key=lambda e: e[0])

    # Session clustering
    total = 0
    subjects: dict[str, int] = {}
    for i in range(len(events)):
        if i == 0:
            gap = TAIL_MINUTES
        else:
            ts1 = events[i - 1][0]
            ts2 = events[i][0]
            if ts1.tzinfo is None:
                ts1 = ts1.replace(tzinfo=timezone.utc)
            if ts2.tzinfo is None:
                ts2 = ts2.replace(tzinfo=timezone.utc)
            gap_sec = (ts2 - ts1).total_seconds() / 60
            gap = min(gap_sec, SESSION_GAP_MINUTES) if gap_sec <= SESSION_GAP_MINUTES else TAIL_MINUTES

        subj = events[i][1]
        total += int(gap)
        subjects[subj] = subjects.get(subj, 0) + int(gap)

    return {"total_minutes": total, "subjects": subjects}
