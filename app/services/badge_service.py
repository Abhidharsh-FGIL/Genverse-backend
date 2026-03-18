"""Badge auto-award service.

Call `check_and_award_badges(user_id, db, org_id)` after XP/streak changes,
submission grading, class enrollment, or quiz completion to automatically
award any newly earned badges.
"""
import uuid
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.gamification import Badge, StudentBadge, WorkspaceGamification
from app.models.classes import ClassStudent, Submission, Assignment


async def check_and_award_badges(
    user_id: uuid.UUID,
    db: AsyncSession,
    org_id: uuid.UUID | None = None,
) -> list[dict]:
    """Check all badge criteria and award any the user qualifies for but hasn't earned yet.
    Returns a list of newly awarded badge dicts (for toast notifications)."""

    # Fetch all badges
    result = await db.execute(select(Badge))
    all_badges = result.scalars().all()

    # Fetch already earned badge IDs
    earned_result = await db.execute(
        select(StudentBadge.badge_id).where(StudentBadge.student_id == user_id)
    )
    earned_ids = {row[0] for row in earned_result.all()}

    # Gather user stats (lazy — only compute what's needed)
    stats_cache: dict = {}

    async def get_stat(stat_type: str) -> int:
        if stat_type in stats_cache:
            return stats_cache[stat_type]

        value = 0
        if stat_type == "streak":
            if org_id:
                ws_result = await db.execute(
                    select(WorkspaceGamification.streak).where(
                        WorkspaceGamification.user_id == user_id,
                        WorkspaceGamification.org_id == org_id,
                    )
                )
            else:
                ws_result = await db.execute(
                    select(WorkspaceGamification.streak).where(
                        WorkspaceGamification.user_id == user_id,
                        WorkspaceGamification.org_id.is_(None),
                    )
                )
            value = ws_result.scalar_one_or_none() or 0

        elif stat_type == "xp":
            if org_id:
                ws_result = await db.execute(
                    select(WorkspaceGamification.xp).where(
                        WorkspaceGamification.user_id == user_id,
                        WorkspaceGamification.org_id == org_id,
                    )
                )
            else:
                ws_result = await db.execute(
                    select(WorkspaceGamification.xp).where(
                        WorkspaceGamification.user_id == user_id,
                        WorkspaceGamification.org_id.is_(None),
                    )
                )
            value = ws_result.scalar_one_or_none() or 0

        elif stat_type == "submissions":
            count_result = await db.execute(
                select(func.count(Submission.id)).where(Submission.student_id == user_id)
            )
            value = count_result.scalar_one() or 0

        elif stat_type == "perfect_score":
            # Count submissions with 100% score
            count_result = await db.execute(
                select(func.count(Submission.id)).where(
                    Submission.student_id == user_id,
                    Submission.status == "graded",
                )
            )
            # Need to check grade JSON — do it in Python for flexibility
            all_graded = await db.execute(
                select(Submission).where(
                    Submission.student_id == user_id,
                    Submission.status == "graded",
                )
            )
            perfect_count = 0
            for sub in all_graded.scalars().all():
                if sub.grade and isinstance(sub.grade, dict):
                    total = sub.grade.get("totalScore", 0)
                    max_s = sub.grade.get("maxScore", 1)
                    if max_s > 0 and total >= max_s:
                        perfect_count += 1
            value = perfect_count

        elif stat_type == "high_scores":
            all_graded = await db.execute(
                select(Submission).where(
                    Submission.student_id == user_id,
                    Submission.status == "graded",
                )
            )
            high_count = 0
            for sub in all_graded.scalars().all():
                if sub.grade and isinstance(sub.grade, dict):
                    total = sub.grade.get("totalScore", 0)
                    max_s = sub.grade.get("maxScore", 1)
                    if max_s > 0 and (total / max_s) >= 0.9:
                        high_count += 1
            value = high_count

        elif stat_type == "classes_joined":
            count_result = await db.execute(
                select(func.count(ClassStudent.id)).where(ClassStudent.student_id == user_id)
            )
            value = count_result.scalar_one() or 0

        elif stat_type == "quizzes_completed":
            # Count practice assessment attempts
            try:
                from app.models.classes import QuizAttempt
                count_result = await db.execute(
                    select(func.count(QuizAttempt.id)).where(
                        QuizAttempt.student_id == user_id,
                        QuizAttempt.submitted_at.isnot(None),
                    )
                )
                value = count_result.scalar_one() or 0
            except Exception:
                value = 0

        stats_cache[stat_type] = value
        return value

    # Check each badge
    newly_awarded = []
    for badge in all_badges:
        if badge.id in earned_ids:
            continue
        criteria = badge.criteria
        if not criteria or not isinstance(criteria, dict):
            continue

        crit_type = criteria.get("type")
        threshold = criteria.get("threshold", 0)
        if not crit_type or not threshold:
            continue

        current_value = await get_stat(crit_type)
        if current_value >= threshold:
            # Award the badge!
            db.add(StudentBadge(student_id=user_id, badge_id=badge.id))
            earned_ids.add(badge.id)
            newly_awarded.append({
                "id": str(badge.id),
                "name": badge.name,
                "description": badge.description,
                "icon": badge.icon,
                "rarity": badge.rarity,
                "xp_reward": badge.xp_reward,
            })
            # Award bonus XP for the badge
            if badge.xp_reward > 0:
                if org_id:
                    ws_result = await db.execute(
                        select(WorkspaceGamification).where(
                            WorkspaceGamification.user_id == user_id,
                            WorkspaceGamification.org_id == org_id,
                        )
                    )
                else:
                    ws_result = await db.execute(
                        select(WorkspaceGamification).where(
                            WorkspaceGamification.user_id == user_id,
                            WorkspaceGamification.org_id.is_(None),
                        )
                    )
                ws_gam = ws_result.scalars().first()
                if ws_gam:
                    ws_gam.xp = (ws_gam.xp or 0) + badge.xp_reward

    if newly_awarded:
        await db.commit()

    return newly_awarded
