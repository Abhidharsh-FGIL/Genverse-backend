import uuid
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, status, Query
from sqlalchemy import select, func

from app.dependencies import DBSession, CurrentUser
from app.models.gamification import Badge, StudentBadge, Title, StudentTitle, WorkspaceGamification
from app.models.user import User
from app.schemas.gamification import (
    BadgeResponse,
    StudentBadgeResponse,
    TitleResponse,
    StudentTitleResponse,
    LeaderboardEntry,
    GamificationSummary,
)
from app.core.exceptions import NotFoundException, ConflictException

router = APIRouter()


def _parse_org_id(org_id: str | None) -> uuid.UUID | None:
    if not org_id or org_id == "personal":
        return None
    try:
        return uuid.UUID(org_id)
    except ValueError:
        return None


async def _get_or_create_ws_gamification(
    user_id: uuid.UUID, org_id_param: str | None, db
) -> WorkspaceGamification:
    """Get or create workspace gamification row for user + workspace."""
    parsed_oid = _parse_org_id(org_id_param)
    if parsed_oid is None:
        q = select(WorkspaceGamification).where(
            WorkspaceGamification.user_id == user_id,
            WorkspaceGamification.org_id.is_(None),
        )
    else:
        q = select(WorkspaceGamification).where(
            WorkspaceGamification.user_id == user_id,
            WorkspaceGamification.org_id == parsed_oid,
        )
    result = await db.execute(q)
    row = result.scalars().first()
    if not row:
        row = WorkspaceGamification(user_id=user_id, org_id=parsed_oid, xp=0, streak=0)
        db.add(row)
        await db.flush()
    return row


@router.post("/xp")
async def award_xp(
    payload: dict,
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Award XP for non-assessment actions (uploads, mindmaps, etc.)."""
    amount = int(payload.get("amount") or 0)
    if amount > 0:
        # Update global XP (backward compatible)
        current_user.xp = (current_user.xp or 0) + amount

        # Update workspace-specific XP
        ws_gam = await _get_or_create_ws_gamification(current_user.id, org_id, db)
        ws_gam.xp = (ws_gam.xp or 0) + amount

        await db.commit()

    # Return workspace-specific XP
    ws_gam = await _get_or_create_ws_gamification(current_user.id, org_id, db)
    return {"xp": ws_gam.xp or 0}


@router.post("/activity")
async def record_activity(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Record activity ping, update streak, and award daily login XP (idempotent per day per workspace)."""
    DAILY_LOGIN_XP = 3
    # Use IST (Asia/Kolkata) for day-boundary comparison so streaks behave for Indian users.
    # Storing UTC timestamps is fine; only the .date() comparison must be local.
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(timezone.utc)
    today = now.astimezone(IST).date()

    # Update workspace-specific streak + daily XP
    ws_gam = await _get_or_create_ws_gamification(current_user.id, org_id, db)
    already_today = False

    if ws_gam.last_activity_date:
        last_dt = ws_gam.last_activity_date
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        ws_last = last_dt.astimezone(IST).date()
        if ws_last == today:
            # Already recorded today — no XP, no streak change
            already_today = True
        elif ws_last == today - timedelta(days=1):
            ws_gam.streak = (ws_gam.streak or 0) + 1
        else:
            ws_gam.streak = 1
    else:
        ws_gam.streak = 1

    if not already_today:
        ws_gam.last_activity_date = now
        # Award daily login XP (only once per day per workspace)
        ws_gam.xp = (ws_gam.xp or 0) + DAILY_LOGIN_XP
        # Also update global XP and streak (backward compatible)
        current_user.xp = (current_user.xp or 0) + DAILY_LOGIN_XP
        if current_user.last_login_date:
            gl_dt = current_user.last_login_date
            if gl_dt.tzinfo is None:
                gl_dt = gl_dt.replace(tzinfo=timezone.utc)
            gl = gl_dt.astimezone(IST).date()
            if gl != today:
                if gl == today - timedelta(days=1):
                    current_user.streak = (current_user.streak or 0) + 1
                else:
                    current_user.streak = 1
        else:
            current_user.streak = 1
        current_user.last_login_date = now

    await db.commit()

    # Check for newly earned badges after streak/XP update
    try:
        from app.services.badge_service import check_and_award_badges
        parsed_oid = _parse_org_id(org_id)
        new_badges = await check_and_award_badges(current_user.id, db, parsed_oid)
    except Exception:
        new_badges = []

    return {"streak": ws_gam.streak or 0, "xp": ws_gam.xp or 0, "new_badges": new_badges}


@router.get("/badges", response_model=list[BadgeResponse])
async def list_all_badges(db: DBSession, current_user: CurrentUser):
    result = await db.execute(select(Badge).order_by(Badge.rarity))
    return result.scalars().all()


@router.get("/my/badges", response_model=list[StudentBadgeResponse])
async def get_my_badges(current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(StudentBadge)
        .where(StudentBadge.student_id == current_user.id)
        .order_by(StudentBadge.earned_at.desc())
    )
    student_badges = result.scalars().all()
    responses = []
    for sb in student_badges:
        badge_result = await db.execute(select(Badge).where(Badge.id == sb.badge_id))
        badge = badge_result.scalar_one_or_none()
        if badge:
            responses.append(
                StudentBadgeResponse(
                    id=sb.id,
                    badge_id=sb.badge_id,
                    earned_at=sb.earned_at,
                    badge=BadgeResponse.model_validate(badge),
                )
            )
    return responses


@router.get("/titles", response_model=list[TitleResponse])
async def list_all_titles(db: DBSession, current_user: CurrentUser):
    result = await db.execute(select(Title))
    return result.scalars().all()


@router.get("/my/titles", response_model=list[StudentTitleResponse])
async def get_my_titles(current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(StudentTitle).where(StudentTitle.student_id == current_user.id)
    )
    student_titles = result.scalars().all()
    responses = []
    for st in student_titles:
        title_result = await db.execute(select(Title).where(Title.id == st.title_id))
        title = title_result.scalar_one_or_none()
        if title:
            responses.append(
                StudentTitleResponse(
                    id=st.id,
                    title_id=st.title_id,
                    is_active=st.is_active,
                    earned_at=st.earned_at,
                    title=TitleResponse.model_validate(title),
                )
            )
    return responses


@router.patch("/my/titles/{title_id}/activate")
async def activate_title(title_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    # Deactivate all current titles
    result = await db.execute(
        select(StudentTitle).where(
            StudentTitle.student_id == current_user.id,
            StudentTitle.is_active == True,
        )
    )
    current_active = result.scalars().all()
    for st in current_active:
        st.is_active = False

    # Activate the selected title
    result = await db.execute(
        select(StudentTitle).where(
            StudentTitle.student_id == current_user.id,
            StudentTitle.title_id == title_id,
        )
    )
    title = result.scalar_one_or_none()
    if not title:
        raise NotFoundException("Title not owned")
    title.is_active = True
    await db.commit()
    return {"message": "Title activated"}


@router.post("/check-badges")
async def check_badges_now(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Manually trigger badge check. Returns list of newly awarded badges."""
    from app.services.badge_service import check_and_award_badges
    parsed_oid = _parse_org_id(org_id)
    new_badges = await check_and_award_badges(current_user.id, db, parsed_oid)
    return {"new_badges": new_badges}


@router.get("/my/summary", response_model=GamificationSummary)
async def get_gamification_summary(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    badge_count_result = await db.execute(
        select(func.count(StudentBadge.id)).where(StudentBadge.student_id == current_user.id)
    )
    badge_count = badge_count_result.scalar_one()

    title_count_result = await db.execute(
        select(func.count(StudentTitle.id)).where(StudentTitle.student_id == current_user.id)
    )
    title_count = title_count_result.scalar_one()

    # Use workspace-specific XP/streak
    ws_gam = await _get_or_create_ws_gamification(current_user.id, org_id, db)
    await db.commit()  # persist if newly created

    xp = ws_gam.xp or 0
    level = xp // 100 + 1
    next_level_xp = level * 100

    return GamificationSummary(
        xp=xp,
        streak=ws_gam.streak or 0,
        level=level,
        next_level_xp=next_level_xp,
        badges_earned=badge_count,
        titles_earned=title_count,
    )


@router.post("/seed-badges")
async def seed_default_badges(current_user: CurrentUser, db: DBSession):
    """Seed default badges into the database if none exist."""
    count_result = await db.execute(select(func.count(Badge.id)))
    count = count_result.scalar_one()
    if count > 0:
        return {"message": f"{count} badges already exist", "seeded": 0}

    DEFAULT_BADGES = [
        {"name": "First Steps", "description": "Join your first class", "icon": "footprints", "category": "special", "rarity": "common", "xp_reward": 10, "criteria": {"type": "classes_joined", "threshold": 1}},
        {"name": "Class Explorer", "description": "Join 3 classes", "icon": "book-open", "category": "special", "rarity": "rare", "xp_reward": 30, "criteria": {"type": "classes_joined", "threshold": 3}},
        {"name": "Bookworm", "description": "Submit 5 assignments", "icon": "graduation-cap", "category": "assignment", "rarity": "common", "xp_reward": 20, "criteria": {"type": "submissions", "threshold": 5}},
        {"name": "Scholar", "description": "Submit 10 assignments", "icon": "brain", "category": "assignment", "rarity": "rare", "xp_reward": 50, "criteria": {"type": "submissions", "threshold": 10}},
        {"name": "Dedicated Learner", "description": "Submit 25 assignments", "icon": "trophy", "category": "assignment", "rarity": "epic", "xp_reward": 100, "criteria": {"type": "submissions", "threshold": 25}},
        {"name": "Streak Starter", "description": "Maintain a 3-day streak", "icon": "flame", "category": "streak", "rarity": "common", "xp_reward": 15, "criteria": {"type": "streak", "threshold": 3}},
        {"name": "Streak Master", "description": "Maintain a 7-day streak", "icon": "zap", "category": "streak", "rarity": "rare", "xp_reward": 40, "criteria": {"type": "streak", "threshold": 7}},
        {"name": "Streak Legend", "description": "Maintain a 30-day streak", "icon": "crown", "category": "streak", "rarity": "legendary", "xp_reward": 200, "criteria": {"type": "streak", "threshold": 30}},
        {"name": "XP Hunter", "description": "Earn 100 XP", "icon": "star", "category": "mastery", "rarity": "common", "xp_reward": 10, "criteria": {"type": "xp", "threshold": 100}},
        {"name": "XP Master", "description": "Earn 500 XP", "icon": "award", "category": "mastery", "rarity": "rare", "xp_reward": 50, "criteria": {"type": "xp", "threshold": 500}},
        {"name": "XP Legend", "description": "Earn 1000 XP", "icon": "medal", "category": "mastery", "rarity": "epic", "xp_reward": 100, "criteria": {"type": "xp", "threshold": 1000}},
        {"name": "Perfectionist", "description": "Score 100% on an assignment", "icon": "trophy", "category": "assignment", "rarity": "rare", "xp_reward": 50, "criteria": {"type": "perfect_score", "threshold": 1}},
        {"name": "High Achiever", "description": "Score 90%+ on 5 assignments", "icon": "trending-up", "category": "assignment", "rarity": "epic", "xp_reward": 100, "criteria": {"type": "high_scores", "threshold": 5}},
        {"name": "Quiz Warrior", "description": "Complete 5 quizzes", "icon": "brain", "category": "quiz", "rarity": "rare", "xp_reward": 40, "criteria": {"type": "quizzes_completed", "threshold": 5}},
        {"name": "Quiz Champion", "description": "Complete 20 quizzes", "icon": "crown", "category": "quiz", "rarity": "epic", "xp_reward": 100, "criteria": {"type": "quizzes_completed", "threshold": 20}},
    ]

    for badge_data in DEFAULT_BADGES:
        db.add(Badge(**badge_data))
    await db.commit()
    return {"message": "Default badges seeded", "seeded": len(DEFAULT_BADGES)}
