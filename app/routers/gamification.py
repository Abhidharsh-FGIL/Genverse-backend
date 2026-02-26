import uuid
from fastapi import APIRouter, status, Query
from sqlalchemy import select, func

from app.dependencies import DBSession, CurrentUser
from app.models.gamification import Badge, StudentBadge, Title, StudentTitle
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


@router.post("/xp")
async def award_xp(payload: dict, current_user: CurrentUser, db: DBSession):
    """Award XP for non-assessment actions (uploads, mindmaps, etc.)."""
    amount = int(payload.get("amount") or 0)
    if amount > 0:
        current_user.xp = (current_user.xp or 0) + amount
        await db.commit()
    return {"xp": current_user.xp or 0}


@router.post("/activity")
async def record_activity(current_user: CurrentUser, db: DBSession):
    """Record activity ping and return current streak."""
    return {"streak": current_user.streak or 0}


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


@router.get("/my/summary", response_model=GamificationSummary)
async def get_gamification_summary(current_user: CurrentUser, db: DBSession):
    badge_count_result = await db.execute(
        select(func.count(StudentBadge.id)).where(StudentBadge.student_id == current_user.id)
    )
    badge_count = badge_count_result.scalar_one()

    title_count_result = await db.execute(
        select(func.count(StudentTitle.id)).where(StudentTitle.student_id == current_user.id)
    )
    title_count = title_count_result.scalar_one()

    xp = current_user.xp or 0
    level = xp // 100 + 1
    next_level_xp = level * 100

    return GamificationSummary(
        xp=xp,
        streak=current_user.streak or 0,
        level=level,
        next_level_xp=next_level_xp,
        badges_earned=badge_count,
        titles_earned=title_count,
    )
