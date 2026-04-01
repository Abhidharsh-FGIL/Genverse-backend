import uuid
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, status, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.insights import CareerGuidanceSession
from app.models.ai import IntelligenceCache
from app.schemas.ai import CareerGuidanceRequest, CareerGuidanceResponse
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()


def _parse_org_id(org_id: str | None) -> uuid.UUID | None:
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


@router.get("/profile")
async def get_career_profile(
    current_user: CurrentUser,
    db: DBSession,
    force_refresh: bool = Query(False),
    org_id: str | None = Query(None),
):
    """
    Returns an AI-generated career profile built from the user's assessment data,
    topic mastery, AI chat history, and past career sessions.
    No user input required — the profile grows automatically as the user uses the platform.
    Cached for 60 minutes per user.
    """
    workspace_tag = org_id or "personal"
    cache_key = f"career-profile:{current_user.id}:{workspace_tag}"

    # Only deduct 1 point on explicit refresh, not on normal page view
    if force_refresh:
        points_service = PointsService()
        await points_service.deduct(user_id=current_user.id, action="view_career_profile", db=db)

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
    profile = await ai.generate_career_profile(user_id=str(current_user.id), db=db, org_id=org_id)

    # Cache for 60 minutes
    try:
        old = await db.execute(
            select(IntelligenceCache).where(
                IntelligenceCache.user_id == current_user.id,
                IntelligenceCache.cache_key == cache_key,
            )
        )
        old_row = old.scalar_one_or_none()
        if old_row:
            await db.delete(old_row)

        expires = datetime.now(timezone.utc) + timedelta(minutes=60)
        cache = IntelligenceCache(
            user_id=current_user.id,
            cache_key=cache_key,
            payload=profile,
            expires_at=expires,
        )
        db.add(cache)
        await db.commit()
    except Exception:
        await db.rollback()

    return {**profile, "cached": False}


@router.post("/analyze", response_model=CareerGuidanceResponse, status_code=status.HTTP_201_CREATED)
async def analyze_career(
    payload: CareerGuidanceRequest,
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Run career compatibility analysis using AI. Cost: 2 pts."""
    parsed_oid = _parse_org_id(org_id)

    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="career_guidance", db=db, org_id=parsed_oid)
    await points_service.deduct(user_id=current_user.id, action="career_guidance", db=db)

    ai = AIService()
    analysis = await ai.analyze_career(
        interests=payload.interests,
        strengths=payload.strengths,
        target_careers=payload.target_careers,
        grade=payload.grade,
        context=payload.context,
    )

    session = CareerGuidanceSession(
        user_id=current_user.id,
        org_id=parsed_oid,
        interests=payload.interests,
        strengths=payload.strengths,
        target_careers=payload.target_careers,
        analysis_json=analysis,
        compatibility_scores=analysis.get("compatibility_scores"),
        points_used=2,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Invalidate cached profile so next load reflects new session data
    workspace_tag = org_id or "personal"
    try:
        old = await db.execute(
            select(IntelligenceCache).where(
                IntelligenceCache.user_id == current_user.id,
                IntelligenceCache.cache_key == f"career-profile:{current_user.id}:{workspace_tag}",
            )
        )
        old_row = old.scalar_one_or_none()
        if old_row:
            await db.delete(old_row)
            await db.commit()
    except Exception:
        await db.rollback()

    return session


@router.get("/sessions", response_model=list[CareerGuidanceResponse])
async def list_career_sessions(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(10, le=50),
    org_id: str | None = Query(None),
):
    q = (
        select(CareerGuidanceSession)
        .where(CareerGuidanceSession.user_id == current_user.id)
    )
    if org_id is not None:
        q = _apply_org_filter(q, CareerGuidanceSession.org_id, org_id)
    q = q.order_by(CareerGuidanceSession.created_at.desc()).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/sessions/{session_id}", response_model=CareerGuidanceResponse)
async def get_career_session(session_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(CareerGuidanceSession).where(
            CareerGuidanceSession.id == session_id,
            CareerGuidanceSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise NotFoundException("Career guidance session not found")
    return session
