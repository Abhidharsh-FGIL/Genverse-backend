import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Query
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import DBSession, CurrentUser, OptionalCurrentUser
from app.models.subscription import (
    Subscription,
    PlanDefinition,
    PointCost,
    PointTransaction,
    SubscriptionAddon,
    FeatureLimit,
    UsageCounter,
)
from app.schemas.subscription import (
    SubscriptionResponse,
    PlanDefinitionResponse,
    UpgradePlanRequest,
    PointDeductRequest,
    PointDeductResponse,
    PointTransactionResponse,
    BuyAddonRequest,
    FeatureLimitResponse,
    UsageCounterResponse,
    AddonResponse,
)
from app.core.exceptions import (
    NotFoundException,
    InsufficientPointsException,
    SubscriptionInactiveException,
    NoSubscriptionException,
)

router = APIRouter()

ADDON_POINT_MAP = {
    "point_pack_100": 100,
    "point_pack_500": 500,
    "point_pack_1000": 1000,
}


async def _get_active_subscription(
    db: AsyncSession,
    user_id: uuid.UUID,
    org_id: uuid.UUID | None = None,
) -> Subscription | None:
    if org_id:
        result = await db.execute(
            select(Subscription).where(
                Subscription.org_id == org_id,
                Subscription.status.in_(["active", "trialing"]),
            )
        )
    else:
        result = await db.execute(
            select(Subscription).where(
                Subscription.user_id == user_id,
                Subscription.workspace_type == "individual",
                Subscription.status.in_(["active", "trialing"]),
            )
        )
    return result.scalar_one_or_none()


@router.get("", response_model=SubscriptionResponse)
@router.get("/me", response_model=SubscriptionResponse)
async def get_my_subscription(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    sub = await _get_active_subscription(
        db, current_user.id, uuid.UUID(org_id) if org_id else None
    )
    if not sub:
        raise NoSubscriptionException()
    return sub


@router.get("/plans", response_model=list[PlanDefinitionResponse])
async def list_plans(db: DBSession, workspace_type: str | None = Query(None)):
    q = select(PlanDefinition).where(PlanDefinition.is_active == True)
    if workspace_type:
        q = q.where(PlanDefinition.workspace_type == workspace_type)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/plans/{plan_name}", response_model=PlanDefinitionResponse)
async def get_plan(plan_name: str, db: DBSession):
    result = await db.execute(
        select(PlanDefinition).where(PlanDefinition.plan == plan_name)
    )
    plan = result.scalar_one_or_none()
    if not plan:
        raise NotFoundException(f"Plan '{plan_name}' not found")
    return plan


@router.get("/feature-limits", response_model=list[FeatureLimitResponse])
@router.get("/features", response_model=list[FeatureLimitResponse])
async def get_feature_limits_by_plan(
    db: DBSession,
    current_user: OptionalCurrentUser = None,
    org_id: str | None = Query(None),
    plan: str | None = Query(None),
):
    plan_name = plan or "free"
    if current_user:
        sub = await _get_active_subscription(
            db, current_user.id, uuid.UUID(org_id) if org_id else None
        )
        if sub:
            plan_name = sub.plan
    result = await db.execute(
        select(FeatureLimit).where(FeatureLimit.plan == plan_name)
    )
    return result.scalars().all()


@router.get("/usage", response_model=list[UsageCounterResponse])
async def get_usage_counters(current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(UsageCounter).where(UsageCounter.user_id == current_user.id)
    )
    return result.scalars().all()


@router.get("/addons", response_model=list[AddonResponse])
async def get_addons(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    sub = await _get_active_subscription(
        db, current_user.id, uuid.UUID(org_id) if org_id else None
    )
    if not sub:
        return []
    result = await db.execute(
        select(SubscriptionAddon).where(SubscriptionAddon.subscription_id == sub.id)
    )
    return result.scalars().all()


@router.post("/upgrade")
async def upgrade_plan(
    payload: UpgradePlanRequest,
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    sub = await _get_active_subscription(
        db, current_user.id, uuid.UUID(org_id) if org_id else None
    )
    if not sub:
        raise NoSubscriptionException()

    plan_result = await db.execute(
        select(PlanDefinition).where(PlanDefinition.plan == payload.plan)
    )
    plan = plan_result.scalar_one_or_none()
    if not plan:
        raise NotFoundException("Plan not found")

    sub.plan = payload.plan
    sub.status = "active"
    sub.points_monthly_quota = plan.monthly_points
    sub.points_balance = plan.monthly_points
    sub.storage_limit_mb = plan.storage_mb
    sub.max_seats = plan.max_seats
    await db.commit()
    return {"message": "Plan upgraded", "plan": payload.plan}


@router.post("/deduct-points", response_model=PointDeductResponse)
async def deduct_points(
    payload: PointDeductRequest,
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    cost_result = await db.execute(
        select(PointCost).where(PointCost.action == payload.action)
    )
    point_cost = cost_result.scalar_one_or_none()
    if not point_cost:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown action")

    sub = await _get_active_subscription(
        db, current_user.id, uuid.UUID(org_id) if org_id else None
    )
    if not sub:
        raise NoSubscriptionException()
    if sub.status not in ("active", "trialing"):
        raise SubscriptionInactiveException()
    if sub.points_balance < point_cost.cost:
        refresh_date = sub.current_period_end.isoformat() if sub.current_period_end else ""
        raise InsufficientPointsException(
            points_needed=point_cost.cost,
            points_available=sub.points_balance,
            refresh_date=refresh_date,
        )

    # Atomic deduction
    sub.points_balance -= point_cost.cost
    transaction = PointTransaction(
        subscription_id=sub.id,
        user_id=current_user.id,
        action=payload.action,
        points_used=point_cost.cost,
        balance_after=sub.points_balance,
    )
    db.add(transaction)
    await db.commit()

    return PointDeductResponse(
        success=True,
        points_used=point_cost.cost,
        remaining_balance=sub.points_balance,
        action=payload.action,
    )


@router.get("/transactions", response_model=list[PointTransactionResponse])
async def list_transactions(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = Query(50, le=200),
):
    result = await db.execute(
        select(PointTransaction)
        .where(PointTransaction.user_id == current_user.id)
        .order_by(PointTransaction.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.post("/addons", response_model=SubscriptionResponse)
async def buy_addon(
    payload: BuyAddonRequest,
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    sub = await _get_active_subscription(
        db, current_user.id, uuid.UUID(org_id) if org_id else None
    )
    if not sub:
        raise NoSubscriptionException()

    points_to_add = ADDON_POINT_MAP.get(payload.addon_type)
    if not points_to_add:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown addon type")

    total_points = points_to_add * payload.quantity
    addon = SubscriptionAddon(
        subscription_id=sub.id,
        addon_type=payload.addon_type,
        quantity=payload.quantity,
        points_added=total_points,
    )
    db.add(addon)
    sub.points_balance += total_points
    await db.commit()
    await db.refresh(sub)
    return sub


