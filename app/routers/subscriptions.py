import uuid
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

import stripe
from app.config import settings
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
    "starter_200": 200,
    "learner_500": 500,
    "pro_1500": 1500,
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
            ).order_by(Subscription.updated_at.desc())
        )
    else:
        result = await db.execute(
            select(Subscription).where(
                Subscription.user_id == user_id,
                Subscription.workspace_type == "individual",
                Subscription.status.in_(["active", "trialing"]),
            ).order_by(Subscription.updated_at.desc())
        )
    return result.scalars().first()


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


@router.get("/point-costs")
async def list_point_costs(
    db: DBSession,
    workspace_type: str | None = Query(None, description="Filter by workspace type: 'personal', 'org', or omit for all"),
):
    q = select(PointCost)
    if workspace_type:
        q = q.where(PointCost.workspace_type.in_([workspace_type, "both"]))
    result = await db.execute(q)
    return result.scalars().all()


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

    old_balance = sub.points_balance
    is_downgrade_to_free = payload.plan == "free"

    sub.plan = payload.plan
    sub.status = "active"
    sub.points_monthly_quota = plan.monthly_points
    sub.storage_limit_mb = plan.storage_mb
    sub.max_seats = plan.max_seats

    if is_downgrade_to_free:
        # Downgrade to free: keep existing balance, don't add 100 again
        sub.points_balance = old_balance
        sub.current_period_start = None
        sub.current_period_end = None
    else:
        # Upgrade: carry over existing points + add new plan's monthly quota
        sub.points_balance = old_balance + plan.monthly_points
        now = datetime.now(timezone.utc)
        sub.current_period_start = now
        sub.current_period_end = now + timedelta(days=30)

        # Record the plan upgrade as a point transaction (credit)
        txn = PointTransaction(
            subscription_id=sub.id,
            user_id=current_user.id,
            action=f"plan_upgrade_{payload.plan}",
            points_used=-plan.monthly_points,  # negative = credit
            balance_after=sub.points_balance,
        )
        db.add(txn)

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
    point_cost = cost_result.scalars().first()
    if not point_cost:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This action is not recognized. Please try again or contact support.")

    # Accept org_id from body or query param (body takes precedence)
    effective_org_id = payload.org_id or org_id
    sub = await _get_active_subscription(
        db, current_user.id, uuid.UUID(effective_org_id) if effective_org_id else None
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This point pack is not available. Please select a valid option.")

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


class ToggleAutoRenewRequest(BaseModel):
    auto_renew: bool


@router.post("/auto-renew")
async def toggle_auto_renew(
    payload: ToggleAutoRenewRequest,
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Enable or disable auto-renewal for the current subscription."""
    sub = await _get_active_subscription(
        db, current_user.id, uuid.UUID(org_id) if org_id else None
    )
    if not sub:
        raise NoSubscriptionException()
    if sub.plan == "free":
        raise HTTPException(status_code=400, detail="Free plan does not support auto-renewal")

    sub.auto_renew = payload.auto_renew

    # If disabling auto-renew, cancel the recurring subscription on the payment gateway
    if not payload.auto_renew:
        if sub.stripe_subscription_id:
            try:
                stripe.api_key = settings.STRIPE_API_KEY
                stripe.Subscription.modify(
                    sub.stripe_subscription_id,
                    cancel_at_period_end=True,
                )
            except Exception as e:
                print(f"[Subscription] Failed to update Stripe subscription: {e}", flush=True)

        if sub.phonepe_subscription_id:
            try:
                from app.routers.payments import _cancel_phonepe_subscription
                await _cancel_phonepe_subscription(sub.phonepe_subscription_id)
                sub.phonepe_subscription_id = None  # Clear mandate reference
            except Exception as e:
                print(f"[Subscription] Failed to cancel PhonePe subscription: {e}", flush=True)

    # If re-enabling auto-renew for Stripe, resume the subscription
    if payload.auto_renew and sub.stripe_subscription_id:
        try:
            stripe.api_key = settings.STRIPE_API_KEY
            stripe.Subscription.modify(
                sub.stripe_subscription_id,
                cancel_at_period_end=False,
            )
        except Exception as e:
            print(f"[Subscription] Failed to resume Stripe subscription: {e}", flush=True)

    await db.commit()
    return {
        "auto_renew": sub.auto_renew,
        "message": f"Auto-renewal {'enabled' if sub.auto_renew else 'disabled'}. "
                   + ("Your plan will continue until the end of the billing period." if not payload.auto_renew else ""),
    }


@router.get("/billing")
async def get_billing_info(
    current_user: CurrentUser,
    db: DBSession,
    org_id: str | None = Query(None),
):
    """Get billing information including next billing date and auto-renewal status."""
    sub = await _get_active_subscription(
        db, current_user.id, uuid.UUID(org_id) if org_id else None
    )
    if not sub:
        raise NoSubscriptionException()

    # Get plan display name
    plan_result = await db.execute(
        select(PlanDefinition).where(PlanDefinition.plan == sub.plan)
    )
    plan_def = plan_result.scalar_one_or_none()

    return {
        "plan": sub.plan,
        "plan_display_name": plan_def.display_name if plan_def else sub.plan,
        "price_inr": plan_def.price_inr if plan_def else 0,
        "status": sub.status,
        "auto_renew": sub.auto_renew,
        "payment_gateway": sub.payment_gateway,
        "current_period_start": sub.current_period_start.isoformat() if sub.current_period_start else None,
        "current_period_end": sub.current_period_end.isoformat() if sub.current_period_end else None,
        "next_billing_date": sub.current_period_end.isoformat() if sub.current_period_end and sub.auto_renew else None,
    }
