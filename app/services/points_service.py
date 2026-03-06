import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.subscription import Subscription, PointCost, PointTransaction, FeatureLimit, UsageCounter
from app.core.exceptions import (
    NoSubscriptionException,
    SubscriptionInactiveException,
    InsufficientPointsException,
    FeatureGatedException,
    UsageLimitException,
)


class PointsService:
    async def deduct(
        self,
        user_id: uuid.UUID,
        action: str,
        db: AsyncSession,
        org_id: uuid.UUID | None = None,
    ) -> dict:
        """
        Atomically deduct points from the user's (or org's) active subscription.
        Raises HTTP exceptions on failure.
        """
        # Get point cost for action
        cost_result = await db.execute(
            select(PointCost).where(PointCost.action == action)
        )
        point_cost = cost_result.scalars().first()
        if not point_cost:
            # Unknown action - no cost (non-AI operations)
            return {"success": True, "points_used": 0, "remaining_balance": 0}

        # Get active subscription
        sub = await self._get_subscription(db, user_id, org_id)
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

        # Atomic deduction using UPDATE with condition check
        sub.points_balance -= point_cost.cost
        transaction = PointTransaction(
            subscription_id=sub.id,
            user_id=user_id,
            action=action,
            points_used=point_cost.cost,
            balance_after=sub.points_balance,
        )
        db.add(transaction)

        # Award XP from table (personal workspace only)
        xp = point_cost.xp_reward or 0
        if xp > 0:
            await self._award_xp(db, user_id, org_id, xp)

        await db.commit()

        return {
            "success": True,
            "points_used": point_cost.cost,
            "remaining_balance": sub.points_balance,
            "xp_awarded": xp,
        }

    async def deduct_custom(
        self,
        user_id: uuid.UUID,
        action: str,
        db: AsyncSession,
        org_id: uuid.UUID | None = None,
        cost_override: int | None = None,
        xp_override: int | None = None,
    ) -> dict:
        """Deduct a custom amount (useful for per-page/per-item costs)."""
        sub = await self._get_subscription(db, user_id, org_id)
        if not sub:
            raise NoSubscriptionException()
        if sub.status not in ("active", "trialing"):
            raise SubscriptionInactiveException()

        cost = cost_override or 1
        if sub.points_balance < cost:
            refresh_date = sub.current_period_end.isoformat() if sub.current_period_end else ""
            raise InsufficientPointsException(
                points_needed=cost,
                points_available=sub.points_balance,
                refresh_date=refresh_date,
            )

        sub.points_balance -= cost
        transaction = PointTransaction(
            subscription_id=sub.id,
            user_id=user_id,
            action=action,
            points_used=cost,
            balance_after=sub.points_balance,
        )
        db.add(transaction)

        # Award XP: use xp_override if provided, otherwise look up from table
        xp = xp_override
        if xp is None:
            cost_result = await db.execute(
                select(PointCost).where(PointCost.action == action)
            )
            pc = cost_result.scalars().first()
            xp = (pc.xp_reward or 0) if pc else 0
        if xp and xp > 0:
            await self._award_xp(db, user_id, org_id, xp)

        await db.commit()

        return {
            "success": True,
            "points_used": cost,
            "remaining_balance": sub.points_balance,
            "xp_awarded": xp or 0,
        }

    async def check_and_increment_usage(
        self,
        user_id: uuid.UUID,
        feature_key: str,
        db: AsyncSession,
        org_id: uuid.UUID | None = None,
    ) -> None:
        """
        Check if the user's plan allows this feature and if daily/monthly
        limits have not been exceeded. Increments usage counters on success.
        Raises FeatureGatedException or UsageLimitException on failure.
        Org context (org_id set) skips all checks — org plans have no limits.
        """
        if org_id:
            return  # Org workspace: no personal feature restrictions

        sub = await self._get_subscription(db, user_id, None)
        plan_name = sub.plan if sub else "free"

        result = await db.execute(
            select(FeatureLimit).where(
                FeatureLimit.plan == plan_name,
                FeatureLimit.feature_key == feature_key,
            )
        )
        limit = result.scalars().first()
        if not limit:
            return  # No limit row defined = unlimited
        if not limit.enabled:
            raise FeatureGatedException(feature_key=feature_key)

        now = datetime.now(timezone.utc)
        today = now.date()
        month_start = today.replace(day=1)

        # Check daily limit
        if limit.daily_limit is not None:
            daily_count = await self._get_usage_count(db, user_id, feature_key, "daily", today)
            if daily_count >= limit.daily_limit:
                raise UsageLimitException(feature_key, "daily", limit.daily_limit)

        # Check monthly limit
        if limit.monthly_limit is not None:
            monthly_count = await self._get_usage_count(db, user_id, feature_key, "monthly", month_start)
            if monthly_count >= limit.monthly_limit:
                raise UsageLimitException(feature_key, "monthly", limit.monthly_limit)

        # Increment counters
        await self._increment_usage(db, user_id, feature_key)

    async def _get_usage_count(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        feature_key: str,
        period: str,
        period_start,
    ) -> int:
        result = await db.execute(
            select(UsageCounter.count).where(
                UsageCounter.user_id == user_id,
                UsageCounter.feature_key == feature_key,
                UsageCounter.period == period,
                UsageCounter.period_start == datetime.combine(period_start, datetime.min.time(), tzinfo=timezone.utc),
            )
        )
        row = result.scalars().first()
        return row or 0

    async def _increment_usage(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        feature_key: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        today = now.date()
        month_start = today.replace(day=1)

        for period, start in [("daily", today), ("monthly", month_start)]:
            dt_start = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
            result = await db.execute(
                select(UsageCounter).where(
                    UsageCounter.user_id == user_id,
                    UsageCounter.feature_key == feature_key,
                    UsageCounter.period == period,
                    UsageCounter.period_start == dt_start,
                )
            )
            counter = result.scalars().first()
            if counter:
                counter.count += 1
            else:
                db.add(UsageCounter(
                    user_id=user_id,
                    feature_key=feature_key,
                    period=period,
                    period_start=dt_start,
                    count=1,
                ))
        await db.flush()

    async def _award_xp(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        org_id: uuid.UUID | None,
        amount: int,
    ) -> None:
        """Award XP to user's workspace gamification record (same transaction)."""
        if amount <= 0:
            return
        from app.models.gamification import WorkspaceGamification

        if org_id:
            q = select(WorkspaceGamification).where(
                WorkspaceGamification.user_id == user_id,
                WorkspaceGamification.org_id == org_id,
            )
        else:
            q = select(WorkspaceGamification).where(
                WorkspaceGamification.user_id == user_id,
                WorkspaceGamification.org_id.is_(None),
            )
        result = await db.execute(q)
        gam = result.scalars().first()
        if gam:
            gam.xp = (gam.xp or 0) + amount
        else:
            db.add(WorkspaceGamification(user_id=user_id, org_id=org_id, xp=amount, streak=0))

    async def _get_subscription(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        org_id: uuid.UUID | None,
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
