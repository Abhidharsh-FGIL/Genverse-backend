import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.subscription import Subscription, PointCost, PointTransaction
from app.core.exceptions import (
    NoSubscriptionException,
    SubscriptionInactiveException,
    InsufficientPointsException,
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
        point_cost = cost_result.scalar_one_or_none()
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
        await db.flush()

        return {
            "success": True,
            "points_used": point_cost.cost,
            "remaining_balance": sub.points_balance,
        }

    async def deduct_custom(
        self,
        user_id: uuid.UUID,
        action: str,
        db: AsyncSession,
        org_id: uuid.UUID | None = None,
        cost_override: int | None = None,
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
        await db.flush()

        return {
            "success": True,
            "points_used": cost,
            "remaining_balance": sub.points_balance,
        }

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
