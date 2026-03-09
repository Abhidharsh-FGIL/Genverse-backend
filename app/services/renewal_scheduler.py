"""
Background scheduler for subscription renewal management.

Runs daily to:
1. Execute PhonePe recurring debits for subscriptions due for renewal
2. Send courtesy renewal reminders 3 days before expiry
3. Expire subscriptions that have passed current_period_end without renewal
   (skips both Stripe and PhonePe auto-renew subs — those are handled by
   webhooks after debit execution)
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models.subscription import Subscription, PlanDefinition, PointTransaction
from app.models.user import User
from app.services.email_service import send_renewal_reminder
from app.config import settings


# Plan prices in paise (amount × 100) — must match PLAN_PRICES in payments.py
PLAN_PRICES_PAISE = {
    "individual_pro": 299_00,
    "individual_power": 799_00,
    "org_basic": 4999_00,
    "org_pro": 15000_00,
    "org_evaluation": 2999_00,
}


async def _execute_phonepe_debit(sub: Subscription, db: AsyncSession):
    """Execute a PhonePe recurring debit for an active mandate."""
    from app.routers.payments import _execute_phonepe_recurring_debit, _fulfil_purchase

    amount_paise = PLAN_PRICES_PAISE.get(sub.plan)
    if not amount_paise:
        print(f"[RenewalScheduler] No price found for plan {sub.plan}, skipping PhonePe debit", flush=True)
        return False

    merchant_order_id = f"phonepe_renewal_{sub.id}_{int(datetime.now(timezone.utc).timestamp())}"

    try:
        result = await _execute_phonepe_recurring_debit(
            merchant_subscription_id=sub.phonepe_subscription_id,
            amount_paise=amount_paise,
            merchant_order_id=merchant_order_id,
        )

        state = result.get("state", "") or result.get("data", {}).get("state", "")

        if state in ("COMPLETED", "SUCCESS"):
            # Immediate success — fulfil now
            await _fulfil_purchase(
                item_type="plan_upgrade",
                item_id=sub.plan,
                user_id=sub.user_id,
                org_id=str(sub.org_id) if sub.org_id else None,
                db=db,
                merchant_order_id=merchant_order_id,
                gateway="phonepe",
                phonepe_subscription_id=sub.phonepe_subscription_id,
                is_renewal=True,
            )
            print(f"[RenewalScheduler] PhonePe debit SUCCESS for sub {sub.id}", flush=True)
            return True

        elif state in ("PENDING", "INITIATED"):
            # Debit initiated — webhook will handle fulfilment when PhonePe confirms
            print(f"[RenewalScheduler] PhonePe debit PENDING for sub {sub.id} — webhook will handle", flush=True)
            return True  # Don't expire — waiting for webhook

        else:
            print(f"[RenewalScheduler] PhonePe debit FAILED for sub {sub.id}: {state}", flush=True)
            return False

    except Exception as e:
        print(f"[RenewalScheduler] PhonePe debit error for sub {sub.id}: {e}", flush=True)
        return False


async def _check_renewals():
    """Check for expiring and expired subscriptions."""
    print("[RenewalScheduler] Running renewal check...", flush=True)

    async with AsyncSessionLocal() as db:
        now = datetime.now(timezone.utc)
        three_days_from_now = now + timedelta(days=3)

        # ── 1. Execute PhonePe recurring debits for subscriptions due today ──
        result = await db.execute(
            select(Subscription).where(
                Subscription.status == "active",
                Subscription.plan != "free",
                Subscription.current_period_end.isnot(None),
                Subscription.current_period_end <= now,
                Subscription.payment_gateway == "phonepe",
                Subscription.auto_renew == True,
                Subscription.phonepe_subscription_id.isnot(None),
            )
        )
        phonepe_due_subs = result.scalars().all()
        phonepe_success = 0
        phonepe_failed = 0

        for sub in phonepe_due_subs:
            success = await _execute_phonepe_debit(sub, db)
            if success:
                phonepe_success += 1
            else:
                phonepe_failed += 1

        # ── 2. Send courtesy reminders 3 days before expiry ──
        result = await db.execute(
            select(Subscription).where(
                Subscription.status == "active",
                Subscription.plan != "free",
                Subscription.current_period_end.isnot(None),
                Subscription.current_period_end <= three_days_from_now,
                Subscription.current_period_end > now,
                Subscription.auto_renew == True,
            )
        )
        expiring_subs = result.scalars().all()
        reminders_sent = 0

        for sub in expiring_subs:
            user_result = await db.execute(
                select(User).where(User.id == sub.user_id)
            )
            user = user_result.scalar_one_or_none()
            if not user:
                continue

            plan_result = await db.execute(
                select(PlanDefinition).where(PlanDefinition.plan == sub.plan)
            )
            plan_def = plan_result.scalar_one_or_none()
            plan_name = plan_def.display_name if plan_def else sub.plan

            expiry_str = sub.current_period_end.strftime("%B %d, %Y")
            renew_url = f"{settings.FRONTEND_URL}/u/plans"

            send_renewal_reminder(user.email, plan_name, expiry_str, renew_url)
            reminders_sent += 1
            print(f"[RenewalScheduler] Sent reminder to {user.email} for {plan_name} expiring {expiry_str}", flush=True)

        # ── 3. Expire subscriptions past their period end ──
        # Skip Stripe auto-renew (webhook handles) and PhonePe auto-renew with mandate (debit handles)
        result = await db.execute(
            select(Subscription).where(
                Subscription.status == "active",
                Subscription.plan != "free",
                Subscription.current_period_end.isnot(None),
                Subscription.current_period_end <= now,
                # Don't expire Stripe auto-renewing subs
                ~and_(
                    Subscription.payment_gateway == "stripe",
                    Subscription.auto_renew == True,
                    Subscription.stripe_subscription_id.isnot(None),
                ),
                # Don't expire PhonePe auto-renewing subs with active mandate
                ~and_(
                    Subscription.payment_gateway == "phonepe",
                    Subscription.auto_renew == True,
                    Subscription.phonepe_subscription_id.isnot(None),
                ),
            )
        )
        expired_subs = result.scalars().all()

        for sub in expired_subs:
            # Double-check: skip any auto-renew subs with active gateway IDs
            if sub.auto_renew:
                if sub.payment_gateway == "stripe" and sub.stripe_subscription_id:
                    continue
                if sub.payment_gateway == "phonepe" and sub.phonepe_subscription_id:
                    continue

            old_plan = sub.plan
            sub.status = "expired"
            sub.auto_renew = False

            txn = PointTransaction(
                subscription_id=sub.id,
                user_id=sub.user_id,
                action=f"plan_expired_{old_plan}",
                points_used=0,
                balance_after=sub.points_balance,
            )
            db.add(txn)
            print(f"[RenewalScheduler] Expired subscription {sub.id} (plan: {old_plan})", flush=True)

        await db.commit()
        print(
            f"[RenewalScheduler] Done. "
            f"PhonePe debits: {phonepe_success} ok / {phonepe_failed} failed, "
            f"Reminders: {reminders_sent}, "
            f"Expired: {len(expired_subs)}",
            flush=True,
        )


async def run_renewal_scheduler():
    """Background loop that checks renewals once per day."""
    print("[RenewalScheduler] Started background renewal scheduler", flush=True)
    while True:
        try:
            await _check_renewals()
        except Exception as e:
            print(f"[RenewalScheduler] Error: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

        # Sleep 24 hours
        await asyncio.sleep(24 * 60 * 60)
