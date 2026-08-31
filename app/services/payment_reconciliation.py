"""
Background reconciliation sweep for PhonePe payment_intents.

Both the frontend's redirect-back status check and PhonePe's server-to-server
webhook are best-effort — either can fail to reach us (closed browser tab
right after paying, a webhook delivery failure, a transient error during
fulfilment). When that happens, a payment can succeed on PhonePe's side while
our `payment_intents` row is stuck at status='pending' forever, with nothing
ever re-checking it. This sweep is the safety net: periodically re-query
PhonePe directly for any intent that's been pending too long, and fulfil it
if PhonePe confirms it actually succeeded.
"""

import asyncio
from datetime import datetime, timezone, timedelta

from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.subscription import PaymentIntent

# Give a normal in-progress checkout time to complete naturally before we
# start treating it as possibly-stuck (PhonePe orders expire after 20 min
# anyway — see _create_phonepe_order's expireAfter).
PENDING_GRACE_MINUTES = 10
# Beyond this, a still-"pending" intent is almost certainly abandoned
# (user never completed payment) rather than a fulfilment gap — stop
# re-checking it and mark it failed instead of retrying forever.
ABANDONED_AFTER_HOURS = 24


async def _reconcile_pending_intents():
    from app.routers.payments import (
        _check_phonepe_payment_status,
        _fulfil_purchase,
        _try_claim_payment_intent,
        _release_payment_intent_claim,
    )

    async with AsyncSessionLocal() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=PENDING_GRACE_MINUTES)
        abandoned_cutoff = datetime.now(timezone.utc) - timedelta(hours=ABANDONED_AFTER_HOURS)

        result = await db.execute(
            select(PaymentIntent).where(
                PaymentIntent.status == "pending",
                PaymentIntent.gateway == "phonepe",
                PaymentIntent.created_at <= cutoff,
            )
        )
        intents = result.scalars().all()
        if not intents:
            return

        print(f"[PaymentReconciliation] Checking {len(intents)} pending intent(s)", flush=True)

        for intent in intents:
            # Every step for this intent — including fulfilment — lives inside
            # one try/except with an explicit rollback on failure. Without the
            # rollback, a DB-level error (e.g. a constraint violation) leaves
            # the shared session in an aborted state that would then reject
            # every subsequent command for the REST of this sweep, silently
            # blocking reconciliation for every other pending intent too.
            claimed = False
            try:
                check = await _check_phonepe_payment_status(intent.merchant_order_id, flow=intent.flow)

                if check["normalized_state"] == "COMPLETED":
                    # Claim it atomically first — the frontend's redirect
                    # check or the webhook could confirm this exact same
                    # payment at the same moment this sweep picks it up.
                    claimed = await _try_claim_payment_intent(db, intent.merchant_order_id)
                    if not claimed:
                        print(f"[PaymentReconciliation] {intent.merchant_order_id} already claimed elsewhere, skipping", flush=True)
                        continue

                    print(
                        f"[PaymentReconciliation] Found a stuck-but-successful payment: "
                        f"{intent.merchant_order_id} (user {intent.user_id}, {intent.item_type}/{intent.item_id}) — fulfilling now",
                        flush=True,
                    )
                    await _fulfil_purchase(
                        item_type=intent.item_type,
                        item_id=intent.item_id,
                        user_id=intent.user_id,
                        org_id=str(intent.org_id) if intent.org_id else None,
                        db=db,
                        merchant_order_id=intent.merchant_order_id,
                        gateway="phonepe",
                        phonepe_subscription_id=intent.merchant_order_id if intent.flow == "subscription" else None,
                        promo_code=intent.promo_code,
                        promo_discount=intent.promo_discount,
                        amount_inr=intent.amount_inr,
                    )
                    intent.status = "fulfilled"
                    await db.commit()

                    from app.services.notification_service import create_notification
                    from app.models.notification import NotificationType
                    await create_notification(
                        db, user_id=intent.user_id,
                        notification_type=NotificationType.PAYMENT_SUCCESS,
                        title="Payment Successful",
                        body="Your payment was confirmed and your account has been updated.",
                        icon="check-circle-2", priority="normal",
                        data_json={"plan": intent.item_id if intent.item_type == "plan_upgrade" else None, "link": "/u/plans"},
                    )
                    await db.commit()

                elif check["normalized_state"] == "FAILED":
                    intent.status = "failed"
                    await db.commit()

                elif intent.created_at <= abandoned_cutoff:
                    print(f"[PaymentReconciliation] Giving up on stale pending intent {intent.merchant_order_id}", flush=True)
                    intent.status = "failed"
                    await db.commit()

            except Exception as e:
                print(f"[PaymentReconciliation] Failed to process {intent.merchant_order_id}: {e}", flush=True)
                await db.rollback()
                if claimed:
                    # The claim itself was already committed in its own
                    # transaction before fulfilment failed — rollback above
                    # doesn't undo that. Without this, the intent would be
                    # stuck at "fulfilling" forever, since this sweep's own
                    # query only ever looks at status="pending".
                    try:
                        await _release_payment_intent_claim(db, intent.merchant_order_id)
                    except Exception as release_err:
                        print(f"[PaymentReconciliation] Failed to release claim for {intent.merchant_order_id}: {release_err}", flush=True)
                continue


async def run_payment_reconciliation():
    """Background loop that sweeps stuck payment_intents every 15 minutes."""
    print("[PaymentReconciliation] Started background reconciliation sweep", flush=True)
    while True:
        try:
            await _reconcile_pending_intents()
        except Exception as e:
            print(f"[PaymentReconciliation] Error: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

        await asyncio.sleep(15 * 60)
