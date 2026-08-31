"""
Payment gateway integration: PhonePe Standard Checkout v2 (India / UPI) and Stripe (International).

PhonePe flow — redirect-based:
  1. Frontend calls POST /checkout → backend gets OAuth token, creates PhonePe order → returns redirect URL
  2. User completes payment on PhonePe's hosted page
  3. PhonePe redirects back to success_url with merchantOrderId
  4. Frontend calls GET /phonepe/status/{merchant_order_id} to confirm & fulfil

Stripe flow — subscription-based for plan upgrades, one-time for point packs:
  1. Frontend calls POST /checkout → backend creates Stripe Customer + Subscription/Checkout → returns redirect URL
  2. User completes payment on Stripe's hosted page
  3. Stripe redirects back; webhook handles invoice.paid for auto-renewal
  4. Frontend calls GET /status/{session_id} to confirm

Auto-renewal:
  - Stripe: Uses native recurring subscriptions (invoice.paid webhook)
  - PhonePe: Manual renewal (email reminders sent before expiry via background scheduler)
"""

import uuid
import hmac
import hashlib
from datetime import datetime, timezone, timedelta
from urllib.parse import quote as url_quote

import httpx
import stripe
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select, and_, update

from app.config import settings
from app.dependencies import DBSession, CurrentUser
from app.models.subscription import Subscription, PlanDefinition, SubscriptionAddon, PointTransaction, PaymentIntent
from app.models.promo import PromoCode, PromoUsage
from app.models.user import User
from app.services.email_service import send_purchase_invoice_email

router = APIRouter()

# ---------------------------------------------------------------------------
# Configure Stripe
# ---------------------------------------------------------------------------
stripe.api_key = settings.STRIPE_API_KEY

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CheckoutRequest(BaseModel):
    gateway: str  # "phonepe" or "stripe"
    item_type: str  # "plan_upgrade" or "point_pack"
    item_id: str
    amount_inr: int
    org_id: Optional[str] = None
    promo_code: Optional[str] = None
    success_url: str = "http://localhost:4200/u/plans?payment=success"
    cancel_url: str = "http://localhost:4200/u/plans?payment=cancelled"


class PromoValidateRequest(BaseModel):
    code: str
    item_type: str  # "plan_upgrade" or "point_pack"
    item_id: str
    amount_inr: int


class PromoValidateResponse(BaseModel):
    valid: bool
    message: str
    discount_amount: int = 0
    final_amount: int = 0
    discount_label: str = ""


class CheckoutResponse(BaseModel):
    session_id: str
    redirect_url: str
    gateway: str


class PaymentStatusResponse(BaseModel):
    status: str  # "success" | "pending" | "failed"
    message: str
    plan: Optional[str] = None
    points_added: Optional[int] = None


# ---------------------------------------------------------------------------
# Plan price map (amount in INR) — ideally from DB, hardcoded for MVP
# ---------------------------------------------------------------------------
PLAN_PRICES = {
    "individual_pro": 299,
    "individual_power": 799,
    "org_basic": 4999,
    "org_pro": 15000,
    "org_evaluation": 2999,
}

ADDON_POINT_MAP = {
    "starter_200": {"points": 200, "price": 49},
    "learner_500": {"points": 500, "price": 99},
    "pro_1500": {"points": 1500, "price": 249},
    # Frontend sends these IDs (points_N format)
    "points_200": {"points": 200, "price": 49},
    "points_500": {"points": 500, "price": 99},
    "points_1500": {"points": 1500, "price": 249},
}


# ---------------------------------------------------------------------------
# PhonePe helpers
# ---------------------------------------------------------------------------

# In-memory token cache (single-server; for multi-server use Redis)
_phonepe_token_cache: dict = {"token": None, "expires_at": 0}


async def _get_phonepe_access_token() -> str:
    """Get or refresh a PhonePe OAuth access token."""
    now = datetime.now(timezone.utc).timestamp()
    if _phonepe_token_cache["token"] and _phonepe_token_cache["expires_at"] > now + 60:
        return _phonepe_token_cache["token"]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            settings.PHONEPE_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": settings.PHONEPE_CLIENT_ID,
                "client_secret": settings.PHONEPE_CLIENT_SECRET,
                "client_version": str(settings.PHONEPE_CLIENT_VERSION),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            print(f"[PhonePe] Token request failed: {resp.status_code} {resp.text}", flush=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Payment service is temporarily unavailable. Please try again in a few minutes.",
            )
        data = resp.json()
        _phonepe_token_cache["token"] = data["access_token"]
        _phonepe_token_cache["expires_at"] = now + data.get("expires_in", 3600)
        return data["access_token"]


async def _create_phonepe_order(
    amount_inr: int,
    merchant_order_id: str,
    redirect_url: str,
    user_id: uuid.UUID | None = None,
    item_type: str = "",
    item_id: str = "",
    org_id: str = "",
    promo_code: str = "",
    promo_discount: int = 0,
) -> dict:
    """Create a PhonePe Standard Checkout order."""
    access_token = await _get_phonepe_access_token()
    order_url = f"{settings.PHONEPE_BASE_URL}/checkout/v2/pay"

    payload = {
        "merchantOrderId": merchant_order_id,
        "amount": amount_inr * 100,  # paise
        "expireAfter": 1200,  # 20 minutes
        "metaInfo": {
            # udf fields carry order context for the server-side webhook
            "udf1": str(user_id) if user_id else "",
            "udf2": item_type,
            "udf3": item_id,
            "udf4": org_id or "",
            "udf5": f"{promo_code}|{promo_discount}",
        },
        "paymentFlow": {
            "type": "PG_CHECKOUT",
            "merchantUrls": {
                "redirectUrl": redirect_url,
            },
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            order_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"O-Bearer {access_token}",
            },
        )
        data = resp.json()
        print(f"[PhonePe] Order response ({resp.status_code}): {data}", flush=True)
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not initiate the payment. Please try again.",
            )
        return data


async def _get_phonepe_order_status(merchant_order_id: str) -> dict:
    """Check the status of a PhonePe ONE-TIME order (Standard Checkout).

    Only valid for merchant_order_ids created via _create_phonepe_order.
    Do not call this for a recurring-subscription order — use
    _get_phonepe_subscription_status instead (see _check_phonepe_payment_status).
    """
    access_token = await _get_phonepe_access_token()
    status_url = f"{settings.PHONEPE_BASE_URL}/checkout/v2/order/{merchant_order_id}/status"

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            status_url,
            headers={"Authorization": f"O-Bearer {access_token}"},
        )
        if resp.status_code != 200:
            print(f"[PhonePe] Order status check failed: {resp.status_code} {resp.text}", flush=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to check payment status",
            )
        return resp.json()


async def _get_phonepe_subscription_status(merchant_subscription_id: str) -> dict:
    """Check the status of a PhonePe recurring UPI AutoPay subscription/mandate.

    This is a COMPLETELY DIFFERENT API from _get_phonepe_order_status, with a
    different state vocabulary (ACTIVATION_IN_PROGRESS / ACTIVE / EXPIRED /
    FAILED / CANCELLED / REVOKED / ... — never "COMPLETED"). Required because
    create_checkout tries a recurring mandate FIRST for every plan upgrade
    (see _create_phonepe_subscription) — a payment made through that flow can
    never be confirmed by querying the one-time-order status endpoint.
    """
    access_token = await _get_phonepe_access_token()
    status_url = f"{settings.PHONEPE_BASE_URL}/subscriptions/v2/{merchant_subscription_id}/status?details=true"

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            status_url,
            headers={"Authorization": f"O-Bearer {access_token}"},
        )
        if resp.status_code != 200:
            print(f"[PhonePe] Subscription status check failed: {resp.status_code} {resp.text}", flush=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to check payment status",
            )
        return resp.json()


async def _check_phonepe_payment_status(merchant_order_id: str, flow: str | None = None) -> dict:
    """Resolve a merchant_order_id's payment status regardless of whether it
    was created via the standard one-time Checkout API or the recurring UPI
    AutoPay subscription API, normalising both into one vocabulary.

    Pass `flow` ("order" | "subscription") when known — from the local
    PaymentIntent record — to check the right endpoint directly. When
    unknown, tries the order-status endpoint first and falls back to the
    subscription-status endpoint (covers orders created before PaymentIntent
    existed, or if the two ever get out of sync).

    Returns {"flow": "order"|"subscription", "normalized_state": "COMPLETED"|"PENDING"|"FAILED", "raw": {...}}.
    """
    async def _try_order() -> dict | None:
        try:
            raw = await _get_phonepe_order_status(merchant_order_id)
        except HTTPException:
            return None
        state = raw.get("state") or raw.get("data", {}).get("state", "")
        if not state:
            return None
        # Only map to FAILED for a state we KNOW means failure. Any
        # unrecognized value (e.g. this endpoint returning something odd for
        # what's actually a subscription-flow ID, or a future PhonePe state
        # we haven't seen) defaults to PENDING — safe to recheck later, unlike
        # FAILED which stops fulfilment from ever being retried for a
        # payment that may have genuinely succeeded.
        if state == "COMPLETED":
            normalized = "COMPLETED"
        elif state in ("FAILED", "CANCELLED", "EXPIRED"):
            normalized = "FAILED"
        else:  # PENDING, INITIATED, or anything unrecognized
            normalized = "PENDING"
        return {"flow": "order", "normalized_state": normalized, "raw": raw}

    async def _try_subscription() -> dict | None:
        try:
            raw = await _get_phonepe_subscription_status(merchant_order_id)
        except HTTPException:
            return None
        state = raw.get("state", "")
        if not state:
            return None
        # Same reasoning as _try_order: only a state we KNOW is terminal-failure
        # maps to FAILED; anything unrecognized defaults to PENDING rather than
        # risking a real payment getting permanently marked as failed.
        if state == "ACTIVE":
            normalized = "COMPLETED"
        elif state in ("EXPIRED", "FAILED", "CANCELLED", "REVOKED"):
            normalized = "FAILED"
        else:  # ACTIVATION_IN_PROGRESS, PAUSE_IN_PROGRESS, PAUSED, UNPAUSE_IN_PROGRESS,
               # CANCEL_IN_PROGRESS, REVOKE_IN_PROGRESS, or anything unrecognized
            normalized = "PENDING"
        return {"flow": "subscription", "normalized_state": normalized, "raw": raw}

    if flow == "subscription":
        result = await _try_subscription()
        if result:
            return result
        result = await _try_order()
    else:
        result = await _try_order()
        if result:
            return result
        result = await _try_subscription()

    if result:
        return result

    print(f"[PhonePe] Could not resolve status for {merchant_order_id} via either order or subscription API", flush=True)
    return {"flow": "unknown", "normalized_state": "PENDING", "raw": {}}


async def _try_claim_payment_intent(db, merchant_order_id: str) -> bool:
    """Atomically claim a PaymentIntent for fulfilment, using its `status`
    column as a compare-and-swap lock (pending -> fulfilling).

    Needed because /phonepe/status/{id} (frontend redirect), /phonepe/webhook,
    and the payment_reconciliation sweep can now all genuinely succeed at
    confirming the same payment — before this fix, in practice only one path
    ever actually worked (the others were broken), so this race was
    theoretical. Now that all three work, they can fire close enough together
    to both pass _fulfil_purchase's time-window dedup check before either has
    committed, double-crediting the user. This claim makes only one caller
    proceed; the loser should re-read the intent's (now updated) status
    instead of calling _fulfil_purchase itself.

    Returns True only if THIS caller won the race. On any exception during
    the caller's subsequent fulfilment work, the caller MUST call
    _release_payment_intent_claim to revert the status back to "pending" —
    otherwise the intent is stuck at "fulfilling" forever and the
    reconciliation sweep (which only looks at status="pending") will never
    retry it.
    """
    result = await db.execute(
        update(PaymentIntent)
        .where(PaymentIntent.merchant_order_id == merchant_order_id, PaymentIntent.status == "pending")
        .values(status="fulfilling")
    )
    await db.commit()
    return result.rowcount > 0


async def _release_payment_intent_claim(db, merchant_order_id: str) -> None:
    """Revert a claimed-but-failed PaymentIntent back to "pending" so it can
    be retried (by a later webhook call, redirect check, or the
    reconciliation sweep) instead of being stuck at "fulfilling" forever."""
    await db.execute(
        update(PaymentIntent)
        .where(PaymentIntent.merchant_order_id == merchant_order_id, PaymentIntent.status == "fulfilling")
        .values(status="pending")
    )
    await db.commit()


# ---------------------------------------------------------------------------
# PhonePe Recurring (UPI AutoPay) helpers
# ---------------------------------------------------------------------------

async def _create_phonepe_subscription(
    amount_inr: int,
    merchant_subscription_id: str,
    merchant_user_id: str,
    redirect_url: str,
) -> dict:
    """Create a PhonePe recurring subscription (UPI AutoPay mandate)."""
    access_token = await _get_phonepe_access_token()
    sub_url = f"{settings.PHONEPE_BASE_URL}/v3/recurring/subscription/create"

    payload = {
        "merchantSubscriptionId": merchant_subscription_id,
        "merchantUserId": merchant_user_id,
        "authWorkflowType": "PENNY_DROP",
        "amountType": "FIXED",
        "amount": amount_inr * 100,  # paise
        "frequency": "MONTHLY",
        "recurringCount": 12,  # max 12 months
        "description": f"GenVerse Plan Subscription - {amount_inr} INR/month",
        "paymentFlow": {
            "type": "PG_CHECKOUT",
            "merchantUrls": {
                "redirectUrl": redirect_url,
            },
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            sub_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"O-Bearer {access_token}",
            },
        )
        data = resp.json()
        print(f"[PhonePe] Subscription response ({resp.status_code}): {data}", flush=True)
        if resp.status_code not in (200, 201):
            # Fallback to regular payment if recurring API not available
            print(f"[PhonePe] Recurring API not available, falling back to one-time payment", flush=True)
            return {"fallback": True}
        return data


async def _cancel_phonepe_subscription(merchant_subscription_id: str) -> bool:
    """Cancel a PhonePe recurring subscription."""
    access_token = await _get_phonepe_access_token()
    cancel_url = f"{settings.PHONEPE_BASE_URL}/v3/recurring/subscription/{merchant_subscription_id}/cancel"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            cancel_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"O-Bearer {access_token}",
            },
        )
        print(f"[PhonePe] Cancel subscription response ({resp.status_code}): {resp.text}", flush=True)
        return resp.status_code in (200, 201)


async def _execute_phonepe_recurring_debit(
    merchant_subscription_id: str,
    amount_paise: int,
    merchant_order_id: str,
) -> dict:
    """Execute a recurring debit against an active PhonePe mandate.

    Called by the renewal scheduler on the subscription's expiry date
    to auto-charge the user's linked UPI account.
    """
    access_token = await _get_phonepe_access_token()
    debit_url = f"{settings.PHONEPE_BASE_URL}/v3/recurring/debit/execute"

    payload = {
        "merchantSubscriptionId": merchant_subscription_id,
        "transactionId": merchant_order_id,
        "amount": amount_paise,
        "notifyUrl": f"{settings.BACKEND_URL}/api/v1/payments/phonepe/recurring-webhook",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            debit_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"O-Bearer {access_token}",
            },
        )
        data = resp.json()
        print(f"[PhonePe] Recurring debit response ({resp.status_code}): {data}", flush=True)
        return data


# ---------------------------------------------------------------------------
# Stripe helpers
# ---------------------------------------------------------------------------

async def _get_or_create_stripe_customer(user_id: uuid.UUID, email: str, name: str, db) -> str:
    """Get existing Stripe customer ID from subscription or create a new one."""
    # Check if user already has a Stripe customer ID
    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.stripe_customer_id.isnot(None),
        )
    )
    sub = result.scalar_one_or_none()
    if sub and sub.stripe_customer_id:
        return sub.stripe_customer_id

    # Create new Stripe customer
    customer = stripe.Customer.create(
        email=email,
        name=name,
        metadata={"user_id": str(user_id)},
    )
    return customer.id


async def _cancel_existing_stripe_subscription(sub: Subscription):
    """Cancel any existing Stripe subscription before creating a new one."""
    if sub.stripe_subscription_id:
        try:
            stripe.Subscription.cancel(sub.stripe_subscription_id)
            print(f"[Stripe] Cancelled old subscription {sub.stripe_subscription_id}", flush=True)
        except Exception as e:
            print(f"[Stripe] Failed to cancel old subscription: {e}", flush=True)


# ---------------------------------------------------------------------------
# Fulfilment helper
# ---------------------------------------------------------------------------

async def _fulfil_purchase(
    item_type: str, item_id: str, user_id: uuid.UUID, org_id: str | None, db,
    merchant_order_id: str | None = None,
    gateway: str | None = None,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    phonepe_subscription_id: str | None = None,
    is_renewal: bool = False,
    promo_code: str | None = None,
    promo_discount: int = 0,
    background_tasks: BackgroundTasks | None = None,
    amount_inr: int | None = None,
):
    """Apply the purchased plan/addon to the user's subscription.
    Idempotent — skips if the same order was already fulfilled.

    `amount_inr` should be the actual amount charged for THIS purchase
    (post-promo, the final amount) — pass it whenever it's known (from the
    local PaymentIntent record, or straight off a webhook payload) rather
    than leaving it unset, since PLAN_PRICES/ADDON_POINT_MAP are hardcoded
    fallbacks that can silently drift out of sync with the real price
    (e.g. plan_definitions.price_inr for individual_power is 999 while
    PLAN_PRICES has 799 — one of them is stale)."""
    # Deduplication: if a transaction for this exact item exists in the last 2 min, skip
    # Uses exact action match and .first() to avoid errors with multiple rows
    if merchant_order_id:
        two_min_ago = datetime.now(timezone.utc) - timedelta(minutes=2)
        # Build the exact action name that will be created
        if item_type == "plan_upgrade":
            expected_action = f"renewal_{item_id}" if is_renewal else f"plan_upgrade_{item_id}"
        else:
            # Point pack: normalise to canonical name
            canonical = item_id
            if item_id.startswith("points_"):
                pts = item_id.replace("points_", "")
                name_map = {"200": "starter_200", "500": "learner_500", "1500": "pro_1500"}
                canonical = name_map.get(pts, item_id)
            expected_action = f"purchased_{canonical}"

        dup_check = await db.execute(
            select(PointTransaction).where(
                PointTransaction.user_id == user_id,
                PointTransaction.action == expected_action,
                PointTransaction.created_at >= two_min_ago,
            )
        )
        if dup_check.scalars().first():
            print(f"[Payment] Skipping duplicate fulfilment for {merchant_order_id} (action: {expected_action})", flush=True)
            result = await db.execute(
                select(Subscription).where(Subscription.user_id == user_id)
            )
            return result.scalars().first()

    # Find or create subscription
    filters = [Subscription.user_id == user_id]
    if org_id:
        filters.append(Subscription.org_id == uuid.UUID(org_id))
    else:
        filters.append(Subscription.org_id.is_(None))

    result = await db.execute(select(Subscription).where(and_(*filters)))
    sub = result.scalar_one_or_none()

    if not sub:
        sub = Subscription(user_id=user_id, plan="free", status="active")
        db.add(sub)
        await db.flush()

    now = datetime.now(timezone.utc)

    if item_type == "plan_upgrade":
        old_balance = sub.points_balance
        sub.plan = item_id
        sub.status = "active"

        # Set billing period (30 days from now)
        sub.current_period_start = now
        sub.current_period_end = now + timedelta(days=30)

        # Enable auto-renewal and store gateway info
        sub.auto_renew = True
        if gateway:
            sub.payment_gateway = gateway
        if stripe_customer_id:
            sub.stripe_customer_id = stripe_customer_id
        if stripe_subscription_id:
            sub.stripe_subscription_id = stripe_subscription_id
        if phonepe_subscription_id:
            sub.phonepe_subscription_id = phonepe_subscription_id

        # Update points quota from plan definition
        plan_result = await db.execute(
            select(PlanDefinition).where(PlanDefinition.plan == item_id)
        )
        plan_def = plan_result.scalar_one_or_none()
        if plan_def:
            sub.points_monthly_quota = plan_def.monthly_points
            # Add new plan's monthly points to existing balance (carry over remaining)
            sub.points_balance = old_balance + plan_def.monthly_points
            sub.storage_limit_mb = plan_def.storage_mb
            if plan_def.max_seats:
                sub.max_seats = plan_def.max_seats

            # Record the plan upgrade/renewal as a point transaction (credit)
            action_label = f"renewal_{item_id}" if is_renewal else f"plan_upgrade_{item_id}"
            txn_meta = {"gateway": gateway or "N/A"}
            final_amount = amount_inr if amount_inr is not None else PLAN_PRICES.get(item_id, 0)
            txn_meta["final_amount"] = final_amount
            if promo_code and promo_discount:
                txn_meta["promo_code"] = promo_code
                txn_meta["promo_discount"] = promo_discount
                txn_meta["original_amount"] = final_amount + promo_discount
            txn = PointTransaction(
                subscription_id=sub.id,
                user_id=user_id,
                action=action_label,
                points_used=-plan_def.monthly_points,  # negative = credit
                balance_after=sub.points_balance,
                metadata_json=txn_meta,
            )
            db.add(txn)

    elif item_type == "point_pack":
        addon_info = ADDON_POINT_MAP.get(item_id)
        if addon_info:
            points_to_add = addon_info["points"]
            sub.points_balance += points_to_add

            # Normalise addon_type to the canonical name for display
            canonical_type = item_id
            if item_id.startswith("points_"):
                pts = item_id.replace("points_", "")
                name_map = {"200": "starter_200", "500": "learner_500", "1500": "pro_1500"}
                canonical_type = name_map.get(pts, item_id)

            # Record addon
            addon = SubscriptionAddon(
                subscription_id=sub.id,
                addon_type=canonical_type,
                quantity=1,
                points_added=points_to_add,
            )
            db.add(addon)
            # Record transaction
            addon_meta = {"gateway": gateway or "N/A"}
            addon_final_amount = amount_inr if amount_inr is not None else addon_info.get("price", 0)
            addon_meta["final_amount"] = addon_final_amount
            if promo_code and promo_discount:
                addon_meta["promo_code"] = promo_code
                addon_meta["promo_discount"] = promo_discount
                addon_meta["original_amount"] = addon_final_amount + promo_discount
            txn = PointTransaction(
                subscription_id=sub.id,
                user_id=user_id,
                action=f"purchased_{canonical_type}",
                points_used=-points_to_add,  # negative = credit
                balance_after=sub.points_balance,
                metadata_json=addon_meta,
            )
            db.add(txn)

    await db.commit()

    # ── Send invoice email (fire-and-forget, don't block on failure) ──
    try:
        user_result = await db.execute(select(User).where(User.id == user_id))
        user_obj = user_result.scalar_one_or_none()
        if user_obj and user_obj.email:
            # Build invoice details
            inv_date = datetime.now(timezone.utc).strftime("%Y%m%d")
            inv_id = f"GV-INV-{inv_date}-{str(sub.id)[:8].upper()}"
            txn_id = str(merchant_order_id or sub.id)

            if item_type == "plan_upgrade":
                plan_def_result = await db.execute(
                    select(PlanDefinition).where(PlanDefinition.plan == item_id)
                )
                pd = plan_def_result.scalar_one_or_none()
                label = pd.display_name if pd else item_id.replace("_", " ").title()
                final_price = amount_inr if amount_inr is not None else PLAN_PRICES.get(item_id, 0)
                price = final_price + promo_discount if promo_discount else final_price
                period_start = sub.current_period_start.strftime("%b %d, %Y") if sub.current_period_start else None
                period_end = sub.current_period_end.strftime("%b %d, %Y") if sub.current_period_end else None

                invoice_kwargs = dict(
                    to_email=user_obj.email,
                    user_name=user_obj.name or "User",
                    invoice_id=inv_id,
                    transaction_id=txn_id,
                    item_label=label,
                    item_type="Plan Auto-Renewal" if is_renewal else "Plan Subscription",
                    plan_name=label,
                    amount_inr=final_price,
                    payment_gateway=gateway or sub.payment_gateway or "N/A",
                    is_renewal=is_renewal,
                    billing_period_start=period_start,
                    billing_period_end=period_end,
                    promo_code=promo_code,
                    promo_discount=promo_discount,
                    original_amount=price if promo_discount else None,
                )
                if background_tasks:
                    background_tasks.add_task(send_purchase_invoice_email, **invoice_kwargs)
                else:
                    send_purchase_invoice_email(**invoice_kwargs)
            elif item_type == "point_pack":
                addon_info = ADDON_POINT_MAP.get(item_id, {})
                pts = addon_info.get("points", 0)
                label = f"{pts} AI Points Pack"
                final_price = amount_inr if amount_inr is not None else addon_info.get("price", 0)
                price = final_price + promo_discount if promo_discount else final_price

                invoice_kwargs = dict(
                    to_email=user_obj.email,
                    user_name=user_obj.name or "User",
                    invoice_id=inv_id,
                    transaction_id=txn_id,
                    item_label=label,
                    item_type="Add-on Points Pack",
                    plan_name=sub.plan.replace("_", " ").title() if sub.plan else "Free",
                    amount_inr=final_price,
                    payment_gateway=gateway or sub.payment_gateway or "N/A",
                    is_renewal=False,
                    promo_code=promo_code,
                    promo_discount=promo_discount,
                    original_amount=price if promo_discount else None,
                )
                if background_tasks:
                    background_tasks.add_task(send_purchase_invoice_email, **invoice_kwargs)
                else:
                    send_purchase_invoice_email(**invoice_kwargs)
    except Exception as e:
        # Email failure should never block the purchase flow
        print(f"[Payment] Invoice email failed (non-blocking): {e}", flush=True)

    return sub


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def _validate_promo(
    code: str, item_type: str, item_id: str, amount_inr: int,
    user_id: uuid.UUID, db,
) -> dict:
    """Validate a promo code and return discount details.
    Returns dict with keys: valid, message, discount_amount, final_amount, discount_label, promo (model).
    """
    code_upper = code.strip().upper()
    result = await db.execute(
        select(PromoCode).where(PromoCode.code == code_upper, PromoCode.is_active == True)
    )
    promo = result.scalar_one_or_none()

    if not promo:
        return {"valid": False, "message": "Invalid promo code", "discount_amount": 0, "final_amount": amount_inr, "discount_label": "", "promo": None}

    now = datetime.now(timezone.utc)
    if promo.valid_from and now < promo.valid_from:
        return {"valid": False, "message": "Promo code is not yet active", "discount_amount": 0, "final_amount": amount_inr, "discount_label": "", "promo": None}
    if promo.valid_until and now > promo.valid_until:
        return {"valid": False, "message": "Promo code has expired", "discount_amount": 0, "final_amount": amount_inr, "discount_label": "", "promo": None}

    # Check applies_to: "plan" for plan_upgrade, "addon" for point_pack
    expected = "plan" if item_type == "plan_upgrade" else "addon"
    if promo.applies_to != expected:
        return {"valid": False, "message": f"This promo is for {'plans' if promo.applies_to == 'plan' else 'add-ons'} only", "discount_amount": 0, "final_amount": amount_inr, "discount_label": "", "promo": None}

    # Check applicable items restriction
    if promo.applicable_items:
        allowed = [x.strip() for x in promo.applicable_items.split(",")]
        if item_id not in allowed:
            return {"valid": False, "message": "This promo code is not valid for this item", "discount_amount": 0, "final_amount": amount_inr, "discount_label": "", "promo": None}

    # Global usage limit
    if promo.used_count >= promo.max_uses:
        return {"valid": False, "message": "Promo code usage limit reached", "discount_amount": 0, "final_amount": amount_inr, "discount_label": "", "promo": None}

    # Per-user limit
    usage_result = await db.execute(
        select(PromoUsage).where(
            PromoUsage.promo_id == promo.id,
            PromoUsage.user_id == user_id,
        )
    )
    user_uses = len(usage_result.scalars().all())
    if user_uses >= promo.per_user_limit:
        return {"valid": False, "message": "You have already used this promo code", "discount_amount": 0, "final_amount": amount_inr, "discount_label": "", "promo": None}

    # Calculate discount
    if promo.discount_type == "percentage":
        discount = int(amount_inr * promo.discount_value / 100)
        label = f"{int(promo.discount_value)}% off"
    else:
        discount = int(min(promo.discount_value, amount_inr))
        label = f"₹{int(promo.discount_value)} off"

    final = max(amount_inr - discount, 0)
    return {
        "valid": True,
        "message": f"Promo applied! {label}",
        "discount_amount": discount,
        "final_amount": final,
        "discount_label": label,
        "promo": promo,
    }


@router.post("/validate-promo", response_model=PromoValidateResponse)
async def validate_promo(payload: PromoValidateRequest, current_user: CurrentUser, db: DBSession):
    """Validate a promo code and return the discount details."""
    result = await _validate_promo(
        payload.code, payload.item_type, payload.item_id,
        payload.amount_inr, current_user.id, db,
    )
    return PromoValidateResponse(
        valid=result["valid"],
        message=result["message"],
        discount_amount=result["discount_amount"],
        final_amount=result["final_amount"],
        discount_label=result["discount_label"],
    )


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(payload: CheckoutRequest, current_user: CurrentUser, db: DBSession):
    """Create a checkout session for PhonePe or Stripe."""
    # PhonePe merchantOrderId max = 63 chars. Use short UUID hex (no dashes) to save space.
    short_uid = str(current_user.id).replace("-", "")[:16]
    ts = int(datetime.now(timezone.utc).timestamp())
    merchant_order_id = f"{short_uid}_{payload.item_id}_{ts}"

    # ── Validate & apply promo code if provided ──
    actual_amount = payload.amount_inr
    promo_code_str = ""
    promo_discount = 0
    promo_label = ""
    if payload.promo_code:
        promo_result = await _validate_promo(
            payload.promo_code, payload.item_type, payload.item_id,
            payload.amount_inr, current_user.id, db,
        )
        if not promo_result["valid"]:
            raise HTTPException(status_code=400, detail=promo_result["message"])
        actual_amount = promo_result["final_amount"]
        promo_code_str = payload.promo_code.strip().upper()
        promo_discount = promo_result["discount_amount"]
        promo_label = promo_result["discount_label"]

        # Record promo usage + increment global count
        promo_obj = promo_result["promo"]
        promo_obj.used_count += 1
        db.add(PromoUsage(promo_id=promo_obj.id, user_id=current_user.id))
        await db.flush()

    if payload.gateway == "phonepe":
        if not settings.PHONEPE_CLIENT_ID:
            raise HTTPException(status_code=500, detail="Payment gateway is currently unavailable. Please contact support.")

        # Build redirect URL that includes item metadata for fulfilment
        # Only merchant_order_id needs to travel through the redirect URL —
        # every other piece of context (item_type, item_id, org_id, promo)
        # is already stored server-side in the PaymentIntent row created
        # below, keyed by merchant_order_id, so it doesn't need to round-trip
        # through the browser's URL at all. This matches the pattern already
        # proven to work in the legacy system: a single static redirect URL,
        # with transaction context recovered from a local table by
        # transaction ID rather than passed through query params. Also keeps
        # the redirect URL short and simple, in case PhonePe's redirect
        # mechanism handles long/complex URLs poorly.
        redirect_url = f"{payload.success_url}&merchant_order_id={url_quote(merchant_order_id)}"

        checkout_url = ""

        phonepe_sub_id = None  # Track subscription ID for recurring mandate

        # For plan upgrades, try PhonePe recurring subscription (UPI AutoPay) first
        # Note: Recurring mandate uses original price (promo is first-month only)
        if payload.item_type == "plan_upgrade":
            sub_result = await _create_phonepe_subscription(
                amount_inr=PLAN_PRICES.get(payload.item_id, payload.amount_inr),
                merchant_subscription_id=merchant_order_id,
                merchant_user_id=str(current_user.id),
                redirect_url=redirect_url,
            )
            if not sub_result.get("fallback"):
                checkout_url = (
                    sub_result.get("redirectUrl")
                    or sub_result.get("data", {}).get("redirectUrl", "")
                )
                if checkout_url:
                    # Store the subscription ID so we can link it after payment
                    phonepe_sub_id = merchant_order_id

        # Fallback to one-time payment if recurring not available or for point packs
        if not checkout_url:
            order = await _create_phonepe_order(
                amount_inr=actual_amount,
                merchant_order_id=merchant_order_id,
                redirect_url=redirect_url,
                user_id=current_user.id,
                item_type=payload.item_type,
                item_id=payload.item_id,
                org_id=payload.org_id or "",
                promo_code=promo_code_str,
                promo_discount=promo_discount,
            )
            checkout_url = order.get("redirectUrl") or order.get("data", {}).get("redirectUrl", "")
            if not checkout_url:
                checkout_url = order.get("data", {}).get("instrumentResponse", {}).get("redirectInfo", {}).get("url", "")

        # Persist the checkout context locally BEFORE the user ever reaches
        # PhonePe's page — this is the record that makes it possible to
        # confirm and fulfil the payment later even if the success redirect
        # and the webhook both fail to reach us (see PaymentIntent docstring).
        # Never let a failure here (e.g. a duplicate merchant_order_id from a
        # double-submitted request within the same second) block returning
        # the checkout_url PhonePe already issued — the user being able to
        # pay matters more than this bookkeeping row.
        if checkout_url:
            try:
                db.add(PaymentIntent(
                    merchant_order_id=merchant_order_id,
                    user_id=current_user.id,
                    org_id=uuid.UUID(payload.org_id) if payload.org_id else None,
                    gateway="phonepe",
                    flow="subscription" if phonepe_sub_id else "order",
                    item_type=payload.item_type,
                    item_id=payload.item_id,
                    amount_inr=actual_amount,
                    promo_code=promo_code_str or None,
                    promo_discount=promo_discount,
                ))
                await db.commit()
            except Exception as e:
                await db.rollback()
                print(f"[Payment] Failed to persist PaymentIntent for {merchant_order_id}: {e}", flush=True)

        return CheckoutResponse(
            session_id=merchant_order_id,
            redirect_url=checkout_url,
            gateway="phonepe",
        )

    elif payload.gateway == "stripe":
        if not settings.STRIPE_API_KEY:
            raise HTTPException(status_code=500, detail="Payment gateway is currently unavailable. Please contact support.")

        # Get or create Stripe customer for recurring billing
        customer_id = await _get_or_create_stripe_customer(
            current_user.id,
            current_user.email,
            current_user.name or "User",
            db,
        )

        # Cancel any existing Stripe subscription if upgrading plan
        if payload.item_type == "plan_upgrade":
            result = await db.execute(
                select(Subscription).where(
                    Subscription.user_id == current_user.id,
                    Subscription.org_id.is_(None) if not payload.org_id
                    else Subscription.org_id == uuid.UUID(payload.org_id),
                )
            )
            existing_sub = result.scalar_one_or_none()
            if existing_sub:
                await _cancel_existing_stripe_subscription(existing_sub)

        if payload.item_type == "plan_upgrade":
            # For plans with promo: use one-time payment mode (promo applies first month only,
            # next month renews at full price via webhook). Without promo: use subscription mode.
            plan_product_name = payload.item_id.replace("_", " ").title()
            stripe_metadata = {
                "item_type": payload.item_type,
                "item_id": payload.item_id,
                "org_id": payload.org_id or "",
                "user_id": str(current_user.id),
                "promo_code": promo_code_str,
                "promo_discount": str(promo_discount),
                "original_amount": str(payload.amount_inr),
            }

            if promo_code_str:
                # Promo applied: first month is discounted (one-time), then create
                # a regular subscription at full price starting next month
                session = stripe.checkout.Session.create(
                    customer=customer_id,
                    payment_method_types=["card"],
                    line_items=[{
                        "price_data": {
                            "currency": "inr",
                            "product_data": {"name": f"{plan_product_name} (First Month — {promo_label})"},
                            "unit_amount": actual_amount * 100,
                        },
                        "quantity": 1,
                    }],
                    mode="payment",
                    success_url=f"{payload.success_url}&session_id={{CHECKOUT_SESSION_ID}}&promo_code={promo_code_str}&promo_discount={promo_discount}",
                    cancel_url=payload.cancel_url,
                    client_reference_id=str(current_user.id),
                    metadata=stripe_metadata,
                )
            else:
                # No promo: use subscription mode for auto-renewal at full price
                session = stripe.checkout.Session.create(
                    customer=customer_id,
                    payment_method_types=["card"],
                    line_items=[{
                        "price_data": {
                            "currency": "inr",
                            "product_data": {"name": plan_product_name},
                            "unit_amount": payload.amount_inr * 100,
                            "recurring": {"interval": "month"},
                        },
                        "quantity": 1,
                    }],
                    mode="subscription",
                    success_url=f"{payload.success_url}&session_id={{CHECKOUT_SESSION_ID}}",
                    cancel_url=payload.cancel_url,
                    client_reference_id=str(current_user.id),
                    metadata=stripe_metadata,
                    subscription_data={"metadata": stripe_metadata},
                )
        else:
            # One-time payment for point packs (promo or not)
            pack_name = payload.item_id.replace("_", " ").title()
            if promo_code_str:
                pack_name = f"{pack_name} ({promo_label})"
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price_data": {
                        "currency": "inr",
                        "product_data": {"name": pack_name},
                        "unit_amount": actual_amount * 100,
                    },
                    "quantity": 1,
                }],
                mode="payment",
                success_url=f"{payload.success_url}&session_id={{CHECKOUT_SESSION_ID}}&promo_code={promo_code_str}&promo_discount={promo_discount}",
                cancel_url=payload.cancel_url,
                client_reference_id=str(current_user.id),
                metadata={
                    "item_type": payload.item_type,
                    "item_id": payload.item_id,
                    "org_id": payload.org_id or "",
                    "user_id": str(current_user.id),
                    "promo_code": promo_code_str,
                    "promo_discount": str(promo_discount),
                    "original_amount": str(payload.amount_inr),
                },
            )

        return CheckoutResponse(
            session_id=session.id,
            redirect_url=session.url,
            gateway="stripe",
        )

    else:
        raise HTTPException(status_code=400, detail="Unsupported payment method. Please select PhonePe or Stripe.")


@router.get("/phonepe/status/{merchant_order_id}", response_model=PaymentStatusResponse)
async def check_phonepe_status(
    merchant_order_id: str,
    item_type: str = "",
    item_id: str = "",
    org_id: str = "",
    promo_code: str = "",
    promo_discount: int = 0,
    current_user: CurrentUser = None,
    db: DBSession = None,
    background_tasks: BackgroundTasks = None,
):
    """Check PhonePe payment status (order OR recurring subscription flow) and fulfil if successful.

    item_type/item_id/org_id/promo_code/promo_discount are now optional —
    the redirect URL no longer carries them (see create_checkout), since
    PaymentIntent already stores all of this server-side, keyed by
    merchant_order_id. They're only used as a fallback for orders created
    before PaymentIntent existed, where an old-format redirect URL might
    still land here with these as query params.
    """
    intent_result = await db.execute(
        select(PaymentIntent).where(PaymentIntent.merchant_order_id == merchant_order_id)
    )
    intent = intent_result.scalar_one_or_none()

    if intent:
        item_type = intent.item_type
        item_id = intent.item_id
        org_id = str(intent.org_id) if intent.org_id else ""
        promo_code = intent.promo_code or ""
        promo_discount = intent.promo_discount

    if not item_type or not item_id:
        raise HTTPException(
            status_code=400,
            detail="This payment link is missing required information — if you were charged, please contact support.",
        )

    check = await _check_phonepe_payment_status(merchant_order_id, flow=intent.flow if intent else None)
    normalized_state = check["normalized_state"]

    if normalized_state == "COMPLETED":
        # If a local intent exists, claim it atomically before fulfilling —
        # avoids double-crediting if the webhook confirms this same payment
        # at nearly the same moment (see _try_claim_payment_intent).
        if intent:
            won_claim = await _try_claim_payment_intent(db, merchant_order_id)
            if not won_claim:
                await db.refresh(intent)
                if intent.status == "fulfilled":
                    pass  # someone else (the webhook) already fulfilled it — report success below
                else:
                    # Being claimed/fulfilled by another request right now, or already failed.
                    return PaymentStatusResponse(status="pending", message="Payment is being processed")
            else:
                try:
                    await _fulfil_purchase(
                        item_type,
                        item_id,
                        current_user.id,
                        org_id if org_id else None,
                        db,
                        merchant_order_id=merchant_order_id,
                        gateway="phonepe",
                        phonepe_subscription_id=merchant_order_id if item_type == "plan_upgrade" else None,
                        promo_code=promo_code if promo_code else None,
                        promo_discount=promo_discount,
                        background_tasks=background_tasks,
                        amount_inr=intent.amount_inr,
                    )
                    intent.status = "fulfilled"
                    await db.commit()
                except Exception:
                    await _release_payment_intent_claim(db, merchant_order_id)
                    raise
        else:
            # No local record (pre-existing order) — fall back to the
            # original time-window dedup inside _fulfil_purchase.
            await _fulfil_purchase(
                item_type,
                item_id,
                current_user.id,
                org_id if org_id else None,
                db,
                merchant_order_id=merchant_order_id,
                gateway="phonepe",
                phonepe_subscription_id=merchant_order_id if item_type == "plan_upgrade" else None,
                promo_code=promo_code if promo_code else None,
                promo_discount=promo_discount,
                background_tasks=background_tasks,
            )

        if item_type == "plan_upgrade":
            plan_result = await db.execute(
                select(PlanDefinition).where(PlanDefinition.plan == item_id)
            )
            plan_def = plan_result.scalar_one_or_none()
            display_name = plan_def.display_name if plan_def else item_id
            return PaymentStatusResponse(
                status="success",
                message=f"Plan upgraded to {display_name}!",
                plan=item_id,
            )
        else:
            addon_info = ADDON_POINT_MAP.get(item_id, {})
            return PaymentStatusResponse(
                status="success",
                message=f"Added {addon_info.get('points', 0)} points",
                points_added=addon_info.get("points", 0),
            )

    elif normalized_state == "PENDING":
        return PaymentStatusResponse(status="pending", message="Payment is being processed")
    else:
        if intent and intent.status == "pending":
            intent.status = "failed"
            await db.commit()
        raw_state = check["raw"].get("state") or check["raw"].get("data", {}).get("state", "unknown")
        return PaymentStatusResponse(status="failed", message=f"Payment was not successful (state: {raw_state})")


@router.get("/status/{session_id}", response_model=PaymentStatusResponse)
async def get_payment_status(session_id: str, current_user: CurrentUser, db: DBSession, background_tasks: BackgroundTasks = None):
    """Check Stripe payment status and fulfil if successful."""
    if not session_id.startswith("cs_"):
        return PaymentStatusResponse(status="pending", message="Use /phonepe/status for PhonePe payments")

    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Payment session not found")

    if session.payment_status == "paid":
        meta = session.metadata or {}
        item_type = meta.get("item_type", "plan_upgrade")
        item_id = meta.get("item_id", "")
        org_id = meta.get("org_id") or None

        # Extract Stripe subscription ID if this was a subscription checkout
        stripe_sub_id = getattr(session, "subscription", None)
        customer_id = getattr(session, "customer", None)

        # Promo info from metadata
        s_promo_code = meta.get("promo_code", "") or ""
        s_promo_discount = int(meta.get("promo_discount", "0") or "0")

        sub = await _fulfil_purchase(
            item_type, item_id, current_user.id, org_id, db,
            merchant_order_id=session_id,
            gateway="stripe",
            stripe_customer_id=customer_id,
            stripe_subscription_id=stripe_sub_id,
            promo_code=s_promo_code if s_promo_code else None,
            promo_discount=s_promo_discount,
            background_tasks=background_tasks,
        )

        if item_type == "plan_upgrade":
            plan_result = await db.execute(
                select(PlanDefinition).where(PlanDefinition.plan == item_id)
            )
            plan_def = plan_result.scalar_one_or_none()
            display_name = plan_def.display_name if plan_def else item_id
            return PaymentStatusResponse(
                status="success", message=f"Plan upgraded to {display_name}!", plan=item_id
            )
        else:
            addon_info = ADDON_POINT_MAP.get(item_id, {})
            return PaymentStatusResponse(
                status="success",
                message=f"Added {addon_info.get('points', 0)} points",
                points_added=addon_info.get("points", 0),
            )

    elif session.payment_status == "unpaid":
        return PaymentStatusResponse(status="pending", message="Payment not yet completed")
    else:
        return PaymentStatusResponse(status="failed", message="Payment was not successful")


# ---------------------------------------------------------------------------
# Stripe Webhook — handles recurring subscription renewals
# ---------------------------------------------------------------------------

@router.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: DBSession):
    """Handle Stripe webhook events for subscription renewals."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if settings.STRIPE_ENDPOINT_SECRET:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.STRIPE_ENDPOINT_SECRET,
            )
        except (ValueError, stripe.error.SignatureVerificationError):
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
    else:
        # Dev mode — no signature verification
        import json
        event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)

    event_type = event["type"]
    print(f"[Stripe Webhook] Received event: {event_type}", flush=True)

    if event_type == "invoice.paid":
        invoice = event["data"]["object"]
        subscription_id = invoice.get("subscription")
        customer_id = invoice.get("customer")

        # Skip the first invoice (initial purchase already handled by checkout flow)
        billing_reason = invoice.get("billing_reason", "")
        if billing_reason == "subscription_create":
            print("[Stripe Webhook] Skipping initial subscription invoice", flush=True)
            return {"status": "ok"}

        # This is a renewal invoice
        if subscription_id:
            stripe_sub = stripe.Subscription.retrieve(subscription_id)
            meta = stripe_sub.get("metadata", {})
            item_type = meta.get("item_type", "plan_upgrade")
            item_id = meta.get("item_id", "")
            user_id_str = meta.get("user_id", "")
            org_id = meta.get("org_id") or None

            if user_id_str and item_id:
                user_id = uuid.UUID(user_id_str)
                print(f"[Stripe Webhook] Processing renewal for user {user_id}, plan {item_id}", flush=True)

                await _fulfil_purchase(
                    item_type=item_type,
                    item_id=item_id,
                    user_id=user_id,
                    org_id=org_id,
                    db=db,
                    merchant_order_id=f"stripe_renewal_{invoice.get('id', '')}",
                    gateway="stripe",
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                    is_renewal=True,
                )
                from app.services.notification_service import create_notification
                from app.models.notification import NotificationType
                await create_notification(
                    db, user_id=user_id,
                    notification_type=NotificationType.PAYMENT_SUCCESS,
                    title="Payment Successful",
                    body=f"Your {item_id.replace('_', ' ').title()} plan has been renewed successfully.",
                    icon="check-circle-2", priority="normal",
                    data_json={"plan": item_id, "link": "/u/plans"},
                )
                await db.commit()

    elif event_type == "customer.subscription.deleted":
        # Subscription cancelled or expired
        sub_data = event["data"]["object"]
        meta = sub_data.get("metadata", {})
        user_id_str = meta.get("user_id", "")

        if user_id_str:
            user_id = uuid.UUID(user_id_str)
            result = await db.execute(
                select(Subscription).where(
                    Subscription.user_id == user_id,
                    Subscription.stripe_subscription_id == sub_data.get("id"),
                )
            )
            sub = result.scalar_one_or_none()
            if sub:
                sub.auto_renew = False
                sub.stripe_subscription_id = None
                print(f"[Stripe Webhook] Disabled auto-renew for user {user_id}", flush=True)
                await db.commit()

    elif event_type == "invoice.payment_failed":
        invoice = event["data"]["object"]
        subscription_id = invoice.get("subscription")
        if subscription_id:
            stripe_sub = stripe.Subscription.retrieve(subscription_id)
            meta = stripe_sub.get("metadata", {})
            user_id_str = meta.get("user_id", "")
            if user_id_str:
                user_id = uuid.UUID(user_id_str)
                # Record a failed renewal transaction for visibility
                result = await db.execute(
                    select(Subscription).where(Subscription.user_id == user_id)
                )
                sub = result.scalar_one_or_none()
                if sub:
                    txn = PointTransaction(
                        subscription_id=sub.id,
                        user_id=user_id,
                        action=f"renewal_failed_{sub.plan}",
                        points_used=0,
                        balance_after=sub.points_balance,
                    )
                    db.add(txn)
                    from app.services.notification_service import create_notification
                    from app.models.notification import NotificationType
                    await create_notification(
                        db, user_id=user_id,
                        notification_type=NotificationType.PAYMENT_FAILURE,
                        title="Payment Failed",
                        body=f"Your plan renewal payment failed. Please update your payment method to avoid service interruption.",
                        icon="alert-circle", priority="high",
                        data_json={"plan": sub.plan, "link": "/u/plans"},
                    )
                    await db.commit()
                    print(f"[Stripe Webhook] Recorded renewal failure for user {user_id}", flush=True)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# PhonePe Recurring Debit Webhook — handles auto-renewal notifications
# ---------------------------------------------------------------------------

@router.post("/phonepe/recurring-webhook")
async def phonepe_recurring_webhook(request: Request, db: DBSession):
    """Handle PhonePe recurring debit notification callbacks.

    PhonePe sends a POST to the notifyUrl after a recurring debit is executed.
    This fulfils the renewal if payment was successful.
    """
    body = await request.json()
    print(f"[PhonePe Recurring Webhook] Received: {body}", flush=True)

    # PhonePe recurring debit notification structure
    txn_id = body.get("transactionId", "")
    merchant_sub_id = body.get("merchantSubscriptionId", "")
    state = body.get("state", "") or body.get("data", {}).get("state", "")

    if not merchant_sub_id:
        return {"status": "ignored", "reason": "no merchantSubscriptionId"}

    # Find the subscription linked to this PhonePe mandate
    result = await db.execute(
        select(Subscription).where(
            Subscription.phonepe_subscription_id == merchant_sub_id,
            Subscription.payment_gateway == "phonepe",
        )
    )
    sub = result.scalar_one_or_none()
    if not sub:
        print(f"[PhonePe Recurring Webhook] No subscription found for {merchant_sub_id}", flush=True)
        return {"status": "ignored", "reason": "subscription not found"}

    if state in ("COMPLETED", "SUCCESS"):
        print(f"[PhonePe Recurring Webhook] Renewal success for sub {sub.id}, plan {sub.plan}", flush=True)
        await _fulfil_purchase(
            item_type="plan_upgrade",
            item_id=sub.plan,
            user_id=sub.user_id,
            org_id=str(sub.org_id) if sub.org_id else None,
            db=db,
            merchant_order_id=txn_id or f"phonepe_renewal_{sub.id}_{int(datetime.now(timezone.utc).timestamp())}",
            gateway="phonepe",
            phonepe_subscription_id=merchant_sub_id,
            is_renewal=True,
        )
        from app.services.notification_service import create_notification
        from app.models.notification import NotificationType
        await create_notification(
            db, user_id=sub.user_id,
            notification_type=NotificationType.PAYMENT_SUCCESS,
            title="Payment Successful",
            body=f"Your {sub.plan.replace('_', ' ').title()} plan has been renewed successfully.",
            icon="check-circle-2",
            data_json={"plan": sub.plan, "link": "/u/plans"},
        )
        await db.commit()
        return {"status": "ok"}

    elif state in ("FAILED", "DECLINED"):
        print(f"[PhonePe Recurring Webhook] Renewal failed for sub {sub.id}", flush=True)
        txn = PointTransaction(
            subscription_id=sub.id,
            user_id=sub.user_id,
            action=f"renewal_failed_{sub.plan}",
            points_used=0,
            balance_after=sub.points_balance,
        )
        db.add(txn)
        from app.services.notification_service import create_notification as _create_notif
        from app.models.notification import NotificationType as _NT
        await _create_notif(
            db, user_id=sub.user_id,
            notification_type=_NT.PAYMENT_FAILURE,
            title="Payment Failed",
            body=f"Your plan renewal payment failed. Please update your payment method.",
            icon="alert-circle", priority="high",
            data_json={"plan": sub.plan, "link": "/u/plans"},
        )
        await db.commit()
        return {"status": "ok"}

    return {"status": "pending"}


# ---------------------------------------------------------------------------
# PhonePe Standard Checkout Webhook — server-to-server payment notification
# ---------------------------------------------------------------------------

@router.post("/phonepe/webhook")
async def phonepe_standard_webhook(request: Request, db: DBSession, background_tasks: BackgroundTasks):
    """Server-to-server webhook for PhonePe Standard Checkout ONE-TIME ORDERS
    ONLY (events: checkout.order.completed / checkout.order.failed).

    IMPORTANT: PhonePe uses a COMPLETELY SEPARATE webhook system for Autopay
    (recurring UPI mandate) events — this URL is never called for those. The
    first payment on a recurring-mandate checkout (see _create_phonepe_subscription
    — tried first for every plan upgrade) is confirmed via the frontend's
    /phonepe/status/{id} redirect callback and, as a safety net, the
    payment_reconciliation background sweep — NOT via this webhook.

    Configure this URL in the PhonePe merchant dashboard:
        https://app.genverse.ai/api/v1/payments/phonepe/webhook

    PhonePe calls this endpoint when a one-time order completes. The handler
    verifies the payment directly with PhonePe (does not trust the webhook
    body's state) and fulfils the purchase, so the DB is updated even if the
    user's browser never loads the success page. _fulfil_purchase is
    idempotent, so a duplicate call is safe.

    Context (user/plan/promo) is read from our own local PaymentIntent record
    (written at checkout time), not from PhonePe's response, since that's
    more reliable than round-tripping through echoed metaInfo. Falls back to
    metaInfo only for orders created before PaymentIntent existed.
    """
    raw_body = await request.body()

    # Optional HMAC-style Basic Auth check PhonePe recommends: Authorization
    # header = SHA256(username:password), configured in the merchant
    # dashboard alongside the webhook URL. Only enforced once configured
    # here (PHONEPE_WEBHOOK_USERNAME/PASSWORD) — matches the same
    # opt-in pattern used for STRIPE_ENDPOINT_SECRET above, so this doesn't
    # start rejecting real webhook calls before the credentials are set.
    if settings.PHONEPE_WEBHOOK_USERNAME and settings.PHONEPE_WEBHOOK_PASSWORD:
        expected = hashlib.sha256(
            f"{settings.PHONEPE_WEBHOOK_USERNAME}:{settings.PHONEPE_WEBHOOK_PASSWORD}".encode()
        ).hexdigest()
        received = request.headers.get("authorization", "")
        if not hmac.compare_digest(expected, received):
            print("[PhonePe Webhook] Authorization header mismatch, rejecting", flush=True)
            raise HTTPException(status_code=401, detail="Invalid webhook authorization")

    import json
    body = json.loads(raw_body)
    print(f"[PhonePe Webhook] Received: {body}", flush=True)

    # PhonePe nests the actual order data under "payload" (top-level has
    # "event" + "payload"). Fall back to top-level for safety in case the
    # exact shape ever differs from the documented one.
    payload = body.get("payload", {}) or {}
    merchant_order_id = payload.get("merchantOrderId") or body.get("merchantOrderId", "")
    if not merchant_order_id:
        print(f"[PhonePe Webhook] No merchantOrderId found in payload: {body}", flush=True)
        return {"status": "ignored", "reason": "no merchantOrderId"}

    intent_result = await db.execute(
        select(PaymentIntent).where(PaymentIntent.merchant_order_id == merchant_order_id)
    )
    intent = intent_result.scalar_one_or_none()

    # Verify payment state directly with PhonePe — do not trust webhook body alone.
    # Checks the right API (order vs subscription) based on how this checkout
    # was actually created, instead of always assuming a one-time order.
    check = await _check_phonepe_payment_status(merchant_order_id, flow=intent.flow if intent else None)
    normalized_state = check["normalized_state"]
    if normalized_state != "COMPLETED":
        print(f"[PhonePe Webhook] Order {merchant_order_id} flow={check['flow']} state={normalized_state}, skipping", flush=True)
        if normalized_state == "FAILED" and intent and intent.status == "pending":
            intent.status = "failed"
            await db.commit()
        return {"status": "pending"}

    if intent:
        user_id = intent.user_id
        item_type = intent.item_type
        item_id = intent.item_id
        org_id = str(intent.org_id) if intent.org_id else None
        promo_code = intent.promo_code
        promo_discount = intent.promo_discount
    else:
        # Fallback for orders created before PaymentIntent existed — only
        # ever populated for the one-time order flow (see docstring above).
        meta = check["raw"].get("metaInfo", {})
        user_id_str = meta.get("udf1", "")
        item_type   = meta.get("udf2", "")
        item_id     = meta.get("udf3", "")
        org_id      = meta.get("udf4", "") or None
        promo_raw   = meta.get("udf5", "|0").split("|", 1)
        promo_code  = promo_raw[0] or None
        promo_discount = int(promo_raw[1]) if len(promo_raw) > 1 and promo_raw[1].isdigit() else 0

        if not user_id_str or not item_type or not item_id:
            print(f"[PhonePe Webhook] No PaymentIntent and no metaInfo context for {merchant_order_id}", flush=True)
            return {"status": "error", "reason": "missing order context"}

        try:
            user_id = uuid.UUID(user_id_str)
        except ValueError:
            print(f"[PhonePe Webhook] Invalid user_id '{user_id_str}' for {merchant_order_id}", flush=True)
            return {"status": "error", "reason": "invalid user_id"}

    # Prefer the amount PhonePe actually reports for this completed payment
    # (payload.amount is in paise) over our own stored estimate — this is
    # the most authoritative source of what was really charged.
    webhook_amount_paise = check["raw"].get("amount") or payload.get("amount")
    amount_inr = (
        webhook_amount_paise // 100 if isinstance(webhook_amount_paise, int)
        else (intent.amount_inr if intent else None)
    )

    # If a local intent exists, claim it atomically before fulfilling — the
    # frontend's /phonepe/status/{id} redirect check can confirm and fulfil
    # this exact same payment at nearly the same moment; without this claim
    # both could pass _fulfil_purchase's time-window dedup check before
    # either commits, double-crediting the user.
    if intent:
        won_claim = await _try_claim_payment_intent(db, merchant_order_id)
        if not won_claim:
            print(f"[PhonePe Webhook] {merchant_order_id} already claimed elsewhere, skipping", flush=True)
            return {"status": "ok"}
        try:
            print(f"[PhonePe Webhook] Fulfilling {item_type}/{item_id} for user {user_id}", flush=True)
            await _fulfil_purchase(
                item_type=item_type,
                item_id=item_id,
                user_id=user_id,
                org_id=org_id,
                db=db,
                merchant_order_id=merchant_order_id,
                gateway="phonepe",
                phonepe_subscription_id=merchant_order_id if check["flow"] == "subscription" else None,
                promo_code=promo_code,
                promo_discount=promo_discount,
                background_tasks=background_tasks,
                amount_inr=amount_inr,
            )
            intent.status = "fulfilled"
            await db.commit()
        except Exception:
            await _release_payment_intent_claim(db, merchant_order_id)
            raise
    else:
        # No local record (pre-existing order) — fall back to the original
        # time-window dedup inside _fulfil_purchase.
        print(f"[PhonePe Webhook] Fulfilling {item_type}/{item_id} for user {user_id}", flush=True)
        await _fulfil_purchase(
            item_type=item_type,
            item_id=item_id,
            user_id=user_id,
            org_id=org_id,
            db=db,
            merchant_order_id=merchant_order_id,
            gateway="phonepe",
            phonepe_subscription_id=merchant_order_id if check["flow"] == "subscription" else None,
            promo_code=promo_code,
            promo_discount=promo_discount,
            background_tasks=background_tasks,
            amount_inr=amount_inr,
        )

    return {"status": "ok"}

