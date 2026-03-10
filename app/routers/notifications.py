"""
Notifications API — list, count, mark-read, dismiss.
"""

from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, func, and_, update

from app.dependencies import DBSession, CurrentUser
from app.models.notification import Notification
from app.schemas.notification import (
    NotificationResponse,
    NotificationListResponse,
    UnreadCountResponse,
)

router = APIRouter()


def _base_filter(user_id: uuid.UUID, org_id: uuid.UUID | None = None):
    """Common WHERE clause: user's non-dismissed notifications, optionally filtered by workspace."""
    clauses = [
        Notification.user_id == user_id,
        Notification.is_dismissed == False,  # noqa: E712
    ]
    if org_id:
        clauses.append(Notification.org_id == org_id)
    return and_(*clauses)


# ── List notifications ──────────────────────────────────────────────────────

@router.get("", response_model=NotificationListResponse)
async def list_notifications(
    current_user: CurrentUser,
    db: DBSession,
    org_id: uuid.UUID | None = Query(None),
    unread_only: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Return paginated notifications for the current user."""
    base = _base_filter(current_user.id, org_id)

    # Total count
    total_q = select(func.count()).select_from(Notification).where(base)
    if unread_only:
        total_q = total_q.where(Notification.is_read == False)  # noqa: E712
    total = (await db.execute(total_q)).scalar() or 0

    # Unread count (always returned regardless of unread_only filter)
    unread_count = (await db.execute(
        select(func.count()).select_from(Notification).where(
            base, Notification.is_read == False  # noqa: E712
        )
    )).scalar() or 0

    # Items
    items_q = (
        select(Notification)
        .where(base)
        .order_by(Notification.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if unread_only:
        items_q = items_q.where(Notification.is_read == False)  # noqa: E712
    items = (await db.execute(items_q)).scalars().all()

    return NotificationListResponse(
        total=total,
        unread_count=unread_count,
        items=[NotificationResponse.model_validate(n) for n in items],
    )


# ── Unread count (lightweight polling endpoint) ─────────────────────────────

@router.get("/unread-count", response_model=UnreadCountResponse)
async def unread_count(
    current_user: CurrentUser,
    db: DBSession,
    org_id: uuid.UUID | None = Query(None),
):
    """Fast count of unread notifications — polled by the frontend bell."""
    count = (await db.execute(
        select(func.count()).select_from(Notification).where(
            _base_filter(current_user.id, org_id),
            Notification.is_read == False,  # noqa: E712
        )
    )).scalar() or 0
    return UnreadCountResponse(count=count)


# ── Mark single notification as read ────────────────────────────────────────

@router.patch("/{notification_id}/read")
async def mark_read(
    notification_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
):
    """Mark a single notification as read."""
    result = await db.execute(
        update(Notification)
        .where(Notification.id == notification_id, Notification.user_id == current_user.id)
        .values(is_read=True)
    )
    if result.rowcount == 0:
        raise HTTPException(404, "Notification not found")
    await db.commit()
    return {"ok": True}


# ── Mark all as read ────────────────────────────────────────────────────────

@router.post("/mark-all-read")
async def mark_all_read(
    current_user: CurrentUser,
    db: DBSession,
    org_id: uuid.UUID | None = Query(None),
):
    """Mark every unread notification as read for the current user."""
    await db.execute(
        update(Notification)
        .where(
            _base_filter(current_user.id, org_id),
            Notification.is_read == False,  # noqa: E712
        )
        .values(is_read=True)
    )
    await db.commit()
    return {"ok": True}


# ── Dismiss (soft-delete) ──────────────────────────────────────────────────

@router.delete("/{notification_id}")
async def dismiss_notification(
    notification_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
):
    """Soft-dismiss a notification (hidden from UI, not deleted)."""
    result = await db.execute(
        update(Notification)
        .where(Notification.id == notification_id, Notification.user_id == current_user.id)
        .values(is_dismissed=True)
    )
    if result.rowcount == 0:
        raise HTTPException(404, "Notification not found")
    await db.commit()
    return {"ok": True}
