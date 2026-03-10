"""
Notification Service — centralised helper for creating in-app notifications
with built-in deduplication.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.notification import Notification

logger = logging.getLogger(__name__)


async def create_notification(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    notification_type: str,
    title: str,
    body: str,
    category: str = "system",
    org_id: uuid.UUID | None = None,
    data_json: dict | None = None,
    icon: str | None = None,
    priority: str = "normal",
    expires_at: datetime | None = None,
    dedup_hours: int = 24,
) -> Notification | None:
    """
    Create a notification for a user.

    If a notification of the same type already exists for this user
    within ``dedup_hours``, the call is silently skipped (returns None).
    Set ``dedup_hours=0`` to disable deduplication.
    """
    try:
        # Dedup check
        if dedup_hours > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=dedup_hours)
            existing = await db.execute(
                select(Notification.id).where(
                    and_(
                        Notification.user_id == user_id,
                        Notification.notification_type == notification_type,
                        Notification.created_at > cutoff,
                        Notification.is_dismissed == False,  # noqa: E712
                    )
                ).limit(1)
            )
            if existing.scalar():
                return None

        notif = Notification(
            id=uuid.uuid4(),
            user_id=user_id,
            org_id=org_id,
            notification_type=notification_type,
            category=category,
            title=title,
            body=body,
            data_json=data_json,
            icon=icon,
            priority=priority,
            expires_at=expires_at,
        )
        db.add(notif)
        await db.flush()
        return notif
    except Exception:
        logger.exception("Failed to create notification type=%s for user=%s", notification_type, user_id)
        return None


async def create_notification_for_many(
    db: AsyncSession,
    *,
    user_ids: list[uuid.UUID],
    notification_type: str,
    title: str,
    body: str,
    category: str = "system",
    org_id: uuid.UUID | None = None,
    data_json: dict | None = None,
    icon: str | None = None,
    priority: str = "normal",
    expires_at: datetime | None = None,
    dedup_hours: int = 24,
) -> int:
    """
    Create the same notification for multiple users (e.g. announcement).
    Returns the number of notifications actually created (after dedup).
    """
    created = 0
    for uid in user_ids:
        result = await create_notification(
            db,
            user_id=uid,
            notification_type=notification_type,
            title=title,
            body=body,
            category=category,
            org_id=org_id,
            data_json=data_json,
            icon=icon,
            priority=priority,
            expires_at=expires_at,
            dedup_hours=dedup_hours,
        )
        if result:
            created += 1
    return created
