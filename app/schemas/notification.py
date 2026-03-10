from datetime import datetime
from uuid import UUID
from typing import Any, Optional
from pydantic import BaseModel


class NotificationResponse(BaseModel):
    id: UUID
    notification_type: str
    category: str
    title: str
    body: str
    data_json: Optional[Any] = None
    icon: Optional[str] = None
    priority: str
    is_read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class NotificationListResponse(BaseModel):
    total: int
    unread_count: int
    items: list[NotificationResponse]


class UnreadCountResponse(BaseModel):
    count: int
