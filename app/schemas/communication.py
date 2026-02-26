import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Any


class GroupChatCreate(BaseModel):
    name: str
    description: Optional[str] = None
    class_id: Optional[str] = None
    org_id: Optional[str] = None


class GroupChatResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    class_id: Optional[uuid.UUID] = None
    org_id: Optional[uuid.UUID] = None
    creator_id: uuid.UUID
    is_active: bool
    created_at: datetime
    unread_count: int = 0

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    user_id: uuid.UUID
    content: str
    attachments: Optional[Any] = None
    is_deleted: bool
    created_at: datetime
    sender_name: Optional[str] = None
    sender_avatar: Optional[str] = None

    model_config = {"from_attributes": True}


class ReadReceiptUpdate(BaseModel):
    chat_id: str
