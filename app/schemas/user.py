import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional, List


class UserBase(BaseModel):
    name: str
    email: EmailStr
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    grade: Optional[int] = None
    persona_band: Optional[str] = None
    language: str = "en"
    subjects: Optional[List[str]] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    grade: Optional[int] = None
    persona_band: Optional[str] = None
    language: Optional[str] = None
    subjects: Optional[List[str]] = None


class UserResponse(UserBase):
    id: uuid.UUID
    role: Optional[str] = None
    xp: int
    streak: int
    is_active: bool
    last_login_date: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserRoleResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AiContextRequest(BaseModel):
    workspace_id: str = "personal"
    grade: Optional[int] = None
    board: Optional[str] = None
    subject: Optional[str] = None
    language: str = "en"
    tone: str = "helpful"
    difficulty: str = "medium"
    output_mode: str = "text"
    student_mode: bool = False


class AiContextResponse(AiContextRequest):
    id: uuid.UUID
    user_id: uuid.UUID
    updated_at: datetime

    model_config = {"from_attributes": True}
