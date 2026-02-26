import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Any


class BadgeResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    rarity: str
    xp_reward: int

    model_config = {"from_attributes": True}


class StudentBadgeResponse(BaseModel):
    id: uuid.UUID
    badge_id: uuid.UUID
    earned_at: datetime
    badge: BadgeResponse

    model_config = {"from_attributes": True}


class TitleResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    rarity: str

    model_config = {"from_attributes": True}


class StudentTitleResponse(BaseModel):
    id: uuid.UUID
    title_id: uuid.UUID
    is_active: bool
    earned_at: datetime
    title: TitleResponse

    model_config = {"from_attributes": True}


class XPAwardRequest(BaseModel):
    user_id: str
    amount: int
    event: str  # upload_document | complete_assessment | ...


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    user_name: str
    xp: int
    streak: int
    badge_count: int


class GamificationSummary(BaseModel):
    xp: int
    streak: int
    level: int
    next_level_xp: int
    badges_earned: int
    titles_earned: int
