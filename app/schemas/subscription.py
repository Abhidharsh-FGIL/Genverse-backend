import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class SubscriptionResponse(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    org_id: Optional[uuid.UUID] = None
    plan: str
    status: str
    workspace_type: str
    points_balance: int
    points_monthly_quota: int
    storage_limit_mb: int
    max_seats: Optional[int] = None
    trial_ends_at: Optional[datetime] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PlanDefinitionResponse(BaseModel):
    id: uuid.UUID
    plan: str
    display_name: str
    price_inr: float
    workspace_type: str
    monthly_points: int
    storage_mb: int
    max_seats: Optional[int] = None
    description: Optional[str] = None

    model_config = {"from_attributes": True}


class UpgradePlanRequest(BaseModel):
    plan: str


class PointDeductRequest(BaseModel):
    action: str
    subscription_id: Optional[str] = None


class PointDeductResponse(BaseModel):
    success: bool
    points_used: int
    remaining_balance: int
    action: str


class PointTransactionResponse(BaseModel):
    id: uuid.UUID
    action: str
    points_used: int
    balance_after: int
    created_at: datetime

    model_config = {"from_attributes": True}


class BuyAddonRequest(BaseModel):
    addon_type: str  # point_pack_100 | point_pack_500 | point_pack_1000
    quantity: int = 1


class FeatureLimitResponse(BaseModel):
    feature_key: str
    enabled: bool
    daily_limit: Optional[int] = None
    monthly_limit: Optional[int] = None
    notes: Optional[str] = None

    model_config = {"from_attributes": True}


class UsageCounterResponse(BaseModel):
    feature_key: str
    period: str
    period_start: datetime
    count: int

    model_config = {"from_attributes": True}


class AddonResponse(BaseModel):
    id: uuid.UUID
    subscription_id: uuid.UUID
    addon_type: str
    quantity: int
    purchased_at: datetime
    expires_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
