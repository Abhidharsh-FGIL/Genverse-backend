import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, Integer, Float, DateTime, Text, func, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base

SUBSCRIPTION_PLAN = SAEnum(
    "free", "individual_pro", "individual_power",
    "org_basic", "org_pro", "org_evaluation",
    name="subscription_plan",
)
SUBSCRIPTION_STATUS = SAEnum("active", "trialing", "expired", "cancelled", "paused", name="subscription_status")
WORKSPACE_TYPE = SAEnum("individual", "organization", name="workspace_type")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    org_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), index=True)
    plan: Mapped[str] = mapped_column(SUBSCRIPTION_PLAN, nullable=False, default="free")
    status: Mapped[str] = mapped_column(SUBSCRIPTION_STATUS, nullable=False, default="trialing")
    workspace_type: Mapped[str] = mapped_column(WORKSPACE_TYPE, nullable=False, default="individual")
    points_balance: Mapped[int] = mapped_column(Integer, default=100)
    points_monthly_quota: Mapped[int] = mapped_column(Integer, default=100)
    storage_limit_mb: Mapped[int] = mapped_column(Integer, default=100)
    max_seats: Mapped[int | None] = mapped_column(Integer)
    trial_ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="subscriptions")  # noqa: F821
    organization: Mapped["Organization"] = relationship(back_populates="subscriptions")  # noqa: F821
    transactions: Mapped[list["PointTransaction"]] = relationship(back_populates="subscription", cascade="all, delete-orphan")
    addons: Mapped[list["SubscriptionAddon"]] = relationship(back_populates="subscription", cascade="all, delete-orphan")


class PlanDefinition(Base):
    __tablename__ = "plan_definitions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan: Mapped[str] = mapped_column(SUBSCRIPTION_PLAN, nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    price_inr: Mapped[float] = mapped_column(Float, default=0.0)
    workspace_type: Mapped[str] = mapped_column(WORKSPACE_TYPE, nullable=False)
    monthly_points: Mapped[int] = mapped_column(Integer, default=100)
    storage_mb: Mapped[int] = mapped_column(Integer, default=100)
    max_seats: Mapped[int | None] = mapped_column(Integer)
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class PointCost(Base):
    __tablename__ = "point_costs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    action: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    cost: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)


class PointTransaction(Base):
    __tablename__ = "point_transactions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    points_used: Mapped[int] = mapped_column(Integer, nullable=False)
    balance_after: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    subscription: Mapped["Subscription"] = relationship(back_populates="transactions")


class SubscriptionAddon(Base):
    __tablename__ = "subscription_addons"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    addon_type: Mapped[str] = mapped_column(String(50), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    points_added: Mapped[int] = mapped_column(Integer, default=0)
    purchased_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    subscription: Mapped["Subscription"] = relationship(back_populates="addons")


class FeatureLimit(Base):
    __tablename__ = "feature_limits"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan: Mapped[str] = mapped_column(SUBSCRIPTION_PLAN, nullable=False)
    feature_key: Mapped[str] = mapped_column(String(100), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    daily_limit: Mapped[int | None] = mapped_column(Integer)
    monthly_limit: Mapped[int | None] = mapped_column(Integer)


class UsageCounter(Base):
    __tablename__ = "usage_counters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_key: Mapped[str] = mapped_column(String(100), nullable=False)
    period: Mapped[str] = mapped_column(String(20), nullable=False)  # 'daily' | 'monthly'
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
