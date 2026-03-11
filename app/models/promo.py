import uuid
from datetime import datetime
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class PromoCode(Base):
    __tablename__ = "promo_codes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    # "plan" or "addon"
    applies_to: Mapped[str] = mapped_column(String(20), nullable=False)
    # "percentage" or "fixed"
    discount_type: Mapped[str] = mapped_column(String(20), nullable=False, default="percentage")
    # e.g. 20.0 for 20% off, or 50 for ₹50 off
    discount_value: Mapped[float] = mapped_column(Float, nullable=False)
    # Optional: restrict to specific plan IDs or addon IDs (comma-separated), null = all
    applicable_items: Mapped[str | None] = mapped_column(String(500))
    max_uses: Mapped[int] = mapped_column(Integer, default=100)
    used_count: Mapped[int] = mapped_column(Integer, default=0)
    # Per-user usage limit (1 = each user can use it once)
    per_user_limit: Mapped[int] = mapped_column(Integer, default=1)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PromoUsage(Base):
    """Tracks per-user promo code usage to enforce per_user_limit."""
    __tablename__ = "promo_usages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    promo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
