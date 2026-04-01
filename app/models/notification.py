import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Index, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.database import Base


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    org_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    notification_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(30), nullable=False, default="system")  # personalized | system
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    data_json: Mapped[dict | None] = mapped_column(JSONB)
    icon: Mapped[str | None] = mapped_column(String(50))
    priority: Mapped[str] = mapped_column(String(10), nullable=False, default="normal")  # low | normal | high
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_dismissed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_notif_user_unread", "user_id", "is_read", "is_dismissed"),
        Index("idx_notif_user_created", "user_id", created_at.desc()),
        Index("idx_notif_expires", "expires_at"),
    )


# Notification type constants
class NotificationType:
    # Personalized
    DAILY_IMPROVEMENT = "daily_improvement"
    FEATURE_DISCOVERY = "feature_discovery"
    STUDY_CONTINUATION = "study_continuation"
    WEAK_TOPIC_ALERT = "weak_topic_alert"
    STREAK_MILESTONE = "streak_milestone"
    STREAK_AT_RISK = "streak_at_risk"
    XP_MILESTONE = "xp_milestone"
    SCORE_IMPROVEMENT = "score_improvement"

    # System
    PLAN_EXPIRY_APPROACHING = "plan_expiry_approaching"
    POINTS_LOW = "points_low"
    POINTS_EXHAUSTED = "points_exhausted"
    PLAN_EXPIRED = "plan_expired"
    PAYMENT_SUCCESS = "payment_success"
    PAYMENT_FAILURE = "payment_failure"
    NEW_ANNOUNCEMENT = "new_announcement"
    BADGE_EARNED = "badge_earned"
    TITLE_EARNED = "title_earned"
    ASSIGNMENT_POSTED = "assignment_posted"
    SUBMISSION_GRADED = "submission_graded"

    # Study time
    STUDY_TIME_MILESTONE = "study_time_milestone"
    STUDY_TIME_INACTIVE = "study_time_inactive"
    STUDY_TIME_WEEKLY_SUMMARY = "study_time_weekly_summary"
    STUDY_TIME_CONSISTENCY = "study_time_consistency"
