import uuid
from datetime import datetime, date
from sqlalchemy import String, Integer, Date, DateTime, Index, UniqueConstraint, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class StudyTimeDaily(Base):
    __tablename__ = "study_time_daily"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    org_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    subject: Mapped[str] = mapped_column(String(100), nullable=False, default="General")
    total_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assessment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assessment_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    peak_hour: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "org_id", "date", "subject", name="uq_study_time_daily"),
        Index("idx_study_time_user_date", "user_id", "date"),
        Index("idx_study_time_org_date", "org_id", "date"),
    )
