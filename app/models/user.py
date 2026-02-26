import uuid
from datetime import datetime
from sqlalchemy import (
    String, Boolean, Integer, DateTime, Text, ARRAY,
    ForeignKey, func, Enum as SAEnum
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.database import Base

APP_ROLE = SAEnum(
    "admin", "moderator", "normal_user", "student", "teacher", "guardian", "org_admin",
    name="app_role",
)

PERSONA_BAND = SAEnum("A", "B", "C", "D", "E", name="persona_band")


class User(Base):
    __tablename__ = "profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(String(500))
    phone: Mapped[str | None] = mapped_column(String(20))
    address: Mapped[str | None] = mapped_column(Text)
    grade: Mapped[int | None] = mapped_column(Integer)
    persona_band: Mapped[str | None] = mapped_column(PERSONA_BAND)
    language: Mapped[str] = mapped_column(String(10), default="en")
    subjects: Mapped[list | None] = mapped_column(ARRAY(String))
    xp: Mapped[int] = mapped_column(Integer, default=0)
    streak: Mapped[int] = mapped_column(Integer, default=0)
    last_login_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    roles: Mapped[list["UserRole"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    @property
    def role(self) -> str:
        """Returns the user's primary role based on priority (highest privilege first)."""
        if self.roles:
            _priority = ["org_admin", "teacher", "guardian", "student", "normal_user"]
            roles_set = {r.role for r in self.roles}
            for r in _priority:
                if r in roles_set:
                    return r
        return "normal_user"
    ai_chats: Mapped[list["AiChat"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    library_items: Mapped[list["UserLibraryItem"]] = relationship(back_populates="user", cascade="all, delete-orphan")  # noqa: F821
    subscriptions: Mapped[list["Subscription"]] = relationship(back_populates="user")  # noqa: F821


class UserRole(Base):
    __tablename__ = "user_roles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(APP_ROLE, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="roles")
