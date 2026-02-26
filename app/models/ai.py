import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, Integer, Float, DateTime, Text, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.database import Base


class AiChat(Base):
    __tablename__ = "ai_chats"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    scope: Mapped[str] = mapped_column(String(50), default="personal")  # personal | class
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    class_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("classes.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="ai_chats")  # noqa: F821
    messages: Mapped[list["AiChatMessage"]] = relationship(back_populates="chat", cascade="all, delete-orphan", order_by="AiChatMessage.created_at")
    settings: Mapped["AiChatSetting | None"] = relationship(back_populates="chat", uselist=False, cascade="all, delete-orphan")


class AiChatMessage(Base):
    __tablename__ = "ai_chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ai_chats.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(10))
    sources_json: Mapped[dict | None] = mapped_column(JSONB)  # [{title, url, snippet}]
    token_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chat: Mapped["AiChat"] = relationship(back_populates="messages")


class AiChatSetting(Base):
    __tablename__ = "ai_chat_settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ai_chats.id", ondelete="CASCADE"), nullable=False, unique=True)
    difficulty: Mapped[str] = mapped_column(String(20), default="medium")
    personality: Mapped[str] = mapped_column(String(50), default="helpful")
    content_length: Mapped[str] = mapped_column(String(20), default="medium")
    explain_3ways: Mapped[bool] = mapped_column(Boolean, default=False)
    mind_map: Mapped[bool] = mapped_column(Boolean, default=False)
    examples: Mapped[bool] = mapped_column(Boolean, default=True)
    output_mode: Mapped[str] = mapped_column(String(50), default="text")
    # Enhancement feature toggles
    student_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    video_refs: Mapped[bool] = mapped_column(Boolean, default=False)
    followup: Mapped[bool] = mapped_column(Boolean, default=False)
    practice: Mapped[bool] = mapped_column(Boolean, default=False)
    next_steps: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    chat: Mapped["AiChat"] = relationship(back_populates="settings")


class AiContextSession(Base):
    __tablename__ = "ai_context_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    workspace_id: Mapped[str] = mapped_column(String(100), nullable=False)  # 'personal' or org_id
    grade: Mapped[int | None] = mapped_column(Integer)
    board: Mapped[str | None] = mapped_column(String(50))
    subject: Mapped[str | None] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(10), default="en")
    tone: Mapped[str] = mapped_column(String(50), default="helpful")
    difficulty: Mapped[str] = mapped_column(String(20), default="medium")
    output_mode: Mapped[str] = mapped_column(String(50), default="text")
    student_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AiInteractionHistory(Base):
    __tablename__ = "ai_interaction_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False, index=True)
    service: Mapped[str] = mapped_column(String(100), nullable=False)  # ai-assistant | ask-doc | generate-ebook | ...
    query: Mapped[str | None] = mapped_column(Text)
    response_summary: Mapped[str | None] = mapped_column(Text)
    context_snapshot: Mapped[dict | None] = mapped_column(JSONB)  # {grade, board, subject, language, tone, difficulty}
    points_used: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    model_used: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IntelligenceCache(Base):
    __tablename__ = "intelligence_cache"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    cache_key: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
