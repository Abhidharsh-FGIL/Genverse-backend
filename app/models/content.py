import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, Integer, Float, DateTime, Text, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.database import Base


class UserLibraryItem(Base):
    __tablename__ = "user_library_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str | None] = mapped_column(String(50))  # pdf, docx, txt, jpg, etc.
    storage_path: Mapped[str | None] = mapped_column(String(1000))
    file_size_mb: Mapped[float | None] = mapped_column(Float)
    folder: Mapped[str | None] = mapped_column(String(255))
    tags: Mapped[dict | None] = mapped_column(JSONB)  # string[]
    extracted_text_ref: Mapped[str | None] = mapped_column(String(1000))
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="library_items")  # noqa: F821
    chunks: Mapped[list["DocChunk"]] = relationship(back_populates="library_item", cascade="all, delete-orphan")


class DocChunk(Base):
    __tablename__ = "doc_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    library_item_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("user_library_items.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_order: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    # Embeddings are stored in FAISS index files (see FAISSService) — not in the DB

    library_item: Mapped["UserLibraryItem"] = relationship(back_populates="chunks")


class Ebook(Base):
    __tablename__ = "ebooks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(100))
    grade: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(10), default="en")
    source_type: Mapped[str | None] = mapped_column(String(50))  # topic | vault | pdf
    source_ref_id: Mapped[str | None] = mapped_column(String(500))
    ebook_json: Mapped[dict | None] = mapped_column(JSONB)  # Full eBook content
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    points_used: Mapped[int] = mapped_column(Integer, default=0)
    storage_path: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    audiobook: Mapped["Audiobook | None"] = relationship(back_populates="ebook", uselist=False, cascade="all, delete-orphan")


class Audiobook(Base):
    __tablename__ = "audiobooks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ebook_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ebooks.id", ondelete="CASCADE"), nullable=False, unique=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False, index=True)
    audio_path: Mapped[str | None] = mapped_column(String(1000))
    language: Mapped[str] = mapped_column(String(10), default="en")
    voice_profile: Mapped[str | None] = mapped_column(String(100))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ebook: Mapped["Ebook"] = relationship(back_populates="audiobook")


class MindMap(Base):
    __tablename__ = "mindmaps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(100))
    grade: Mapped[int | None] = mapped_column(Integer)
    board: Mapped[str | None] = mapped_column(String(50))
    mindmap_json: Mapped[dict | None] = mapped_column(JSONB)  # Node/edge structure
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class VideoProject(Base):
    __tablename__ = "video_projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(100))
    grade: Mapped[int | None] = mapped_column(Integer)
    script_json: Mapped[dict | None] = mapped_column(JSONB)  # Full script with scenes
    visuals_json: Mapped[dict | None] = mapped_column(JSONB)  # Visual references
    references_json: Mapped[dict | None] = mapped_column(JSONB)  # Reference materials
    status: Mapped[str] = mapped_column(String(20), default="draft")
    points_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PastPaper(Base):
    __tablename__ = "past_papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    board: Mapped[str | None] = mapped_column(String(50))
    grade: Mapped[int | None] = mapped_column(Integer)
    subject: Mapped[str | None] = mapped_column(String(100))
    year: Mapped[int | None] = mapped_column(Integer)
    exam_type: Mapped[str | None] = mapped_column(String(100))  # Midterm, Final, Board, etc.
    storage_path: Mapped[str | None] = mapped_column(String(1000))
    file_url: Mapped[str | None] = mapped_column(String(1000))
    question_json: Mapped[dict | None] = mapped_column(JSONB)  # Parsed questions
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"))
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
