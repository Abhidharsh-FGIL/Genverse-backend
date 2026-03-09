import uuid
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Text, func, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class PublicFolder(Base):
    """Two-level folder structure: Board → Grade. Files (one per subject) live inside grade folders."""
    __tablename__ = "public_folders"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("public_folders.id", ondelete="CASCADE"), index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    folder_type: Mapped[str | None] = mapped_column(String(50))  # "board" | "grade"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Self-referential relationships
    parent: Mapped["PublicFolder | None"] = relationship(
        back_populates="children", remote_side="PublicFolder.id",
    )
    children: Mapped[list["PublicFolder"]] = relationship(
        back_populates="parent", cascade="all, delete-orphan",
    )
    files: Mapped[list["PublicFile"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("parent_id", "name", name="uq_public_folders_parent_name"),
    )


class PublicFile(Base):
    """A file inside a public folder, with text extracted and chunked into FAISS."""
    __tablename__ = "public_files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    folder_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("public_folders.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(255))  # e.g. "Mathematics", "Physics"
    file_type: Mapped[str | None] = mapped_column(String(255))
    storage_path: Mapped[str | None] = mapped_column(String(1000))
    file_size_mb: Mapped[float | None] = mapped_column(Float)
    is_processed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    folder: Mapped["PublicFolder"] = relationship(back_populates="files")
    chunks: Mapped[list["PublicFileChunk"]] = relationship(
        back_populates="file", cascade="all, delete-orphan",
    )


class PublicFileChunk(Base):
    """Text chunks for a public file — embeddings stored in FAISS under key 'public_library'."""
    __tablename__ = "public_file_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("public_files.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_order: Mapped[int] = mapped_column(Integer, nullable=False)

    file: Mapped["PublicFile"] = relationship(back_populates="chunks")
