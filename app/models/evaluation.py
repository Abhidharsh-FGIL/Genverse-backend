import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, Integer, Float, DateTime, Text, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.database import Base


class EvaluationQuestionPaper(Base):
    __tablename__ = "evaluation_question_papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    board: Mapped[str | None] = mapped_column(String(50))
    grade: Mapped[int | None] = mapped_column(Integer)
    total_marks: Mapped[int | None] = mapped_column(Integer)
    negative_marking: Mapped[bool] = mapped_column(Boolean, default=False)
    negative_mark_value: Mapped[float] = mapped_column(Float, default=0.25)
    time_limit: Mapped[int | None] = mapped_column(Integer)  # minutes
    mode: Mapped[str] = mapped_column(String(20), default="exam")  # exam | practice
    status: Mapped[str] = mapped_column(String(20), default="draft")
    config: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    subjects: Mapped[list["EvaluationPaperSubject"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    questions: Mapped[list["EvaluationQuestion"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    assessments: Mapped[list["EvaluationAssessment"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class EvaluationPaperSubject(Base):
    __tablename__ = "evaluation_paper_subjects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evaluation_question_papers.id", ondelete="CASCADE"), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(100), nullable=False)
    marks_allocated: Mapped[int | None] = mapped_column(Integer)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    paper: Mapped["EvaluationQuestionPaper"] = relationship(back_populates="subjects")
    chapters: Mapped[list["EvaluationPaperChapter"]] = relationship(back_populates="paper_subject", cascade="all, delete-orphan")


class EvaluationPaperChapter(Base):
    __tablename__ = "evaluation_paper_chapters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_subject_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evaluation_paper_subjects.id", ondelete="CASCADE"), nullable=False, index=True)
    chapter_name: Mapped[str] = mapped_column(String(255), nullable=False)
    weightage: Mapped[float] = mapped_column(Float, default=1.0)  # percentage of total marks
    question_count: Mapped[int | None] = mapped_column(Integer)

    paper_subject: Mapped["EvaluationPaperSubject"] = relationship(back_populates="chapters")


class EvaluationQuestion(Base):
    __tablename__ = "evaluation_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evaluation_question_papers.id", ondelete="CASCADE"), nullable=False, index=True)
    question_type: Mapped[str] = mapped_column(String(30), nullable=False)  # MCQ | fill | match | true_false | short | long
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[dict | None] = mapped_column(JSONB)  # For MCQ: {A, B, C, D}
    correct_answer: Mapped[str | None] = mapped_column(Text)
    marks: Mapped[float] = mapped_column(Float, default=1.0)
    negative_marks: Mapped[float] = mapped_column(Float, default=0.0)
    subject: Mapped[str | None] = mapped_column(String(100))
    chapter: Mapped[str | None] = mapped_column(String(255))
    difficulty: Mapped[str | None] = mapped_column(String(20))
    explanation: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[dict | None] = mapped_column(JSONB)
    is_ai_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["EvaluationQuestionPaper"] = relationship(back_populates="questions")


class EvaluationAssessment(Base):
    __tablename__ = "evaluation_assessments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evaluation_question_papers.id", ondelete="CASCADE"), nullable=False, index=True)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False, index=True)
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    mode: Mapped[str] = mapped_column(String(20), default="exam")
    time_limit: Mapped[int | None] = mapped_column(Integer)
    negative_marking: Mapped[bool] = mapped_column(Boolean, default=False)
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    paper: Mapped["EvaluationQuestionPaper"] = relationship(back_populates="assessments")
    invitations: Mapped[list["EvaluationInvitation"]] = relationship(back_populates="assessment", cascade="all, delete-orphan")
    attempts: Mapped[list["EvaluationAttempt"]] = relationship(back_populates="assessment", cascade="all, delete-orphan")


class EvaluationInvitation(Base):
    __tablename__ = "evaluation_invitations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evaluation_assessments.id", ondelete="CASCADE"), nullable=False, index=True)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False, index=True)
    class_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("classes.id"))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | accepted | completed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    assessment: Mapped["EvaluationAssessment"] = relationship(back_populates="invitations")


class EvaluationAttempt(Base):
    __tablename__ = "evaluation_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evaluation_assessments.id", ondelete="CASCADE"), nullable=False, index=True)
    student_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False, index=True)
    responses: Mapped[dict | None] = mapped_column(JSONB)  # {questionId: answer}
    score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float | None] = mapped_column(Float)
    percentage: Mapped[float | None] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="in_progress")

    assessment: Mapped["EvaluationAssessment"] = relationship(back_populates="attempts")
