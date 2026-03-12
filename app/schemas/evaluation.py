import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Any


class EvalPaperCreate(BaseModel):
    title: str
    board: Optional[str] = None
    grade: Optional[int] = None
    total_marks: Optional[int] = None
    negative_marking: bool = False
    negative_mark_value: float = 0.25
    time_limit: Optional[int] = None
    mode: str = "exam"
    difficulty: Optional[str] = None
    question_count: Optional[int] = None
    max_score: Optional[float] = None


class EvalPaperResponse(BaseModel):
    id: uuid.UUID
    org_id: uuid.UUID
    title: str
    board: Optional[str] = None
    grade: Optional[int] = None
    total_marks: Optional[int] = None
    negative_marking: bool
    negative_mark_value: float
    time_limit: Optional[int] = None
    mode: str
    status: str
    difficulty: Optional[str] = None
    question_count: Optional[int] = None
    max_score: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalSubjectCreate(BaseModel):
    subject: str
    marks_allocated: Optional[int] = None
    order_index: int = 0


class EvalChapterCreate(BaseModel):
    paper_subject_id: str
    chapter_name: str
    weightage: float = 1.0
    question_count: Optional[int] = None


class EvalQuestionCreate(BaseModel):
    paper_id: str
    question_type: str
    question_text: str
    options: Optional[dict] = None
    correct_answer: Optional[str] = None
    marks: float = 1.0
    negative_marks: float = 0.0
    subject: Optional[str] = None
    chapter: Optional[str] = None
    difficulty: Optional[str] = None
    explanation: Optional[str] = None
    tags: Optional[List[str]] = None
    source_type: Optional[str] = None
    blooms_level: Optional[str] = None


class EvalQuestionUpdate(BaseModel):
    question_text: Optional[str] = None
    options: Optional[dict] = None
    correct_answer: Optional[str] = None
    marks: Optional[float] = None
    explanation: Optional[str] = None
    tags: Optional[List[str]] = None


class EvalQuestionResponse(BaseModel):
    id: uuid.UUID
    paper_id: uuid.UUID
    question_type: str
    question_text: str
    options: Optional[Any] = None
    correct_answer: Optional[str] = None
    marks: float
    negative_marks: float
    subject: Optional[str] = None
    chapter: Optional[str] = None
    difficulty: Optional[str] = None
    explanation: Optional[str] = None
    source_type: Optional[str] = None
    blooms_level: Optional[str] = None
    is_ai_generated: bool
    order_index: int

    model_config = {"from_attributes": True}


class GeneratePaperRequest(BaseModel):
    paper_id: str
    subjects: List[dict]  # [{subject, chapters: [{name, weightage, count}]}]
    question_types: Optional[List[str]] = None


# ---- New schemas for full paper generation flow ----

class EvalSubjectConfigSchema(BaseModel):
    subject: str
    weightage: float = 100
    source_type: str = "online"  # online | text | file
    source_text: Optional[str] = None
    chapters: Optional[List[dict]] = None  # [{name, weightage}]


class GenerateEvalPaperRequest(BaseModel):
    title: str
    grade: Optional[int] = None
    board: Optional[str] = None
    difficulty: str = "medium"
    blooms_level: str = "mixed"
    question_count: int = 20
    question_types: List[str] = ["mcq"]
    mcq_subtypes: Optional[List[str]] = None
    type_weightage: Optional[dict] = None
    negative_marking: bool = False
    negative_mark_value: float = 0.25
    subjects: List[EvalSubjectConfigSchema]


class GenerateEvalPaperResponse(BaseModel):
    question_json: List[dict]
    answer_key_json: List[dict]


class SaveEvalPaperRequest(BaseModel):
    org_id: str
    config: dict  # Full EvalPaperConfig from frontend
    questions: List[dict]
    answer_key: Optional[List[dict]] = None


class EvalAssessmentCreate(BaseModel):
    paper_id: Optional[str] = None
    title: str
    grade: Optional[int] = None
    board: Optional[str] = None
    difficulty: Optional[str] = None
    question_ids: Optional[List[str]] = None
    question_count: Optional[int] = None
    max_score: Optional[float] = None
    due_date: Optional[datetime] = None
    mode: str = "exam"
    time_limit: Optional[int] = None
    time_limit_seconds: Optional[int] = None
    negative_marking: bool = False
    negative_mark_value: Optional[float] = None
    scheduled_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None
    max_attempts: Optional[int] = None  # for practice mode; None = unlimited


class EvalAssessmentResponse(BaseModel):
    id: uuid.UUID
    paper_id: Optional[uuid.UUID] = None
    org_id: uuid.UUID
    title: str
    mode: str
    time_limit: Optional[int] = None
    negative_marking: bool
    negative_mark_value: Optional[float] = None
    question_count: Optional[int] = None
    max_score: Optional[float] = None
    difficulty: Optional[str] = None
    grade: Optional[int] = None
    board: Optional[str] = None
    due_date: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None
    max_attempts: Optional[int] = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DistributeAssessmentRequest(BaseModel):
    emails: Optional[List[str]] = None
    class_id: Optional[str] = None
    org_id: Optional[str] = None
    # Legacy fields
    class_ids: Optional[List[str]] = None
    student_ids: Optional[List[str]] = None


class EvalAttemptSubmit(BaseModel):
    responses: dict  # {questionId: answer}


class EvalAttemptResponse(BaseModel):
    id: uuid.UUID
    assessment_id: uuid.UUID
    student_id: Optional[uuid.UUID] = None
    invitation_id: Optional[uuid.UUID] = None
    score: Optional[float] = None
    max_score: Optional[float] = None
    percentage: Optional[float] = None
    started_at: datetime
    submitted_at: Optional[datetime] = None
    status: str

    model_config = {"from_attributes": True}


# ---- Invitation response (for report enrichment) ----

class EvalInvitationResponse(BaseModel):
    id: uuid.UUID
    assessment_id: uuid.UUID
    student_id: Optional[uuid.UUID] = None
    email: str
    name: Optional[str] = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---- Public assessment endpoints (no auth required) ----

class PublicAssessmentInfoResponse(BaseModel):
    assessment_title: str
    mode: str
    question_count: Optional[int] = None
    time_limit: Optional[int] = None
    negative_marking: bool = False
    due_date: Optional[datetime] = None
    max_attempts: Optional[int] = None
    attempts_used: int = 0
    can_attempt: bool = True
    invitation_status: str = "pending"
    email: str = ""
    organization_name: Optional[str] = None


class PublicStartAttemptRequest(BaseModel):
    name: Optional[str] = None


class PublicAttemptResponse(BaseModel):
    attempt_id: uuid.UUID
    assessment_id: uuid.UUID
    questions: List[dict]  # NO correct_answer included
    time_limit: Optional[int] = None
    started_at: datetime
    responses: Optional[dict] = None  # Included on resume so student sees their saved answers


class PublicAutosaveRequest(BaseModel):
    responses: dict  # {questionId: answer}


class PublicSubmitRequest(BaseModel):
    responses: dict  # {questionId: answer}
    metadata: Optional[dict] = None  # {tab_violations, auto_submitted, submit_reason}


class PublicSubmitResponse(BaseModel):
    attempt_id: uuid.UUID
    status: str
    message: str


# ---- Student's own evaluation results (org-scoped) ----

class MyEvalResultResponse(BaseModel):
    attempt_id: uuid.UUID
    assessment_id: uuid.UUID
    assessment_title: str
    mode: str
    score: Optional[float] = None
    max_score: Optional[float] = None
    percentage: Optional[float] = None
    status: str
    started_at: datetime
    submitted_at: Optional[datetime] = None
    org_id: uuid.UUID
    org_name: str
    difficulty: Optional[str] = None
    question_count: Optional[int] = None
    grade: Optional[int] = None
    board: Optional[str] = None
