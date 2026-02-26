import uuid
from datetime import datetime
from pydantic import BaseModel, model_validator
from typing import Optional, List, Any


class AssessmentCreate(BaseModel):
    title: str
    subject: str
    board: Optional[str] = None
    grade: Optional[int] = None
    topics: Optional[List[str]] = None
    difficulty: str = "medium"
    mode: str = "practice"
    question_count: int = 10
    question_types: Optional[List[str]] = None
    time_limit: Optional[int] = None
    is_adaptive: bool = False
    negative_marking: bool = False
    negative_mark_value: float = 0.25


class GenerateAssessmentRequest(AssessmentCreate):
    pass


class AssessmentResponse(BaseModel):
    id: uuid.UUID
    title: str
    subject: str
    board: Optional[str] = None
    grade: Optional[int] = None
    topics: Optional[Any] = None
    difficulty: str
    mode: str
    question_count: int
    question_types: Optional[Any] = None
    question_json: Optional[Any] = None
    time_limit: Optional[int] = None
    is_adaptive: bool
    negative_marking: bool = False
    negative_mark_value: float = 0.25
    created_at: datetime

    model_config = {"from_attributes": True}


class AttemptStartResponse(BaseModel):
    id: uuid.UUID
    assessment_id: uuid.UUID
    started_at: datetime
    status: str

    model_config = {"from_attributes": True}


class AttemptSubmitRequest(BaseModel):
    responses: dict  # {questionId: answer}


class AttemptResponse(BaseModel):
    id: uuid.UUID
    assessment_id: uuid.UUID
    user_id: uuid.UUID
    score: Optional[float] = None
    max_score: Optional[float] = None
    percentage: Optional[float] = None
    responses_json: Optional[Any] = None
    feedback_json: Optional[Any] = None
    topic_mastery_update: Optional[Any] = None
    xp_earned: int
    started_at: datetime
    submitted_at: Optional[datetime] = None
    time_taken_seconds: Optional[int] = None
    status: str

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def compute_time_taken(self):
        if self.time_taken_seconds is None and self.submitted_at and self.started_at:
            self.time_taken_seconds = int(
                (self.submitted_at - self.started_at).total_seconds()
            )
        return self


class TopicMasteryResponse(BaseModel):
    id: uuid.UUID
    subject: str
    topic: str
    mastery_level: float
    attempts_count: int
    correct_count: int
    trend: str
    last_attempted_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TopicMasteryUpsert(BaseModel):
    subject: str
    topic: str
    mastery_level: float
    total_attempts: int = 0
    correct_count: int = 0


class IntegrityEventRequest(BaseModel):
    attempt_id: str
    event_type: str  # tab_switch | copy_paste | screenshot | ...
    event_data: Optional[dict] = None


class AssessmentSaveRequest(BaseModel):
    """Save a pre-generated (reviewed) assessment to the library."""
    title: str
    subject: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[int] = None
    topics: Optional[List[str]] = None
    difficulty: str = "medium"
    mode: str = "practice"
    time_limit: Optional[int] = None
    negative_marking: bool = False
    negative_mark_value: float = 0.25
    questions: List[Any] = []
