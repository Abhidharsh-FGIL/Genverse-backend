import uuid
from datetime import datetime
from pydantic import BaseModel, field_validator
from typing import Optional, List, Any


class UserInsightResponse(BaseModel):
    id: uuid.UUID
    insight_type: str
    title: str
    content: str
    data_json: Optional[Any] = None
    is_read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class InsightArticleResponse(BaseModel):
    id: uuid.UUID
    title: str
    summary: Optional[str] = None
    content: str
    subject: Optional[str] = None
    tags: Optional[Any] = None
    reading_time_minutes: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class GenerateInsightsRequest(BaseModel):
    force_refresh: bool = False


class IntelligenceRequest(BaseModel):
    modules: Optional[List[str]] = None  # ['dashboard', 'bloom', 'recommendations']
    force_refresh: bool = False


class IntelligenceResponse(BaseModel):
    dashboard_snapshot: Optional[dict] = None
    bloom_profile: Optional[dict] = None
    recommendations: Optional[List[dict]] = None
    learning_trends: Optional[dict] = None
    topic_strengths: Optional[List[dict]] = None
    topic_gaps: Optional[List[dict]] = None
    cached: bool = False
    generated_at: Optional[str] = None


class RecommendationResponse(BaseModel):
    id: uuid.UUID
    rec_type: str
    title: str
    description: Optional[str] = None
    reason: Optional[str] = None
    metadata_json: Optional[Any] = None
    is_acted_on: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class CriterionAverage(BaseModel):
    name: str
    average: float


class WeakOutcome(BaseModel):
    criterion: str
    average: float


class ClassRecommendationRequest(BaseModel):
    class_id: str
    class_name: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    total_students: int = 0
    submissions_graded: int = 0
    class_average: float = 0
    students_needing_help: int = 0
    criterion_averages: List[CriterionAverage] = []
    weak_outcomes: List[WeakOutcome] = []

    @field_validator('grade', 'board', 'subject', 'class_name', mode='before')
    @classmethod
    def coerce_to_str(cls, v: Any) -> Optional[str]:
        return str(v) if v is not None else None


class ClassRecommendationItem(BaseModel):
    title: str
    description: str
    targets: str
    action_type: str
    priority: str


class LearningCurveResponse(BaseModel):
    user_id: str
    assessment_scores: List[dict]
    topic_mastery_trend: List[dict]
    xp_trend: List[dict]
    streak_history: List[int]
    weekly_activity: dict
