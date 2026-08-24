import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Any


class AiChatCreate(BaseModel):
    title: str
    scope: str = "personal"
    class_id: Optional[str] = None
    org_id: Optional[str] = None


class AiChatUpdate(BaseModel):
    title: Optional[str] = None


class AiChatResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    scope: str
    title: str
    class_id: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AiMessageCreate(BaseModel):
    content: str
    role: str = "user"
    selected_files: Optional[List[str]] = None


class SendMessageRequest(BaseModel):
    chat_id: Optional[str] = None
    message: str
    context: Optional[dict] = None
    selected_files: Optional[List[str]] = None
    chat_settings: Optional[dict] = None


class AiMessageResponse(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    role: str
    content: str
    language: Optional[str] = None
    sources_json: Optional[Any] = None
    enhancements_json: Optional[Any] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSettingsUpdate(BaseModel):
    difficulty: Optional[str] = None
    personality: Optional[str] = None
    content_length: Optional[str] = None
    explain_3ways: Optional[bool] = None
    mind_map: Optional[bool] = None
    infographic: Optional[bool] = None
    examples: Optional[bool] = None
    output_mode: Optional[str] = None
    student_mode: Optional[bool] = None
    video_refs: Optional[bool] = None
    followup: Optional[bool] = None
    practice: Optional[bool] = None
    next_steps: Optional[bool] = None


class ChatSettingsResponse(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    difficulty: str
    personality: str
    content_length: str
    explain_3ways: bool
    mind_map: bool
    infographic: bool
    examples: bool
    output_mode: str
    student_mode: bool
    video_refs: bool
    followup: bool
    practice: bool
    next_steps: bool
    updated_at: datetime

    model_config = {"from_attributes": True}


class PlaygroundRequest(BaseModel):
    topic: str
    mode: str  # experiment | play | challenge | imagine
    messages: Optional[List[dict]] = None
    grade: Optional[int] = None
    board: Optional[str] = None
    student_mode: bool = True
    harder_mode: bool = False
    context: Optional[dict] = None


class CareerGuidanceRequest(BaseModel):
    interests: List[str]
    strengths: List[str]
    target_careers: Optional[List[str]] = None
    grade: Optional[int] = None
    context: Optional[dict] = None


class CareerGuidanceResponse(BaseModel):
    id: uuid.UUID
    interests: Optional[Any] = None
    strengths: Optional[Any] = None
    target_careers: Optional[Any] = None
    analysis_json: Optional[Any] = None
    compatibility_scores: Optional[Any] = None
    points_used: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AudioQARequest(BaseModel):
    question: str
    context: Optional[dict] = None
    language: str = "en"


class AudioQAResponse(BaseModel):
    text_response: str
    audio_path: Optional[str] = None
    language: str
    points_used: int


class FollowUpRequest(BaseModel):
    message: str
    response: str
    count: int = 4
    language: Optional[str] = None


class FollowUpResponse(BaseModel):
    questions: List[str]


class VideoRefsRequest(BaseModel):
    message: str
    response: str
    grade: Optional[int] = None
    student_mode: Optional[bool] = None
    language: Optional[str] = None


class VideoResult(BaseModel):
    title: str
    channel: str
    thumbnail: str
    video_id: str
    url: str


class VideoRefsResponse(BaseModel):
    videos: List[VideoResult]


class NextStepsRequest(BaseModel):
    message: str
    response: str
    count: int = 4
    language: Optional[str] = None


class NextStepsResponse(BaseModel):
    steps: List[str]


class PracticeExercisesRequest(BaseModel):
    message: str
    response: str
    count: int = 3
    grade: Optional[int] = None
    difficulty: str = "medium"
    language: Optional[str] = None


class PracticeExercise(BaseModel):
    question: str
    answer: str
    type: str  # short | mcq | true_false
    options: Optional[List[str]] = None


class PracticeExercisesResponse(BaseModel):
    exercises: List[PracticeExercise]


class GeneratePracticeAssessmentRequest(BaseModel):
    topic: Optional[str] = None
    multi_topics: Optional[List[str]] = None
    subject: str
    difficulty: str = "medium"
    question_count: int = 10
    question_types: Optional[List[str]] = None
    mcq_subtypes: Optional[List[str]] = None
    grade: Optional[int] = None
    board: Optional[str] = None
    mode: str = "practice"
    blooms_level: Optional[str] = None
    negative_marking: bool = False
    negative_mark_value: Optional[float] = None
    source_text: Optional[str] = None
    source_type: Optional[str] = None
    source_ref_id: Optional[str] = None  # Vault file ID — content fetched server-side
    source_ref_ids: Optional[List[str]] = None
    library_file_ids: Optional[List[str]] = None  # Public library file IDs for RAG
    type_weightage: Optional[dict] = None
    topic_weightage: Optional[dict] = None
    chapter_weightage: Optional[dict] = None
    language: Optional[str] = None  # Response language (e.g. "Tamil", "Hindi")
    org_id: Optional[str] = None  # Org workspace: deduct from org subscription
    # Accept remaining extra fields from the frontend without validation errors
    model_config = {"extra": "ignore"}


class GeneratedQuestionsResponse(BaseModel):
    question_json: List[dict]
    answer_key_json: List[dict]


class MindMapRequest(BaseModel):
    message: str
    response: str
    grade: Optional[int] = None
    language: str = "English"


class MindMapResponse(BaseModel):
    mindmap: Any  # JSON tree structure


class InfographicRequest(BaseModel):
    message: str
    response: str
    grade: Optional[int] = None
    language: str = "English"


class InfographicResponse(BaseModel):
    infographic: Any  # JSON structure with sections
