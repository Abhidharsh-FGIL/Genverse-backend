import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Any, Literal


class LibraryItemResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    file_type: Optional[str] = None
    storage_path: Optional[str] = None
    file_size_mb: Optional[float] = None
    folder: Optional[str] = None
    tags: Optional[Any] = None
    extracted_text_ref: Optional[str] = None
    is_processed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class LibraryItemUpdate(BaseModel):
    title: Optional[str] = None
    folder: Optional[str] = None
    tags: Optional[List[str]] = None
    extracted_text: Optional[str] = None  # For updating OCR extracted text


class VaultQueryRequest(BaseModel):
    query: str
    file_ids: Optional[List[str]] = None
    context: Optional[dict] = None


class VaultQueryResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None
    points_used: int = 3


class EbookOutlineRequest(BaseModel):
    title: str
    topic: str
    subject: Optional[str] = None
    grade: Optional[int] = None
    language: str = "en"
    book_size: Optional[Literal["short", "medium", "large"]] = "short"
    tone: Optional[Literal["academic", "simple", "story_based", "exam_oriented"]] = "academic"


class EbookOutlineChapter(BaseModel):
    title: str
    description: str


class EbookOutlineResponse(BaseModel):
    chapters: List[EbookOutlineChapter]


class EbookChapterInput(BaseModel):
    title: str
    description: Optional[str] = None


class AssessmentConfig(BaseModel):
    enabled: bool = False
    placement: str = "end_of_chapter"   # end_of_chapter | final_section | both
    difficulty: str = "medium"          # easy | medium | hard | mixed
    questionTypes: List[str] = ["MCQ"]  # MCQ | Fill in Blank | Short Answer
    bloomsLevel: str = "understand"     # remember | understand | apply | analyze | evaluate | create


class EbookGenerateRequest(BaseModel):
    title: str
    topic: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[int] = None
    language: str = "en"
    author: Optional[str] = None
    source_type: str = "topic"  # topic | vault | pdf
    source_ref_id: Optional[str] = None
    chapters: Optional[List[EbookChapterInput]] = None  # from wizard outline step
    outline: Optional[List[str]] = None  # legacy flat list
    book_size: Optional[Literal["short", "medium", "large"]] = None
    page_count: Optional[int] = None
    tone: Optional[Literal["academic", "simple", "story_based", "exam_oriented"]] = "academic"
    image_density: Optional[Literal["minimal", "standard", "visual_heavy"]] = "standard"
    image_types: Optional[List[str]] = None
    assessment_config: Optional[AssessmentConfig] = None


class EbookGeneratedContent(BaseModel):
    ebook_json: Any
    page_count: int
    points_used: int


class EbookCreateRequest(BaseModel):
    title: str
    ebook_json: Any
    language: str = "en"
    source_type: str = "topic"
    source_ref_id: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[int] = None
    page_count: Optional[int] = None
    points_used: Optional[int] = None


class EbookResponse(BaseModel):
    id: uuid.UUID
    title: str
    subject: Optional[str] = None
    grade: Optional[int] = None
    language: str
    source_type: Optional[str] = None
    ebook_json: Optional[Any] = None
    page_count: int
    points_used: int
    storage_path: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AudiobookGenerateRequest(BaseModel):
    ebook_id: Optional[str] = None
    language: str = "en"
    voice_profile: Optional[str] = None
    narration_style: Optional[Literal["standard", "slow_clear", "energetic", "calm"]] = "standard"


class AudiobookResponse(BaseModel):
    id: uuid.UUID
    ebook_id: uuid.UUID
    audio_path: Optional[str] = None
    language: str
    voice_profile: Optional[str] = None
    narration_style: Optional[str] = None
    duration_seconds: Optional[int] = None
    chapter_timestamps: Optional[List[dict]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AudiobookVoicesResponse(BaseModel):
    language: str
    voices: List[dict]


class MindMapGenerateRequest(BaseModel):
    topic: str
    subject: Optional[str] = None
    grade: Optional[int] = None
    board: Optional[str] = None
    depth: int = 3


class MindMapResponse(BaseModel):
    id: uuid.UUID
    title: str
    topic: str
    subject: Optional[str] = None
    mindmap_json: Optional[Any] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class VideoScriptRequest(BaseModel):
    topic: str
    subject: Optional[str] = None
    grade: Optional[int] = None
    duration_minutes: int = 5
    style: str = "educational"


class VideoProjectResponse(BaseModel):
    id: uuid.UUID
    title: str
    topic: str
    subject: Optional[str] = None
    script_json: Optional[Any] = None
    visuals_json: Optional[Any] = None
    references_json: Optional[Any] = None
    status: str
    points_used: int
    created_at: datetime

    model_config = {"from_attributes": True}


class PastPaperResponse(BaseModel):
    id: uuid.UUID
    title: str
    board: Optional[str] = None
    grade: Optional[int] = None
    subject: Optional[str] = None
    year: Optional[int] = None
    exam_type: Optional[str] = None
    file_url: Optional[str] = None
    is_public: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class OCRRequest(BaseModel):
    language: str = "en"


class OCRResponse(BaseModel):
    extracted_text: str
    word_count: int
    language: str
    storage_path: Optional[str] = None


class OCRExtractResponse(BaseModel):
    item: LibraryItemResponse
    extracted_text: str
    word_count: int
    language: str
