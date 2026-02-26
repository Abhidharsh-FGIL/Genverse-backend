import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Any


class ClassCreate(BaseModel):
    name: str
    board: str
    grade: int
    subject: str
    section: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    org_id: Optional[str] = None
    teacher_id: Optional[str] = None


class ClassUpdate(BaseModel):
    name: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[int] = None
    subject: Optional[str] = None
    section: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ClassResponse(BaseModel):
    id: uuid.UUID
    name: str
    board: str
    grade: int
    subject: str
    section: Optional[str] = None
    join_code: str
    teacher_id: uuid.UUID
    color: Optional[str] = None
    description: Optional[str] = None
    is_active: bool
    student_count: int = 0
    org_id: Optional[uuid.UUID] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ClassStudentResponse(BaseModel):
    id: uuid.UUID
    class_id: uuid.UUID
    student_id: uuid.UUID
    roll_no: Optional[str] = None
    joined_at: datetime
    student_name: Optional[str] = None
    student_email: Optional[str] = None

    model_config = {"from_attributes": True}


class JoinClassRequest(BaseModel):
    join_code: str


class AssignmentCreate(BaseModel):
    class_id: str
    title: str
    topic: Optional[str] = None
    instructions: str
    due_date: Optional[datetime] = None
    points: int = 100
    rubric_id: Optional[str] = None
    status: str = "draft"
    questions: Optional[List[Any]] = None
    attachments: Optional[List[Any]] = None
    target_student_ids: Optional[List[str]] = None


class AssignmentUpdate(BaseModel):
    title: Optional[str] = None
    topic: Optional[str] = None
    instructions: Optional[str] = None
    due_date: Optional[datetime] = None
    points: Optional[int] = None
    rubric_id: Optional[str] = None
    status: Optional[str] = None
    questions: Optional[List[Any]] = None
    attachments: Optional[List[Any]] = None
    target_student_ids: Optional[List[str]] = None


class AssignmentResponse(BaseModel):
    id: uuid.UUID
    class_id: uuid.UUID
    title: str
    topic: Optional[str] = None
    instructions: str
    due_date: Optional[datetime] = None
    points: int
    rubric_id: Optional[uuid.UUID] = None
    status: str
    questions: Optional[Any] = None
    attachments: Optional[Any] = None
    target_student_ids: Optional[Any] = None
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SubmissionCreate(BaseModel):
    assignment_id: str
    text_response: Optional[str] = None
    answers: Optional[Any] = None   # question-id → answer map sent by student
    files: Optional[Any] = None     # list of {name, url} uploaded by student


class SubmissionGradeRequest(BaseModel):
    grade: dict  # {totalScore, maxScore, criterionScores, overallComment}
    remediation_plan: Optional[List[Any]] = None
    return_to_student: bool = True


class SubmissionResponse(BaseModel):
    id: uuid.UUID
    assignment_id: uuid.UUID
    student_id: uuid.UUID
    submitted_at: Optional[datetime] = None
    status: str
    files: Optional[Any] = None
    text_response: Optional[str] = None
    grade: Optional[Any] = None
    ai_grade_suggestion: Optional[Any] = None
    remediation_plan: Optional[Any] = None
    graded_by: Optional[uuid.UUID] = None
    graded_at: Optional[datetime] = None
    created_at: datetime
    student_name: Optional[str] = None

    model_config = {"from_attributes": True}


class RubricCreate(BaseModel):
    title: str
    board: str
    grade: int
    subject: str
    criteria: List[Any]


class RubricUpdate(BaseModel):
    title: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[int] = None
    subject: Optional[str] = None
    criteria: Optional[List[Any]] = None


class RubricResponse(BaseModel):
    id: uuid.UUID
    title: str
    board: str
    grade: int
    subject: str
    criteria: Any
    created_by: uuid.UUID
    is_ai_generated: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LessonPlanRequest(BaseModel):
    class_id: str
    topic: str
    additional_context: Optional[str] = None


class LessonPlanResponse(BaseModel):
    id: uuid.UUID
    class_id: uuid.UUID
    title: str
    topic: str
    objectives: Optional[Any] = None
    time_estimate: Optional[int] = None
    steps: Optional[Any] = None
    practice_tasks: Optional[Any] = None
    formative_check: Optional[str] = None
    homework: Optional[str] = None
    differentiation: Optional[Any] = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AnnouncementCreate(BaseModel):
    class_id: str
    content: str
    allow_comments: bool = True


class AnnouncementResponse(BaseModel):
    id: uuid.UUID
    class_id: uuid.UUID
    author_id: uuid.UUID
    content: str
    allow_comments: bool
    created_at: datetime
    author_name: Optional[str] = None
    comment_count: int = 0

    model_config = {"from_attributes": True}


class CommentCreate(BaseModel):
    content: str


class CommentResponse(BaseModel):
    id: uuid.UUID
    announcement_id: uuid.UUID
    author_id: uuid.UUID
    content: str
    created_at: datetime
    author_name: Optional[str] = None

    model_config = {"from_attributes": True}


class QuizCreate(BaseModel):
    class_id: str
    title: str
    questions: List[Any]
    time_limit: Optional[int] = None


class QuizResponse(BaseModel):
    id: uuid.UUID
    class_id: uuid.UUID
    title: str
    questions: Any
    time_limit: Optional[int] = None
    is_published: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class QuizAttemptSubmit(BaseModel):
    responses: dict  # {questionId: answer}


class QuizAttemptResponse(BaseModel):
    id: uuid.UUID
    quiz_id: uuid.UUID
    student_id: uuid.UUID
    score: Optional[float] = None
    max_score: Optional[float] = None
    xp_earned: int
    started_at: datetime
    submitted_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class GenerateRubricRequest(BaseModel):
    board: str
    grade: int
    subject: str
    topic: str
    criteria_count: int = 4


class SuggestQuestionsRequest(BaseModel):
    class_id: str
    topic: str
    question_types: Optional[List[str]] = None
    count: int = 5


class AutoGradeRequest(BaseModel):
    submission_id: str
    rubric_id: str


class GradebookEntry(BaseModel):
    student_id: str
    student_name: str
    assignments: List[dict]
    average_score: float
    total_xp: int
