import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, status, Query
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.evaluation import (
    EvaluationQuestionPaper,
    EvaluationPaperSubject,
    EvaluationPaperChapter,
    EvaluationQuestion,
    EvaluationAssessment,
    EvaluationInvitation,
    EvaluationAttempt,
)
from app.schemas.evaluation import (
    EvalPaperCreate,
    EvalPaperResponse,
    EvalSubjectCreate,
    EvalChapterCreate,
    EvalQuestionCreate,
    EvalQuestionUpdate,
    EvalQuestionResponse,
    GeneratePaperRequest,
    EvalAssessmentCreate,
    EvalAssessmentResponse,
    DistributeAssessmentRequest,
    EvalAttemptSubmit,
    EvalAttemptResponse,
)
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.ai_service import AIService

router = APIRouter()


# ---- Question Papers ----

@router.post("/papers", response_model=EvalPaperResponse, status_code=status.HTTP_201_CREATED)
async def create_paper(
    payload: EvalPaperCreate,
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    paper = EvaluationQuestionPaper(
        org_id=uuid.UUID(org_id),
        created_by=current_user.id,
        title=payload.title,
        board=payload.board,
        grade=payload.grade,
        total_marks=payload.total_marks,
        negative_marking=payload.negative_marking,
        negative_mark_value=payload.negative_mark_value,
        time_limit=payload.time_limit,
        mode=payload.mode,
        status="draft",
    )
    db.add(paper)
    await db.commit()
    await db.refresh(paper)
    return paper


@router.get("/papers", response_model=list[EvalPaperResponse])
async def list_papers(
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    result = await db.execute(
        select(EvaluationQuestionPaper)
        .where(EvaluationQuestionPaper.org_id == uuid.UUID(org_id))
        .order_by(EvaluationQuestionPaper.created_at.desc())
    )
    return result.scalars().all()


@router.get("/papers/{paper_id}", response_model=EvalPaperResponse)
async def get_paper(paper_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(EvaluationQuestionPaper).where(EvaluationQuestionPaper.id == paper_id)
    )
    paper = result.scalar_one_or_none()
    if not paper:
        raise NotFoundException("Question paper not found")
    return paper


# ---- Questions ----

@router.post("/papers/{paper_id}/questions", response_model=EvalQuestionResponse, status_code=status.HTTP_201_CREATED)
async def add_question(
    paper_id: uuid.UUID, payload: EvalQuestionCreate, current_user: CurrentUser, db: DBSession
):
    question = EvaluationQuestion(
        paper_id=paper_id,
        question_type=payload.question_type,
        question_text=payload.question_text,
        options=payload.options,
        correct_answer=payload.correct_answer,
        marks=payload.marks,
        negative_marks=payload.negative_marks,
        subject=payload.subject,
        chapter=payload.chapter,
        difficulty=payload.difficulty,
        explanation=payload.explanation,
        tags=payload.tags,
        is_ai_generated=False,
    )
    db.add(question)
    await db.commit()
    await db.refresh(question)
    return question


@router.get("/papers/{paper_id}/questions", response_model=list[EvalQuestionResponse])
async def list_questions(
    paper_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
    subject: str | None = Query(None),
    question_type: str | None = Query(None),
    limit: int = Query(200, le=500),
):
    q = select(EvaluationQuestion).where(EvaluationQuestion.paper_id == paper_id)
    if subject:
        q = q.where(EvaluationQuestion.subject == subject)
    if question_type:
        q = q.where(EvaluationQuestion.question_type == question_type)
    q = q.order_by(EvaluationQuestion.order_index).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.patch("/questions/{question_id}", response_model=EvalQuestionResponse)
async def update_question(
    question_id: uuid.UUID, payload: EvalQuestionUpdate, current_user: CurrentUser, db: DBSession
):
    result = await db.execute(
        select(EvaluationQuestion).where(EvaluationQuestion.id == question_id)
    )
    question = result.scalar_one_or_none()
    if not question:
        raise NotFoundException("Question not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(question, key, value)
    await db.commit()
    await db.refresh(question)
    return question


@router.delete("/questions/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_question(question_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(EvaluationQuestion).where(EvaluationQuestion.id == question_id)
    )
    question = result.scalar_one_or_none()
    if not question:
        raise NotFoundException("Question not found")
    await db.delete(question)
    await db.commit()


@router.post("/papers/{paper_id}/generate-questions")
async def ai_generate_questions(
    paper_id: uuid.UUID, payload: GeneratePaperRequest, current_user: CurrentUser, db: DBSession
):
    """Use AI to generate questions for the question bank."""
    ai = AIService()
    questions = await ai.generate_evaluation_paper(
        paper_id=str(paper_id),
        subjects=payload.subjects,
        question_types=payload.question_types,
    )
    new_questions = []
    for q_data in questions:
        question = EvaluationQuestion(
            paper_id=paper_id,
            question_type=q_data.get("type"),
            question_text=q_data.get("text"),
            options=q_data.get("options"),
            correct_answer=q_data.get("correct_answer"),
            marks=q_data.get("marks", 1.0),
            subject=q_data.get("subject"),
            chapter=q_data.get("chapter"),
            difficulty=q_data.get("difficulty"),
            explanation=q_data.get("explanation"),
            is_ai_generated=True,
        )
        db.add(question)
        new_questions.append(question)
    await db.commit()
    return {"generated": len(new_questions), "message": "Questions generated successfully"}


# ---- Assessments ----

@router.post("/assessments", response_model=EvalAssessmentResponse, status_code=status.HTTP_201_CREATED)
async def create_assessment(
    payload: EvalAssessmentCreate,
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    assessment = EvaluationAssessment(
        paper_id=uuid.UUID(payload.paper_id),
        org_id=uuid.UUID(org_id),
        created_by=current_user.id,
        title=payload.title,
        mode=payload.mode,
        time_limit=payload.time_limit,
        negative_marking=payload.negative_marking,
        scheduled_at=payload.scheduled_at,
        ends_at=payload.ends_at,
        status="draft",
    )
    db.add(assessment)
    await db.commit()
    await db.refresh(assessment)
    return assessment


@router.get("/assessments", response_model=list[EvalAssessmentResponse])
async def list_assessments(
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    result = await db.execute(
        select(EvaluationAssessment)
        .where(EvaluationAssessment.org_id == uuid.UUID(org_id))
        .order_by(EvaluationAssessment.created_at.desc())
    )
    return result.scalars().all()


@router.post("/assessments/{assessment_id}/distribute")
async def distribute_assessment(
    assessment_id: uuid.UUID, payload: DistributeAssessmentRequest, current_user: CurrentUser, db: DBSession
):
    """Distribute an assessment to specific classes or individual students."""
    from app.models.classes import ClassStudent
    invitations = []
    if payload.class_ids:
        for class_id in payload.class_ids:
            students_result = await db.execute(
                select(ClassStudent).where(ClassStudent.class_id == uuid.UUID(class_id))
            )
            students = students_result.scalars().all()
            for student in students:
                inv = EvaluationInvitation(
                    assessment_id=assessment_id,
                    student_id=student.student_id,
                    class_id=uuid.UUID(class_id),
                )
                db.add(inv)
                invitations.append(student.student_id)

    if payload.student_ids:
        for student_id in payload.student_ids:
            inv = EvaluationInvitation(
                assessment_id=assessment_id,
                student_id=uuid.UUID(student_id),
            )
            db.add(inv)
            invitations.append(student_id)

    # Update assessment status
    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if assessment:
        assessment.status = "distributed"

    await db.commit()
    return {"distributed_to": len(invitations), "message": "Assessment distributed"}


@router.get("/my-assessments", response_model=list[EvalAssessmentResponse])
async def get_my_eval_assessments(current_user: CurrentUser, db: DBSession):
    """Get evaluation assessments the current student is invited to."""
    result = await db.execute(
        select(EvaluationAssessment)
        .join(EvaluationInvitation, EvaluationInvitation.assessment_id == EvaluationAssessment.id)
        .where(
            EvaluationInvitation.student_id == current_user.id,
            EvaluationInvitation.status == "pending",
        )
    )
    return result.scalars().all()


@router.post("/assessments/{assessment_id}/attempt/start", response_model=EvalAttemptResponse)
async def start_eval_attempt(assessment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    attempt = EvaluationAttempt(
        assessment_id=assessment_id,
        student_id=current_user.id,
        status="in_progress",
    )
    db.add(attempt)

    # Mark invitation as accepted
    inv_result = await db.execute(
        select(EvaluationInvitation).where(
            EvaluationInvitation.assessment_id == assessment_id,
            EvaluationInvitation.student_id == current_user.id,
        )
    )
    inv = inv_result.scalar_one_or_none()
    if inv:
        inv.status = "accepted"

    await db.commit()
    await db.refresh(attempt)
    return attempt


@router.post("/assessments/{assessment_id}/attempt/{attempt_id}/submit", response_model=EvalAttemptResponse)
async def submit_eval_attempt(
    assessment_id: uuid.UUID,
    attempt_id: uuid.UUID,
    payload: EvalAttemptSubmit,
    current_user: CurrentUser,
    db: DBSession,
):
    attempt_result = await db.execute(
        select(EvaluationAttempt).where(
            EvaluationAttempt.id == attempt_id,
            EvaluationAttempt.student_id == current_user.id,
            EvaluationAttempt.status == "in_progress",
        )
    )
    attempt = attempt_result.scalar_one_or_none()
    if not attempt:
        raise NotFoundException("Attempt not found or already submitted")

    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()

    questions_result = await db.execute(
        select(EvaluationQuestion).where(EvaluationQuestion.paper_id == assessment.paper_id)
    )
    questions = questions_result.scalars().all()

    score = 0
    max_score = sum(q.marks for q in questions)
    for q in questions:
        student_answer = payload.responses.get(str(q.id))
        if student_answer and str(student_answer).strip().lower() == str(q.correct_answer or "").strip().lower():
            score += q.marks
        elif student_answer and assessment.negative_marking:
            score -= q.negative_marks

    attempt.responses = payload.responses
    attempt.score = max(0, score)
    attempt.max_score = max_score
    attempt.percentage = (attempt.score / max_score * 100) if max_score else 0
    attempt.submitted_at = datetime.now(timezone.utc)
    attempt.status = "submitted"

    inv_result = await db.execute(
        select(EvaluationInvitation).where(
            EvaluationInvitation.assessment_id == assessment_id,
            EvaluationInvitation.student_id == current_user.id,
        )
    )
    inv = inv_result.scalar_one_or_none()
    if inv:
        inv.status = "completed"

    await db.commit()
    await db.refresh(attempt)
    return attempt
