import uuid
import tempfile
from datetime import datetime, timezone
from fastapi import APIRouter, status, Query, UploadFile, File, HTTPException
from sqlalchemy import select, distinct, union_all
from sqlalchemy.orm import selectinload

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
    GenerateEvalPaperRequest,
    GenerateEvalPaperResponse,
    SaveEvalPaperRequest,
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

@router.post("/papers/generate", response_model=GenerateEvalPaperResponse)
async def generate_paper(
    payload: GenerateEvalPaperRequest,
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """AI-generate questions for a new paper. Does NOT persist — frontend reviews first."""
    ai = AIService()

    # Build subjects list for AI service
    subjects_for_ai = []
    for s in payload.subjects:
        subj = {
            "subject": s.subject,
            "source_type": s.source_type,
            "source_text": s.source_text,
            "chapters": s.chapters or [],
        }
        subjects_for_ai.append(subj)

    questions, answer_key = await ai.generate_evaluation_paper(
        subjects=subjects_for_ai,
        question_types=payload.question_types,
        difficulty=payload.difficulty,
        blooms_level=payload.blooms_level,
        question_count=payload.question_count,
        grade=payload.grade,
        board=payload.board,
        mcq_subtypes=payload.mcq_subtypes,
        type_weightage=payload.type_weightage,
        negative_marking=payload.negative_marking,
    )

    return GenerateEvalPaperResponse(question_json=questions, answer_key_json=answer_key)


@router.post("/papers/upload-source")
async def upload_source_file(
    file: UploadFile = File(...),
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Upload a file and extract text for use as question source material."""
    allowed_types = {
        "image/jpeg", "image/png", "image/gif", "image/webp",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain", "text/markdown",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Allowed: PDF, DOCX, TXT, MD, images.",
        )

    # Save to temp file for AI extraction
    content = await file.read()
    suffix = "." + (file.filename or "file").rsplit(".", 1)[-1] if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    ai = AIService()

    # For plain text files, just read the content directly
    if file.content_type in {"text/plain", "text/markdown"}:
        extracted_text = content.decode("utf-8", errors="replace")
    else:
        extracted_text = await ai.extract_text_from_file(file_path=tmp_path)

    # Clean up temp file
    import os
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    return {
        "extracted_text": extracted_text or "",
        "filename": file.filename or "uploaded_file",
        "word_count": len(extracted_text.split()) if extracted_text else 0,
    }


@router.post("/papers/save", response_model=EvalPaperResponse, status_code=status.HTTP_201_CREATED)
async def save_paper(
    payload: SaveEvalPaperRequest,
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Save a reviewed paper with its questions and answer key."""
    config = payload.config
    questions_data = payload.questions

    # Calculate total marks
    total_marks = sum(q.get("points", q.get("marks", 1)) for q in questions_data)

    paper = EvaluationQuestionPaper(
        org_id=uuid.UUID(payload.org_id),
        created_by=current_user.id,
        title=config.get("title", "Untitled Paper"),
        board=config.get("board"),
        grade=config.get("grade"),
        total_marks=int(total_marks),
        negative_marking=config.get("negativeMarking", False),
        negative_mark_value=config.get("negativeMarkValue", 0.25),
        time_limit=config.get("timeLimitSeconds"),
        mode=config.get("mode", "exam"),
        difficulty=config.get("difficulty"),
        question_count=len(questions_data),
        max_score=total_marks,
        status="draft",
        config={"answer_key": payload.answer_key, "original_config": config},
    )
    db.add(paper)
    await db.flush()

    # Create subjects
    subjects_config = config.get("subjects", [])
    for i, sc in enumerate(subjects_config):
        paper_subject = EvaluationPaperSubject(
            paper_id=paper.id,
            subject=sc.get("subject", ""),
            marks_allocated=None,
            order_index=i,
        )
        db.add(paper_subject)
        await db.flush()

        # Create chapters
        for ch in sc.get("chapters", []):
            db.add(EvaluationPaperChapter(
                paper_subject_id=paper_subject.id,
                chapter_name=ch.get("name", ""),
                weightage=ch.get("weightage", 1.0),
            ))

    # Create questions
    for i, q in enumerate(questions_data):
        # Determine source_type from the subject config
        q_subject = q.get("subject", "")
        source_type = "online"
        for sc in subjects_config:
            if sc.get("subject", "").lower() == q_subject.lower():
                source_type = sc.get("sourceType", "online")
                break

        # For match questions, store pairs alongside options in the JSONB field
        q_options = q.get("options")
        if q.get("type") == "match" and q.get("pairs"):
            q_options = {"options": q_options, "pairs": q.get("pairs")}

        question = EvaluationQuestion(
            paper_id=paper.id,
            question_type=q.get("type", "mcq"),
            question_text=q.get("text", ""),
            options=q_options,
            correct_answer=str(q.get("correctAnswer", "")) if q.get("correctAnswer") is not None else None,
            marks=q.get("points", q.get("marks", 1.0)),
            negative_marks=0.0,
            subject=q_subject,
            chapter=q.get("chapter"),
            difficulty=config.get("difficulty"),
            explanation=q.get("explanation"),
            source_type=source_type,
            blooms_level=q.get("blooms_level"),
            is_ai_generated=True,
            order_index=i,
        )
        db.add(question)

    await db.commit()
    await db.refresh(paper)
    return paper


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
        difficulty=payload.difficulty,
        question_count=payload.question_count,
        max_score=payload.max_score,
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


@router.delete("/papers/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(paper_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(EvaluationQuestionPaper).where(EvaluationQuestionPaper.id == paper_id)
    )
    paper = result.scalar_one_or_none()
    if not paper:
        raise NotFoundException("Question paper not found")
    await db.delete(paper)
    await db.commit()


@router.patch("/papers/{paper_id}", response_model=EvalPaperResponse)
async def update_paper(paper_id: uuid.UUID, payload: dict, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(EvaluationQuestionPaper).where(EvaluationQuestionPaper.id == paper_id)
    )
    paper = result.scalar_one_or_none()
    if not paper:
        raise NotFoundException("Question paper not found")
    allowed_fields = {"title", "status", "board", "grade", "total_marks", "time_limit", "difficulty"}
    for key, value in payload.items():
        if key in allowed_fields:
            setattr(paper, key, value)
    await db.commit()
    await db.refresh(paper)
    return paper


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

@router.get("/questions", response_model=list[EvalQuestionResponse])
async def list_all_questions(
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
    subject: str | None = Query(None),
    question_type: str | None = Query(None, alias="type"),
    difficulty: str | None = Query(None),
    source_type: str | None = Query(None, alias="source"),
    paper_id: str | None = Query(None),
    limit: int = Query(200, le=500),
):
    """Get questions across all papers in the org, with optional filters."""
    q = (
        select(EvaluationQuestion)
        .join(EvaluationQuestionPaper, EvaluationQuestion.paper_id == EvaluationQuestionPaper.id)
        .where(EvaluationQuestionPaper.org_id == uuid.UUID(org_id))
    )
    if paper_id:
        q = q.where(EvaluationQuestion.paper_id == uuid.UUID(paper_id))
    if subject:
        q = q.where(EvaluationQuestion.subject == subject)
    if question_type:
        q = q.where(EvaluationQuestion.question_type == question_type)
    if difficulty:
        q = q.where(EvaluationQuestion.difficulty == difficulty)
    if source_type:
        q = q.where(EvaluationQuestion.source_type == source_type)
    q = q.order_by(EvaluationQuestion.created_at.desc()).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/subjects")
async def list_subjects(
    org_id: str = Query(...),
    grade: int | None = Query(None),
    board: str | None = Query(None),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Get distinct subjects used across all papers in the org, optionally filtered by grade/board."""
    # From questions
    q1 = (
        select(distinct(EvaluationQuestion.subject))
        .join(EvaluationQuestionPaper, EvaluationQuestion.paper_id == EvaluationQuestionPaper.id)
        .where(
            EvaluationQuestionPaper.org_id == uuid.UUID(org_id),
            EvaluationQuestion.subject.isnot(None),
        )
    )
    if grade:
        q1 = q1.where(EvaluationQuestionPaper.grade == grade)
    if board:
        q1 = q1.where(EvaluationQuestionPaper.board == board)

    # From paper subjects
    q2 = (
        select(distinct(EvaluationPaperSubject.subject))
        .join(EvaluationQuestionPaper, EvaluationPaperSubject.paper_id == EvaluationQuestionPaper.id)
        .where(EvaluationQuestionPaper.org_id == uuid.UUID(org_id))
    )
    if grade:
        q2 = q2.where(EvaluationQuestionPaper.grade == grade)
    if board:
        q2 = q2.where(EvaluationQuestionPaper.board == board)

    result1 = await db.execute(q1)
    result2 = await db.execute(q2)
    subjects = set(result1.scalars().all()) | set(result2.scalars().all())
    subjects.discard(None)
    subjects.discard("")
    return sorted(subjects)


@router.get("/chapters")
async def list_chapters(
    org_id: str = Query(...),
    subject: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Get distinct chapters for a subject across all papers in the org."""
    # From questions
    q1 = (
        select(distinct(EvaluationQuestion.chapter))
        .join(EvaluationQuestionPaper, EvaluationQuestion.paper_id == EvaluationQuestionPaper.id)
        .where(
            EvaluationQuestionPaper.org_id == uuid.UUID(org_id),
            EvaluationQuestion.subject == subject,
            EvaluationQuestion.chapter.isnot(None),
        )
    )
    # From paper chapters
    q2 = (
        select(distinct(EvaluationPaperChapter.chapter_name))
        .join(EvaluationPaperSubject, EvaluationPaperChapter.paper_subject_id == EvaluationPaperSubject.id)
        .join(EvaluationQuestionPaper, EvaluationPaperSubject.paper_id == EvaluationQuestionPaper.id)
        .where(
            EvaluationQuestionPaper.org_id == uuid.UUID(org_id),
            EvaluationPaperSubject.subject == subject,
        )
    )
    result1 = await db.execute(q1)
    result2 = await db.execute(q2)
    chapters = set(result1.scalars().all()) | set(result2.scalars().all())
    chapters.discard(None)
    chapters.discard("")
    return sorted(chapters)


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
        source_type=payload.source_type,
        blooms_level=payload.blooms_level,
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
    questions, _ = await ai.generate_evaluation_paper(
        subjects=payload.subjects,
        question_types=payload.question_types,
    )
    new_questions = []
    for q_data in questions:
        q_options = q_data.get("options")
        if q_data.get("type") == "match" and q_data.get("pairs"):
            q_options = {"options": q_options, "pairs": q_data.get("pairs")}

        question = EvaluationQuestion(
            paper_id=paper_id,
            question_type=q_data.get("type"),
            question_text=q_data.get("text"),
            options=q_options,
            correct_answer=q_data.get("correct_answer"),
            marks=q_data.get("marks", 1.0),
            subject=q_data.get("subject"),
            chapter=q_data.get("chapter"),
            difficulty=q_data.get("difficulty"),
            explanation=q_data.get("explanation"),
            source_type=q_data.get("source_type", "online"),
            blooms_level=q_data.get("blooms_level"),
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
    # Resolve time_limit: prefer time_limit_seconds (from frontend), fall back to time_limit (minutes)
    time_limit = payload.time_limit
    if payload.time_limit_seconds is not None:
        time_limit = payload.time_limit_seconds

    assessment = EvaluationAssessment(
        paper_id=uuid.UUID(payload.paper_id) if payload.paper_id else None,
        org_id=uuid.UUID(org_id),
        created_by=current_user.id,
        title=payload.title,
        mode=payload.mode,
        time_limit=time_limit,
        negative_marking=payload.negative_marking,
        negative_mark_value=payload.negative_mark_value,
        question_count=payload.question_count,
        max_score=payload.max_score,
        difficulty=payload.difficulty,
        grade=payload.grade,
        board=payload.board,
        due_date=payload.due_date,
        question_ids=[str(qid) for qid in payload.question_ids] if payload.question_ids else None,
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


@router.patch("/assessments/{assessment_id}/status", response_model=EvalAssessmentResponse)
async def update_assessment_status(
    assessment_id: uuid.UUID,
    payload: dict,
    current_user: CurrentUser,
    db: DBSession,
):
    result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == assessment_id)
    )
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")
    new_status = payload.get("status")
    if new_status and new_status in ("draft", "active", "completed"):
        assessment.status = new_status
    await db.commit()
    await db.refresh(assessment)
    return assessment


@router.delete("/assessments/{assessment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assessment(assessment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == assessment_id)
    )
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")
    await db.delete(assessment)
    await db.commit()


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

    # Fetch questions: prefer question_ids (explicit list), fall back to paper_id
    if assessment.question_ids:
        questions_result = await db.execute(
            select(EvaluationQuestion).where(EvaluationQuestion.id.in_(
                [uuid.UUID(qid) if isinstance(qid, str) else qid for qid in assessment.question_ids]
            ))
        )
    elif assessment.paper_id:
        questions_result = await db.execute(
            select(EvaluationQuestion).where(EvaluationQuestion.paper_id == assessment.paper_id)
        )
    else:
        raise HTTPException(status_code=400, detail="Assessment has no questions configured")
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
