import uuid
import copy
import secrets
import random
import tempfile
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, status, Query, UploadFile, File, HTTPException
from sqlalchemy import select, delete, distinct, union_all, func
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import flag_modified

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
from app.models.organization import Organization, OrgMember
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
    PublicAssessmentInfoResponse,
    PublicStartAttemptRequest,
    PublicAttemptResponse,
    PublicAutosaveRequest,
    PublicSubmitRequest,
    PublicSubmitResponse,
    MyEvalResultResponse,
)
from app.core.exceptions import NotFoundException, ForbiddenException
from app.services.ai_service import AIService

router = APIRouter()


def _score_attempt(attempt, questions, responses, assessment):
    """Score an attempt's responses against correct answers. Mutates attempt in-place."""
    # Get shuffle option_maps for reverse-mapping MCQ answers
    option_maps = (attempt.attempt_metadata or {}).get("shuffle_map", {}).get("option_maps", {})

    score = 0
    max_score = sum(q.marks for q in questions)
    for q in questions:
        student_answer = responses.get(str(q.id))
        # Reverse-map shuffled MCQ answer back to original key
        q_map = option_maps.get(str(q.id))
        if q_map and q.question_type == "mcq" and student_answer:
            student_answer = q_map.get(student_answer, student_answer)
        if student_answer and str(student_answer).strip().lower() == str(q.correct_answer or "").strip().lower():
            score += q.marks
        elif student_answer and assessment.negative_marking:
            penalty = q.negative_marks if q.negative_marks else (assessment.negative_mark_value or 0.25)
            score -= penalty
    attempt.responses = responses
    attempt.score = max(0, score)
    attempt.max_score = max_score
    attempt.percentage = (attempt.score / max_score * 100) if max_score else 0


async def _fetch_assessment_questions(assessment, db):
    """Fetch questions for an assessment by question_ids or paper_id."""
    if assessment.question_ids:
        result = await db.execute(
            select(EvaluationQuestion).where(EvaluationQuestion.id.in_(
                [uuid.UUID(qid) if isinstance(qid, str) else qid for qid in assessment.question_ids]
            ))
        )
    elif assessment.paper_id:
        result = await db.execute(
            select(EvaluationQuestion).where(EvaluationQuestion.paper_id == assessment.paper_id)
        )
    else:
        raise HTTPException(status_code=400, detail="Assessment has no questions configured")
    return result.scalars().all()


logger = logging.getLogger(__name__)


def _shuffle_questions_and_options(questions_data: list[dict]) -> dict:
    """Shuffle question order and MCQ option keys. Returns shuffle_map to store in attempt_metadata."""
    # Deep copy options dicts to avoid mutating cached SQLAlchemy objects
    for q in questions_data:
        if isinstance(q.get("options"), dict):
            q["options"] = copy.deepcopy(q["options"])

    random.shuffle(questions_data)
    option_maps = {}
    for q in questions_data:
        if q.get("question_type") != "mcq" or not isinstance(q.get("options"), dict):
            continue
        orig_keys = sorted(q["options"].keys())  # e.g. ['A', 'B', 'C', 'D']
        shuffled_keys = orig_keys.copy()
        random.shuffle(shuffled_keys)
        new_options = {}
        new_to_orig = {}
        for new_idx, orig_key in enumerate(shuffled_keys):
            new_key = orig_keys[new_idx]  # Assign to A, B, C, D in order
            new_options[new_key] = q["options"][orig_key]
            new_to_orig[new_key] = orig_key
        q["options"] = new_options
        option_maps[q["id"]] = new_to_orig
    # Update order_index
    for idx, q in enumerate(questions_data):
        q["order_index"] = idx

    result = {
        "question_order": [q["id"] for q in questions_data],
        "option_maps": option_maps,
    }
    logger.info("Shuffle result - question_order: %s, option_maps keys: %s", result["question_order"][:3], list(option_maps.keys())[:3])
    return result


def _apply_shuffle_to_questions(questions_data: list[dict], shuffle_map: dict) -> list[dict]:
    """Reapply a stored shuffle_map to questions_data (for resume). Returns reordered list."""
    question_order = shuffle_map.get("question_order", [])
    option_maps = shuffle_map.get("option_maps", {})
    if not question_order:
        return questions_data
    # Reorder questions
    q_by_id = {q["id"]: q for q in questions_data}
    ordered = [q_by_id[qid] for qid in question_order if qid in q_by_id]
    # Add any questions not in the map (shouldn't happen, but safety)
    seen = set(question_order)
    for q in questions_data:
        if q["id"] not in seen:
            ordered.append(q)
    # Reapply option shuffles
    for q in ordered:
        q_map = option_maps.get(q["id"])
        if not q_map or not isinstance(q.get("options"), dict):
            continue
        orig_keys = sorted(q["options"].keys())
        # q_map is {new_key: orig_key}, so we need to rebuild options
        # The stored options are in original order, we need to remap
        orig_options = q["options"].copy()
        # Reverse: orig_to_new mapping
        orig_to_new = {v: k for k, v in q_map.items()}
        new_options = {}
        for orig_key in orig_keys:
            new_key = orig_to_new.get(orig_key, orig_key)
            new_options[new_key] = orig_options[orig_key]
        q["options"] = new_options
    # Update order_index
    for idx, q in enumerate(ordered):
        q["order_index"] = idx
    return ordered


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

    try:
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
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned no questions. Please try again.",
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

@router.get("/questions")
async def list_all_questions(
    org_id: str = Query(...),
    current_user: CurrentUser = None,
    db: DBSession = None,
    subject: str | None = Query(None),
    question_type: str | None = Query(None, alias="type"),
    difficulty: str | None = Query(None),
    source_type: str | None = Query(None, alias="source"),
    grade: int | None = Query(None),
    board: str | None = Query(None),
    blooms_level: str | None = Query(None),
    paper_id: str | None = Query(None),
    limit: int = Query(200, le=500),
):
    """Get questions across all papers in the org, with optional filters."""
    q = (
        select(EvaluationQuestion, EvaluationQuestionPaper.grade, EvaluationQuestionPaper.board)
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
    if grade:
        q = q.where(EvaluationQuestionPaper.grade == grade)
    if board:
        q = q.where(EvaluationQuestionPaper.board == board)
    if blooms_level:
        q = q.where(EvaluationQuestion.blooms_level == blooms_level)
    q = q.order_by(EvaluationQuestion.created_at.desc()).limit(limit)
    result = await db.execute(q)
    rows = result.all()
    # Attach grade and board from paper to each question response
    response = []
    for question, paper_grade, paper_board in rows:
        data = EvalQuestionResponse.model_validate(question)
        data.grade = paper_grade
        data.board = paper_board
        response.append(data)
    return response


@router.get("/subjects")
async def list_subjects(
    org_id: str = Query(...),
    grade: int | None = Query(None),
    board: str | None = Query(None),
    source_type: str | None = Query(None, alias="source"),
    current_user: CurrentUser = None,
    db: DBSession = None,
):
    """Get distinct subjects used across all papers in the org, optionally filtered by grade/board/source_type."""
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
    if source_type:
        q1 = q1.where(EvaluationQuestion.source_type == source_type)

    # From paper subjects — only include if no source_type filter (paper subjects don't have source_type)
    subjects: set[str] = set()
    result1 = await db.execute(q1)
    subjects |= set(result1.scalars().all())

    if not source_type:
        q2 = (
            select(distinct(EvaluationPaperSubject.subject))
            .join(EvaluationQuestionPaper, EvaluationPaperSubject.paper_id == EvaluationQuestionPaper.id)
            .where(EvaluationQuestionPaper.org_id == uuid.UUID(org_id))
        )
        if grade:
            q2 = q2.where(EvaluationQuestionPaper.grade == grade)
        if board:
            q2 = q2.where(EvaluationQuestionPaper.board == board)
        result2 = await db.execute(q2)
        subjects |= set(result2.scalars().all())

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
        max_attempts=payload.max_attempts,
        status="active",
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


@router.get("/assessments/{assessment_id}")
async def get_assessment(assessment_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(EvaluationAssessment)
        .where(EvaluationAssessment.id == assessment_id)
        .options(
            selectinload(EvaluationAssessment.invitations),
            selectinload(EvaluationAssessment.attempts),
        )
    )
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")

    # Build response with invitations and attempts data
    resp = EvalAssessmentResponse.model_validate(assessment)
    resp_dict = resp.model_dump(mode="json")
    resp_dict["invitations"] = [
        {
            "id": str(inv.id),
            "email": inv.email,
            "name": inv.name,
            "status": inv.status,
            "student_id": str(inv.student_id) if inv.student_id else None,
            "token": inv.token,
            "created_at": inv.created_at.isoformat() if inv.created_at else None,
            "reinvited_at": inv.reinvited_at.isoformat() if inv.reinvited_at else None,
        }
        for inv in assessment.invitations
    ]
    resp_dict["attempts"] = [
        {
            "id": str(a.id),
            "student_id": str(a.student_id) if a.student_id else None,
            "invitation_id": str(a.invitation_id) if a.invitation_id else None,
            "score": a.score,
            "max_score": a.max_score,
            "percentage": a.percentage,
            "started_at": a.started_at.isoformat() if a.started_at else None,
            "submitted_at": a.submitted_at.isoformat() if a.submitted_at else None,
            "status": a.status,
            "attempt_metadata": a.attempt_metadata,
        }
        for a in assessment.attempts
    ]

    # Fetch and include questions
    try:
        questions = await _fetch_assessment_questions(assessment, db)
        # Maintain order by question_ids if available
        if assessment.question_ids:
            id_order = {
                (uuid.UUID(qid) if isinstance(qid, str) else qid): idx
                for idx, qid in enumerate(assessment.question_ids)
            }
            questions = sorted(questions, key=lambda q: id_order.get(q.id, 9999))
        else:
            questions = sorted(questions, key=lambda q: q.order_index or 0)

        resp_dict["questions"] = [
            {
                "id": str(q.id),
                "text": q.question_text,
                "type": q.question_type,
                "options": q.options,
                "correct_answer": q.correct_answer,
                "points": q.marks,
                "subject": q.subject,
                "chapter": q.chapter,
                "difficulty": q.difficulty,
                "explanation": q.explanation,
                "tags": q.tags,
                "blooms_level": q.blooms_level,
                "order_index": q.order_index,
            }
            for q in questions
        ]
    except Exception:
        resp_dict["questions"] = []

    return resp_dict


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
    if new_status and new_status in ("active", "completed"):
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
    """Distribute an assessment: create per-email token invitations and send personalized links."""
    import traceback
    from app.models.classes import ClassStudent
    from app.models.user import User
    from app.services.email_service import send_assessment_invitation

    print(f"\n{'='*60}", flush=True)
    print(f"[Distribute] START assessment_id={assessment_id}", flush=True)

    # Fetch the assessment
    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    class_id_for_inv = uuid.UUID(payload.class_id) if payload.class_id else None

    # Collect all target emails
    all_emails: set[str] = set()

    if payload.emails:
        for e in payload.emails:
            if e and e.strip():
                all_emails.add(e.strip().lower())

    # Resolve class members to emails
    if payload.class_id or payload.class_ids:
        class_id_list = []
        if payload.class_id:
            class_id_list.append(payload.class_id)
        if payload.class_ids:
            class_id_list.extend(payload.class_ids)
        for cid in class_id_list:
            students_result = await db.execute(
                select(ClassStudent).where(ClassStudent.class_id == uuid.UUID(cid))
            )
            student_ids = [s.student_id for s in students_result.scalars().all()]
            if student_ids:
                email_result = await db.execute(
                    select(User.email).where(User.id.in_(student_ids))
                )
                for email in email_result.scalars().all():
                    if email:
                        all_emails.add(email.strip().lower())

    # Resolve student_ids to emails (legacy)
    if payload.student_ids:
        sid_uuids = [uuid.UUID(sid) for sid in payload.student_ids]
        email_result = await db.execute(
            select(User.email).where(User.id.in_(sid_uuids))
        )
        for email in email_result.scalars().all():
            if email:
                all_emails.add(email.strip().lower())

    print(f"[Distribute] Total unique emails: {len(all_emails)}", flush=True)

    # Check existing invitations by email
    existing_result = await db.execute(
        select(EvaluationInvitation).where(
            EvaluationInvitation.assessment_id == assessment_id
        )
    )
    existing_invitations = {inv.email.lower(): inv for inv in existing_result.scalars().all()}
    new_emails = all_emails - set(existing_invitations.keys())
    reinvite_emails = all_emails & set(existing_invitations.keys())
    print(f"[Distribute] existing={len(existing_invitations)}, new={len(new_emails)}, reinvite={len(reinvite_emails)}", flush=True)

    # Resolve registered users for student_id linking
    registered_map: dict[str, uuid.UUID] = {}  # email -> user_id
    all_lookup = new_emails | reinvite_emails
    if all_lookup:
        users_result = await db.execute(
            select(User).where(User.email.in_(list(all_lookup)))
        )
        for user in users_result.scalars().all():
            registered_map[user.email.lower()] = user.id

    # Create one invitation per new email with unique token
    email_token_pairs: list[tuple[str, str]] = []
    for email in sorted(new_emails):
        token = secrets.token_urlsafe(32)
        inv = EvaluationInvitation(
            assessment_id=assessment_id,
            student_id=registered_map.get(email),
            email=email,
            token=token,
            class_id=class_id_for_inv,
        )
        db.add(inv)
        email_token_pairs.append((email, token))

    # Re-invite existing emails: regenerate token, set reinvited_at (preserves old attempts)
    now = datetime.now(timezone.utc)
    for email in sorted(reinvite_emails):
        inv = existing_invitations[email]
        inv.token = secrets.token_urlsafe(32)
        inv.status = "pending"
        inv.reinvited_at = now
        email_token_pairs.append((email, inv.token))

    # Ensure assessment is active
    if assessment.status not in ("active", "completed"):
        assessment.status = "active"

    await db.commit()
    print(f"[Distribute] DB committed. {len(email_token_pairs)} new invitations created.", flush=True)

    # Send emails with per-recipient token links
    emails_sent = 0
    if email_token_pairs:
        teacher_name = current_user.name or "Your Teacher"
        # Convert due_date from UTC to IST (UTC+5:30) for display in email
        if assessment.due_date:
            ist = timezone(timedelta(hours=5, minutes=30))
            due_dt = assessment.due_date if assessment.due_date.tzinfo else assessment.due_date.replace(tzinfo=timezone.utc)
            due_str = due_dt.astimezone(ist).strftime("%d %b %Y, %I:%M %p")
        else:
            due_str = None
        time_limit = assessment.time_limit if assessment.time_limit else None

        try:
            await send_assessment_invitation(
                email_token_pairs=email_token_pairs,
                assessment_title=assessment.title,
                teacher_name=teacher_name,
                due_date=due_str,
                time_limit=time_limit,
            )
            emails_sent = len(email_token_pairs)
            print(f"[Distribute] SUCCESS: {emails_sent} emails sent!", flush=True)
        except Exception as e:
            print(f"[Distribute] FAILED to send emails: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    print(f"[Distribute] DONE\n{'='*60}\n", flush=True)

    return {
        "distributed_to": len(new_emails),
        "reinvited": len(reinvite_emails),
        "emails_sent": emails_sent,
        "message": "Assessment distributed",
    }


@router.post("/assessments/{assessment_id}/reinvite")
async def reinvite_assessment(
    assessment_id: uuid.UUID,
    payload: dict,
    current_user: CurrentUser,
    db: DBSession,
):
    """Re-invite specific or all students: regenerate tokens and resend emails."""
    import traceback
    from app.services.email_service import send_assessment_invitation

    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    # Get target invitations
    target_emails = payload.get("emails")  # list of emails, or None for all
    query = select(EvaluationInvitation).where(
        EvaluationInvitation.assessment_id == assessment_id
    )
    if target_emails:
        query = query.where(EvaluationInvitation.email.in_([e.lower() for e in target_emails]))

    inv_result = await db.execute(query)
    invitations = inv_result.scalars().all()

    if not invitations:
        raise HTTPException(status_code=404, detail="No invitations found")

    # Regenerate tokens, set reinvited_at (preserves old attempts history)
    email_token_pairs: list[tuple[str, str]] = []
    now = datetime.now(timezone.utc)

    for inv in invitations:
        inv.token = secrets.token_urlsafe(32)
        inv.status = "pending"
        inv.reinvited_at = now
        email_token_pairs.append((inv.email, inv.token))

    await db.commit()

    # Send emails
    emails_sent = 0
    if email_token_pairs:
        teacher_name = current_user.name or "Your Teacher"
        # Convert due_date from UTC to IST (UTC+5:30) for display in email
        if assessment.due_date:
            ist = timezone(timedelta(hours=5, minutes=30))
            due_dt = assessment.due_date if assessment.due_date.tzinfo else assessment.due_date.replace(tzinfo=timezone.utc)
            due_str = due_dt.astimezone(ist).strftime("%d %b %Y, %I:%M %p")
        else:
            due_str = None
        time_limit = assessment.time_limit if assessment.time_limit else None
        try:
            await send_assessment_invitation(
                email_token_pairs=email_token_pairs,
                assessment_title=assessment.title,
                teacher_name=teacher_name,
                due_date=due_str,
                time_limit=time_limit,
            )
            emails_sent = len(email_token_pairs)
        except Exception as e:
            print(f"[Reinvite] FAILED to send emails: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    return {
        "reinvited": len(email_token_pairs),
        "emails_sent": emails_sent,
        "message": f"Re-invited {len(email_token_pairs)} student(s)",
    }


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
    # Fetch assessment to get questions for shuffle
    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")

    # Fetch questions and generate shuffle map
    questions = await _fetch_assessment_questions(assessment, db)
    questions_data = [{"id": str(q.id), "question_type": q.question_type, "options": q.options} for q in questions]
    shuffle_map = _shuffle_questions_and_options(questions_data)

    attempt = EvaluationAttempt(
        assessment_id=assessment_id,
        student_id=current_user.id,
        status="in_progress",
        attempt_metadata={"shuffle_map": shuffle_map},
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

    questions = await _fetch_assessment_questions(assessment, db)
    _score_attempt(attempt, questions, payload.responses, assessment)
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


# ---- Student's own evaluation results (org-scoped) ----

@router.get("/my-results", response_model=list[MyEvalResultResponse])
async def get_my_eval_results(
    current_user: CurrentUser,
    db: DBSession,
    org_id: uuid.UUID | None = Query(default=None),
):
    """Return the current user's submitted evaluation results, scoped to orgs they belong to."""
    query = (
        select(
            EvaluationAttempt,
            EvaluationAssessment.title.label("assessment_title"),
            EvaluationAssessment.mode,
            EvaluationAssessment.org_id,
            EvaluationAssessment.difficulty,
            EvaluationAssessment.question_count,
            EvaluationAssessment.grade,
            EvaluationAssessment.board,
            Organization.name.label("org_name"),
        )
        .join(EvaluationAssessment, EvaluationAttempt.assessment_id == EvaluationAssessment.id)
        .join(Organization, EvaluationAssessment.org_id == Organization.id)
        .join(
            OrgMember,
            (OrgMember.org_id == EvaluationAssessment.org_id)
            & (OrgMember.user_id == current_user.id)
            & (OrgMember.status == "active"),
        )
        .where(
            EvaluationAttempt.student_id == current_user.id,
            EvaluationAttempt.status == "submitted",
        )
        .order_by(EvaluationAttempt.submitted_at.desc())
    )

    if org_id:
        query = query.where(EvaluationAssessment.org_id == org_id)

    result = await db.execute(query)
    rows = result.all()

    return [
        MyEvalResultResponse(
            attempt_id=row.EvaluationAttempt.id,
            assessment_id=row.EvaluationAttempt.assessment_id,
            assessment_title=row.assessment_title,
            mode=row.mode,
            score=row.EvaluationAttempt.score,
            max_score=row.EvaluationAttempt.max_score,
            percentage=row.EvaluationAttempt.percentage,
            status=row.EvaluationAttempt.status,
            started_at=row.EvaluationAttempt.started_at,
            submitted_at=row.EvaluationAttempt.submitted_at,
            org_id=row.org_id,
            org_name=row.org_name,
            difficulty=row.difficulty,
            question_count=row.question_count,
            grade=row.grade,
            board=row.board,
        )
        for row in rows
    ]


# ---- Public Token-Based Endpoints (no auth required) ----

@router.get("/public/assessment/{token}", response_model=PublicAssessmentInfoResponse)
async def public_assessment_info(token: str, db: DBSession = None):
    """Get assessment info by invitation token. No auth required."""
    inv_result = await db.execute(
        select(EvaluationInvitation).where(EvaluationInvitation.token == token)
    )
    invitation = inv_result.scalar_one_or_none()
    if not invitation:
        raise HTTPException(status_code=404, detail="Invalid or expired assessment link")

    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == invitation.assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    # Auto-submit any stale in_progress attempts for this invitation
    now = datetime.now(timezone.utc)
    stale_result = await db.execute(
        select(EvaluationAttempt).where(
            EvaluationAttempt.invitation_id == invitation.id,
            EvaluationAttempt.status == "in_progress",
        )
    )
    stale_attempts = stale_result.scalars().all()
    for stale in stale_attempts:
        # Check if this attempt has exceeded its time limit (+ 2 min grace)
        if assessment.time_limit:
            deadline = stale.started_at + timedelta(seconds=assessment.time_limit + 120)
        else:
            deadline = stale.started_at + timedelta(hours=24)
        if now > deadline:
            questions = await _fetch_assessment_questions(assessment, db)
            _score_attempt(stale, questions, stale.responses or {}, assessment)
            stale.submitted_at = now
            stale.status = "submitted"
            meta = dict(stale.attempt_metadata or {})
            meta["auto_submitted"] = True
            meta["submit_reason"] = "server_timeout"
            stale.attempt_metadata = meta
            flag_modified(stale, "attempt_metadata")

    if stale_attempts:
        # Also update invitation status if all attempts are now submitted
        all_submitted = all(a.status == "submitted" for a in stale_attempts)
        if all_submitted and invitation.status != "completed":
            invitation.status = "completed"
        await db.commit()

    # Count ALL attempts for this invitation (stale cleanup already ran above)
    attempt_query = select(func.count(EvaluationAttempt.id)).where(
        EvaluationAttempt.invitation_id == invitation.id,
    )
    if invitation.reinvited_at:
        attempt_query = attempt_query.where(
            EvaluationAttempt.started_at > invitation.reinvited_at
        )
    attempts_count_result = await db.execute(attempt_query)
    total_attempts = attempts_count_result.scalar() or 0

    # Count only in_progress attempts (active, not yet submitted)
    active_query = select(func.count(EvaluationAttempt.id)).where(
        EvaluationAttempt.invitation_id == invitation.id,
        EvaluationAttempt.status == "in_progress",
    )
    active_count_result = await db.execute(active_query)
    active_attempts = active_count_result.scalar() or 0

    # attempts_used = completed attempts (total minus any currently active)
    attempts_used = total_attempts - active_attempts

    # Determine if the user can attempt
    can_attempt = True
    if assessment.mode == "exam":
        can_attempt = attempts_used == 0 and active_attempts == 0
    elif assessment.mode == "practice" and assessment.max_attempts is not None:
        can_attempt = attempts_used < assessment.max_attempts and active_attempts == 0

    # Check if assessment has expired
    if assessment.ends_at and now > assessment.ends_at:
        can_attempt = False
    if assessment.due_date and now > assessment.due_date:
        can_attempt = False

    # Fetch organization name
    org_name = None
    if assessment.org_id:
        org_result = await db.execute(
            select(Organization.name).where(Organization.id == assessment.org_id)
        )
        org_name = org_result.scalar_one_or_none()

    return PublicAssessmentInfoResponse(
        assessment_title=assessment.title,
        mode=assessment.mode,
        question_count=assessment.question_count,
        time_limit=assessment.time_limit,
        negative_marking=assessment.negative_marking,
        due_date=assessment.due_date,
        max_attempts=assessment.max_attempts,
        attempts_used=attempts_used,
        can_attempt=can_attempt,
        invitation_status=invitation.status,
        email=invitation.email,
        organization_name=org_name,
    )


@router.post("/public/assessment/{token}/start", response_model=PublicAttemptResponse)
async def public_start_attempt(token: str, payload: PublicStartAttemptRequest, db: DBSession = None):
    """Start an assessment attempt via token link. No auth required."""
    inv_result = await db.execute(
        select(EvaluationInvitation).where(EvaluationInvitation.token == token)
    )
    invitation = inv_result.scalar_one_or_none()
    if not invitation:
        raise HTTPException(status_code=404, detail="Invalid or expired assessment link")

    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == invitation.assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    # Validate timing
    now = datetime.now(timezone.utc)
    if assessment.ends_at and now > assessment.ends_at:
        raise HTTPException(status_code=403, detail="This assessment has expired")
    if assessment.due_date and now > assessment.due_date:
        raise HTTPException(status_code=403, detail="This assessment is past its due date")

    # Store name on invitation if provided
    if payload.name and not invitation.name:
        invitation.name = payload.name

    # Auto-submit any stale in_progress attempts first (timed-out or abandoned)
    stale_result = await db.execute(
        select(EvaluationAttempt).where(
            EvaluationAttempt.invitation_id == invitation.id,
            EvaluationAttempt.status == "in_progress",
        ).order_by(EvaluationAttempt.started_at.desc())
    )
    in_progress_attempts = stale_result.scalars().all()

    existing_attempt = None
    for ip_attempt in in_progress_attempts:
        # Determine if this attempt is stale (exceeded time limit + grace period)
        if assessment.time_limit:
            deadline = ip_attempt.started_at + timedelta(seconds=assessment.time_limit + 120)
        else:
            deadline = ip_attempt.started_at + timedelta(hours=24)

        if now > deadline:
            # Auto-submit stale attempt
            questions_for_score = await _fetch_assessment_questions(assessment, db)
            _score_attempt(ip_attempt, questions_for_score, ip_attempt.responses or {}, assessment)
            ip_attempt.submitted_at = now
            ip_attempt.status = "submitted"
            meta = dict(ip_attempt.attempt_metadata or {})
            meta["auto_submitted"] = True
            meta["submit_reason"] = "server_timeout"
            ip_attempt.attempt_metadata = meta
            flag_modified(ip_attempt, "attempt_metadata")
            logger.info("Auto-submitted stale attempt %s", ip_attempt.id)
        elif existing_attempt is None:
            # Most recent non-stale in_progress attempt = candidate for resume
            existing_attempt = ip_attempt

    # Commit any auto-submitted stale attempts
    if any(a.status == "submitted" for a in in_progress_attempts):
        await db.commit()

    # Count submitted attempts for this invitation (stale cleanup already ran above)
    attempt_query = select(func.count(EvaluationAttempt.id)).where(
        EvaluationAttempt.invitation_id == invitation.id,
        EvaluationAttempt.status == "submitted",
    )
    if invitation.reinvited_at:
        attempt_query = attempt_query.where(
            EvaluationAttempt.started_at > invitation.reinvited_at
        )
    attempts_count_result = await db.execute(attempt_query)
    attempts_used = attempts_count_result.scalar() or 0
    logger.info("Attempt limits check: invitation=%s, submitted=%d, existing_in_progress=%s, mode=%s, max_attempts=%s",
                invitation.id, attempts_used, existing_attempt is not None, assessment.mode, assessment.max_attempts)

    # Enforce attempt limits only if there's no in-progress attempt to resume
    if not existing_attempt:
        if assessment.mode == "exam" and attempts_used > 0:
            raise HTTPException(status_code=403, detail="You have already taken this exam")
        if assessment.mode == "practice" and assessment.max_attempts is not None and attempts_used >= assessment.max_attempts:
            raise HTTPException(status_code=403, detail="Maximum attempts reached")

    # Fetch questions WITHOUT correct_answer/explanation
    if assessment.question_ids:
        questions_result = await db.execute(
            select(EvaluationQuestion).where(EvaluationQuestion.id.in_(
                [uuid.UUID(qid) if isinstance(qid, str) else qid for qid in assessment.question_ids]
            )).order_by(EvaluationQuestion.order_index)
        )
    elif assessment.paper_id:
        questions_result = await db.execute(
            select(EvaluationQuestion)
            .where(EvaluationQuestion.paper_id == assessment.paper_id)
            .order_by(EvaluationQuestion.order_index)
        )
    else:
        raise HTTPException(status_code=400, detail="Assessment has no questions configured")

    questions = questions_result.scalars().all()

    # Strip correct answers and explanations
    questions_data = []
    for q in questions:
        q_dict = {
            "id": str(q.id),
            "question_type": q.question_type,
            "question_text": q.question_text,
            "options": q.options,
            "marks": q.marks,
            "negative_marks": q.negative_marks,
            "subject": q.subject,
            "chapter": q.chapter,
            "difficulty": q.difficulty,
            "order_index": q.order_index,
            "attachment_url": q.attachment_url,
            "attachment_name": q.attachment_name,
        }
        questions_data.append(q_dict)

    if existing_attempt:
        # Resume: reapply stored shuffle from attempt_metadata
        shuffle_map = (existing_attempt.attempt_metadata or {}).get("shuffle_map")
        if shuffle_map:
            questions_data = _apply_shuffle_to_questions(questions_data, shuffle_map)

        return PublicAttemptResponse(
            attempt_id=existing_attempt.id,
            assessment_id=assessment.id,
            questions=questions_data,
            time_limit=assessment.time_limit,
            started_at=existing_attempt.started_at,
            responses=existing_attempt.responses,
        )

    # Create new attempt with shuffled questions
    shuffle_map = _shuffle_questions_and_options(questions_data)

    attempt = EvaluationAttempt(
        assessment_id=assessment.id,
        student_id=invitation.student_id,
        invitation_id=invitation.id,
        status="in_progress",
        attempt_metadata={"shuffle_map": shuffle_map},
    )
    db.add(attempt)

    # Update invitation status
    invitation.status = "accepted"

    await db.commit()
    await db.refresh(attempt)

    return PublicAttemptResponse(
        attempt_id=attempt.id,
        assessment_id=assessment.id,
        questions=questions_data,
        time_limit=assessment.time_limit,
        started_at=attempt.started_at,
    )


@router.post("/public/assessment/{token}/attempt/{attempt_id}/submit", response_model=PublicSubmitResponse)
async def public_submit_attempt(
    token: str, attempt_id: uuid.UUID, payload: PublicSubmitRequest, db: DBSession = None
):
    """Submit answers for a public token-based attempt. No auth required."""
    # Validate token
    inv_result = await db.execute(
        select(EvaluationInvitation).where(EvaluationInvitation.token == token)
    )
    invitation = inv_result.scalar_one_or_none()
    if not invitation:
        raise HTTPException(status_code=404, detail="Invalid assessment link")

    # Validate attempt belongs to this invitation
    attempt_result = await db.execute(
        select(EvaluationAttempt).where(
            EvaluationAttempt.id == attempt_id,
            EvaluationAttempt.invitation_id == invitation.id,
            EvaluationAttempt.status == "in_progress",
        )
    )
    attempt = attempt_result.scalar_one_or_none()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found or already submitted")

    # Fetch assessment
    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == attempt.assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    now = datetime.now(timezone.utc)

    # Fetch questions, score, and finalize
    questions = await _fetch_assessment_questions(assessment, db)
    _score_attempt(attempt, questions, payload.responses, assessment)
    attempt.submitted_at = now
    attempt.status = "submitted"
    if payload.metadata:
        # Merge payload metadata with existing (preserves shuffle_map)
        meta = dict(attempt.attempt_metadata or {})
        meta.update(payload.metadata)
        attempt.attempt_metadata = meta
        flag_modified(attempt, "attempt_metadata")

    # Update invitation status
    invitation.status = "completed"

    await db.commit()

    return PublicSubmitResponse(
        attempt_id=attempt.id,
        status="submitted",
        message="Your response has been submitted successfully.",
    )


@router.patch("/public/assessment/{token}/attempt/{attempt_id}/autosave")
async def public_autosave_attempt(
    token: str, attempt_id: uuid.UUID, payload: PublicAutosaveRequest, db: DBSession = None
):
    """Auto-save responses periodically. No scoring, no status change. Fast."""
    inv_result = await db.execute(
        select(EvaluationInvitation).where(EvaluationInvitation.token == token)
    )
    invitation = inv_result.scalar_one_or_none()
    if not invitation:
        raise HTTPException(status_code=404, detail="Invalid assessment link")

    attempt_result = await db.execute(
        select(EvaluationAttempt).where(
            EvaluationAttempt.id == attempt_id,
            EvaluationAttempt.invitation_id == invitation.id,
            EvaluationAttempt.status == "in_progress",
        )
    )
    attempt = attempt_result.scalar_one_or_none()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found or already submitted")

    attempt.responses = payload.responses
    # Merge autosave timestamp into metadata
    meta = dict(attempt.attempt_metadata or {})
    meta["last_autosave_at"] = datetime.now(timezone.utc).isoformat()
    attempt.attempt_metadata = meta
    flag_modified(attempt, "attempt_metadata")

    await db.commit()
    return {"status": "saved"}


@router.post("/public/assessment/{token}/attempt/{attempt_id}/beacon-submit")
async def public_beacon_submit(
    token: str, attempt_id: uuid.UUID, payload: PublicSubmitRequest, db: DBSession = None
):
    """Beacon-based submit on browser close. Idempotent — silently succeeds if already submitted."""
    inv_result = await db.execute(
        select(EvaluationInvitation).where(EvaluationInvitation.token == token)
    )
    invitation = inv_result.scalar_one_or_none()
    if not invitation:
        raise HTTPException(status_code=404, detail="Invalid assessment link")

    attempt_result = await db.execute(
        select(EvaluationAttempt).where(
            EvaluationAttempt.id == attempt_id,
            EvaluationAttempt.invitation_id == invitation.id,
        )
    )
    attempt = attempt_result.scalar_one_or_none()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    # Idempotent: if already submitted, return success
    if attempt.status == "submitted":
        return {"status": "already_submitted"}

    # Fetch assessment
    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == attempt.assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    # Score and submit
    questions = await _fetch_assessment_questions(assessment, db)
    _score_attempt(attempt, questions, payload.responses, assessment)
    attempt.submitted_at = datetime.now(timezone.utc)
    attempt.status = "submitted"

    # Merge metadata (preserves shuffle_map from attempt start)
    meta = dict(attempt.attempt_metadata or {})
    if payload.metadata:
        meta.update(payload.metadata)
    meta["auto_submitted"] = True
    meta["submit_reason"] = meta.get("submit_reason", "browser_close")
    attempt.attempt_metadata = meta
    flag_modified(attempt, "attempt_metadata")

    invitation.status = "completed"
    await db.commit()

    return {"status": "submitted"}


@router.get("/assessments/{assessment_id}/attempts/{attempt_id}/detail")
async def get_attempt_detail(
    assessment_id: uuid.UUID,
    attempt_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
):
    """Get detailed attempt data including questions, student responses, and correct answers."""
    # Verify assessment exists and user has access
    assessment_result = await db.execute(
        select(EvaluationAssessment).where(EvaluationAssessment.id == assessment_id)
    )
    assessment = assessment_result.scalar_one_or_none()
    if not assessment:
        raise NotFoundException("Assessment not found")

    # Get the attempt
    attempt_result = await db.execute(
        select(EvaluationAttempt).where(
            EvaluationAttempt.id == attempt_id,
            EvaluationAttempt.assessment_id == assessment_id,
        )
    )
    attempt = attempt_result.scalar_one_or_none()
    if not attempt:
        raise NotFoundException("Attempt not found")

    # Get invitation info
    invitation = None
    if attempt.invitation_id:
        inv_result = await db.execute(
            select(EvaluationInvitation).where(EvaluationInvitation.id == attempt.invitation_id)
        )
        invitation = inv_result.scalar_one_or_none()

    # Fetch all questions for this assessment
    if assessment.question_ids:
        questions_result = await db.execute(
            select(EvaluationQuestion).where(EvaluationQuestion.id.in_(
                [uuid.UUID(qid) if isinstance(qid, str) else qid for qid in assessment.question_ids]
            )).order_by(EvaluationQuestion.order_index)
        )
    elif assessment.paper_id:
        questions_result = await db.execute(
            select(EvaluationQuestion)
            .where(EvaluationQuestion.paper_id == assessment.paper_id)
            .order_by(EvaluationQuestion.order_index)
        )
    else:
        questions_result = None

    questions = questions_result.scalars().all() if questions_result else []
    student_responses = attempt.responses or {}

    # Build question-by-question detail
    questions_detail = []
    for q in questions:
        student_answer = student_responses.get(str(q.id))
        correct_answer = q.correct_answer
        is_correct = (
            student_answer is not None
            and str(student_answer).strip().lower() == str(correct_answer or "").strip().lower()
        )
        marks_earned = 0
        if student_answer and is_correct:
            marks_earned = q.marks
        elif student_answer and not is_correct and assessment.negative_marking:
            penalty = q.negative_marks if q.negative_marks else (assessment.negative_mark_value or 0.25)
            marks_earned = -penalty

        questions_detail.append({
            "id": str(q.id),
            "question_type": q.question_type,
            "question_text": q.question_text,
            "options": q.options,
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "is_correct": is_correct,
            "marks": q.marks,
            "negative_marks": q.negative_marks,
            "marks_earned": marks_earned,
            "subject": q.subject,
            "chapter": q.chapter,
            "difficulty": q.difficulty,
            "explanation": q.explanation,
            "order_index": q.order_index,
        })

    return {
        "attempt_id": str(attempt.id),
        "assessment_id": str(assessment.id),
        "assessment_title": assessment.title,
        "student_email": invitation.email if invitation else None,
        "student_name": invitation.name if invitation else None,
        "score": attempt.score,
        "max_score": attempt.max_score,
        "percentage": attempt.percentage,
        "status": attempt.status,
        "started_at": attempt.started_at.isoformat() if attempt.started_at else None,
        "submitted_at": attempt.submitted_at.isoformat() if attempt.submitted_at else None,
        "attempt_metadata": attempt.attempt_metadata,
        "negative_marking": assessment.negative_marking,
        "negative_mark_value": assessment.negative_mark_value,
        "questions": questions_detail,
        "total_questions": len(questions),
        "correct_count": sum(1 for q in questions_detail if q["is_correct"]),
        "wrong_count": sum(1 for q in questions_detail if q["student_answer"] and not q["is_correct"]),
        "unanswered_count": sum(1 for q in questions_detail if not q["student_answer"]),
    }
