import uuid
from typing import Optional
from fastapi import APIRouter, status, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.ai import AiChat, AiChatMessage, AiChatSetting
from app.schemas.ai import (
    AiChatCreate, AiChatUpdate, AiChatResponse, AiMessageResponse,
    SendMessageRequest, ChatSettingsUpdate, ChatSettingsResponse,
    FollowUpRequest, FollowUpResponse, VideoRefsRequest, VideoRefsResponse,
    NextStepsRequest, NextStepsResponse,
    GeneratePracticeAssessmentRequest, GeneratedQuestionsResponse,
)
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()


@router.post("/chats", response_model=AiChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(payload: AiChatCreate, current_user: CurrentUser, db: DBSession):
    chat = AiChat(
        user_id=current_user.id,
        scope=payload.scope,
        title=payload.title,
        class_id=uuid.UUID(payload.class_id) if payload.class_id else None,
    )
    db.add(chat)
    await db.flush()

    # Create default settings
    settings = AiChatSetting(chat_id=chat.id)
    db.add(settings)

    await db.commit()
    await db.refresh(chat)
    return chat


@router.get("/chats", response_model=list[AiChatResponse])
async def list_chats(current_user: CurrentUser, db: DBSession, scope: str | None = Query(None)):
    q = select(AiChat).where(AiChat.user_id == current_user.id)
    if scope:
        q = q.where(AiChat.scope == scope)
    q = q.order_by(AiChat.updated_at.desc())
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/chats/{chat_id}", response_model=AiChatResponse)
async def get_chat(chat_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise NotFoundException("Chat not found")
    return chat


@router.patch("/chats/{chat_id}", response_model=AiChatResponse)
async def update_chat(chat_id: uuid.UUID, payload: AiChatUpdate, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise NotFoundException("Chat not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(chat, key, value)
    await db.commit()
    await db.refresh(chat)
    return chat


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise NotFoundException("Chat not found")
    await db.delete(chat)
    await db.commit()


@router.get("/chats/{chat_id}/messages", response_model=list[AiMessageResponse])
async def get_messages(chat_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(AiChatMessage)
        .where(AiChatMessage.chat_id == chat_id)
        .order_by(AiChatMessage.created_at.asc())
    )
    return result.scalars().all()


@router.post("/chats/{chat_id}/messages/stream")
async def send_message_stream(
    chat_id: uuid.UUID,
    payload: SendMessageRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Stream AI response via SSE (Server-Sent Events).
    Client should use EventSource or fetch with streaming to consume.
    """
    chat_result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise NotFoundException("Chat not found")

    # Deduct points before AI call
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="basic_chat", db=db)

    # Save user message
    user_msg = AiChatMessage(
        chat_id=chat_id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)

    # Auto-title: if chat is still "New Chat", name it from the first 8 words of the message
    if chat.title == "New Chat":
        words = payload.message.strip().split()
        auto_title = " ".join(words[:8])
        if len(words) > 8:
            auto_title += "…"
        chat.title = auto_title

    await db.commit()

    # Fetch chat settings from DB; merge with any client-supplied settings
    settings_result = await db.execute(
        select(AiChatSetting).where(AiChatSetting.chat_id == chat_id)
    )
    db_settings = settings_result.scalar_one_or_none()
    chat_settings: dict = {}
    if db_settings:
        chat_settings = {
            "difficulty": db_settings.difficulty,
            "personality": db_settings.personality,
            "content_length": db_settings.content_length,
            "explain_3ways": db_settings.explain_3ways,
            "mind_map": db_settings.mind_map,
            "examples": db_settings.examples,
            "output_mode": db_settings.output_mode,
            "student_mode": db_settings.student_mode,
            "video_refs": db_settings.video_refs,
            "followup": db_settings.followup,
            "practice": db_settings.practice,
            "next_steps": db_settings.next_steps,
        }
    # Client-supplied settings overlay DB values (kept for backwards compatibility)
    if payload.chat_settings:
        chat_settings.update(payload.chat_settings)

    # Get chat history
    history_result = await db.execute(
        select(AiChatMessage)
        .where(AiChatMessage.chat_id == chat_id)
        .order_by(AiChatMessage.created_at.asc())
        .limit(20)
    )
    history = history_result.scalars().all()
    messages = [{"role": m.role, "content": m.content} for m in history]

    ai = AIService()

    async def event_stream():
        full_response = ""
        async for chunk in ai.stream_chat(messages=messages, context=payload.context, chat_settings=chat_settings or None):
            full_response += chunk
            # Encode newlines so the SSE "data:" line stays intact (decoded by client)
            encoded = chunk.replace('\n', '\\n')
            yield f"data: {encoded}\n\n"

        # Save assistant message after streaming completes
        async with db as session:
            assistant_msg = AiChatMessage(
                chat_id=chat_id,
                role="assistant",
                content=full_response,
            )
            session.add(assistant_msg)
            chat_obj = await session.get(AiChat, chat_id)
            if chat_obj:
                from datetime import datetime, timezone
                chat_obj.updated_at = datetime.now(timezone.utc)
            await session.commit()

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chats/{chat_id}/messages", response_model=AiMessageResponse)
async def send_message(
    chat_id: uuid.UUID,
    payload: SendMessageRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Send a message and get a non-streaming response."""
    chat_result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    chat = chat_result.scalar_one_or_none()
    if not chat:
        raise NotFoundException("Chat not found")

    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="basic_chat", db=db)

    user_msg = AiChatMessage(chat_id=chat_id, role="user", content=payload.message)
    db.add(user_msg)
    await db.flush()

    history_result = await db.execute(
        select(AiChatMessage)
        .where(AiChatMessage.chat_id == chat_id)
        .order_by(AiChatMessage.created_at.asc())
        .limit(20)
    )
    history = history_result.scalars().all()
    messages = [{"role": m.role, "content": m.content} for m in history]

    ai = AIService()
    response_text = await ai.chat(messages=messages, context=payload.context)

    assistant_msg = AiChatMessage(
        chat_id=chat_id,
        role="assistant",
        content=response_text,
    )
    db.add(assistant_msg)
    await db.commit()
    await db.refresh(assistant_msg)
    return assistant_msg


@router.get("/chats/{chat_id}/settings", response_model=ChatSettingsResponse)
async def get_chat_settings(chat_id: uuid.UUID, current_user: CurrentUser, db: DBSession):
    result = await db.execute(
        select(AiChatSetting).where(AiChatSetting.chat_id == chat_id)
    )
    settings = result.scalar_one_or_none()
    if not settings:
        raise NotFoundException("Chat settings not found")
    return settings


@router.patch("/chats/{chat_id}/settings", response_model=ChatSettingsResponse)
async def update_chat_settings(
    chat_id: uuid.UUID, payload: ChatSettingsUpdate, current_user: CurrentUser, db: DBSession
):
    result = await db.execute(
        select(AiChatSetting).where(AiChatSetting.chat_id == chat_id)
    )
    settings = result.scalar_one_or_none()
    if not settings:
        settings = AiChatSetting(chat_id=chat_id)
        db.add(settings)
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(settings, key, value)
    await db.commit()
    await db.refresh(settings)
    return settings


@router.post("/chats/{chat_id}/follow-ups", response_model=FollowUpResponse)
async def generate_follow_ups(
    chat_id: uuid.UUID,
    payload: FollowUpRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate AI-predicted follow-up questions based on the last Q&A exchange."""
    # Verify chat ownership
    chat_result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise NotFoundException("Chat not found")

    ai = AIService()
    questions = await ai.generate_follow_up_questions(
        user_message=payload.message,
        ai_response=payload.response,
        count=payload.count,
    )
    return FollowUpResponse(questions=questions)


@router.post("/chats/{chat_id}/video-refs", response_model=VideoRefsResponse)
async def get_video_refs(
    chat_id: uuid.UUID,
    payload: VideoRefsRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Search YouTube for educational videos relevant to the last Q&A exchange."""
    # Verify chat ownership
    chat_result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise NotFoundException("Chat not found")

    ai = AIService()
    search_query = await ai.extract_video_search_query(
        user_message=payload.message,
        ai_response=payload.response,
    )

    from app.services.youtube_service import YouTubeService
    yt = YouTubeService()
    videos = await yt.search_videos(query=search_query, max_results=3)
    return VideoRefsResponse(videos=videos)


@router.post("/chats/{chat_id}/next-steps", response_model=NextStepsResponse)
async def get_next_steps(
    chat_id: uuid.UUID,
    payload: NextStepsRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate AI-suggested next steps based on the last Q&A exchange."""
    chat_result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise NotFoundException("Chat not found")

    ai = AIService()
    steps = await ai.generate_next_steps(
        user_message=payload.message,
        ai_response=payload.response,
        count=payload.count,
    )
    return NextStepsResponse(steps=steps)


@router.post("/generate-practice-assessment", response_model=GeneratedQuestionsResponse)
async def generate_practice_assessment_preview(
    payload: GeneratePracticeAssessmentRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate AI practice questions for review — does NOT save to DB."""
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="generate_assessment", db=db)

    # Resolve topic list: prefer explicit multi_topics, fall back to splitting topic string
    if payload.multi_topics:
        topics = payload.multi_topics
    elif payload.topic:
        topics = [t.strip() for t in payload.topic.split(",") if t.strip()]
    else:
        topics = None

    ai = AIService()
    raw = await ai.generate_practice_assessment(
        subject=payload.subject,
        topics=topics,
        grade=payload.grade,
        board=payload.board,
        difficulty=payload.difficulty,
        question_count=payload.question_count,
        question_types=payload.question_types,
        mode=payload.mode,
        blooms_level=payload.blooms_level or "mixed",
        mcq_subtypes=payload.mcq_subtypes,
        type_weightage=payload.type_weightage,
        topic_weightage=payload.topic_weightage,
        negative_marking=payload.negative_marking,
        source_text=payload.source_text,
    )

    import uuid as _uuid

    # Allowed types from the user's selection (strict enforcement)
    allowed_types = {t.lower() for t in (payload.question_types or ["mcq"])}

    question_json = []
    answer_key_json = []
    for q in raw:
        qid = q.get("id") or str(_uuid.uuid4())
        q_type = (q.get("type") or "mcq").lower()

        # Hard enforcement: skip questions with types the user didn't select
        if q_type not in allowed_types:
            continue

        opts = q.get("options")
        if isinstance(opts, dict):
            opts = list(opts.values())
        question_json.append({
            "id": qid,
            "type": q_type,
            "subtype": q.get("subtype"),
            "text": q.get("text") or q.get("question", ""),
            "options": opts,
            "pairs": q.get("pairs"),
            "points": q.get("marks") or q.get("points") or 1,
            "blooms_level": q.get("blooms_level"),
        })
        answer_key_json.append({
            "id": qid,
            "correctAnswer": q.get("correct_answer") or q.get("correctAnswer", ""),
            "explanation": q.get("explanation", ""),
        })

    return GeneratedQuestionsResponse(question_json=question_json, answer_key_json=answer_key_json)


class AutoGradeRequest(BaseModel):
    submissionText: Optional[str] = None
    rubric: Optional[dict] = None
    questions: Optional[list] = None
    answers: Optional[dict] = None
    studentName: Optional[str] = None
    assignmentDocumentUrl: Optional[str] = None
    studentFileUrls: Optional[list] = None
    feedbackOnly: bool = False


@router.post("/auto-grade")
async def auto_grade(
    payload: AutoGradeRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Auto-grade a submission using AI. Accepts rich payload directly from the grading page."""
    ai = AIService()
    result = await ai.auto_grade_direct(
        submission_text=payload.submissionText,
        rubric=payload.rubric,
        questions=payload.questions,
        answers=payload.answers,
        student_name=payload.studentName,
        feedback_only=payload.feedbackOnly,
    )
    return result


class AutoEvaluateAttemptRequest(BaseModel):
    responses_json: list  # [{ "questionId": "...", "answer": "..." }]
    answer_key_json: list  # [{ "id": "...", "correctAnswer": "...", "points": 1 }]
    questions_json: list   # [{ "id": "...", "type": "mcq", "text": "...", "points": 1, "options": [...] }]
    subject: Optional[str] = ""


@router.post("/auto-evaluate-attempt")
async def auto_evaluate_attempt(
    payload: AutoEvaluateAttemptRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Per-question AI grading for assignment/quiz attempts. Used by the teacher grading page."""
    ai = AIService()
    result = await ai.evaluate_assignment_attempt(
        responses_json=payload.responses_json,
        answer_key_json=payload.answer_key_json,
        questions_json=payload.questions_json,
        subject=payload.subject or "",
    )
    return result


class SuggestQuestionsRequest(BaseModel):
    topic: str
    subject: str = ""
    grade: int = 10
    mcqCount: int = 0
    fibCount: int = 0
    shortAnswerCount: int = 0
    trueFalseCount: int = 0
    matchCount: int = 0
    difficulty: Optional[str] = "medium"


@router.post("/suggest-questions")
async def suggest_questions_for_assignment(
    payload: SuggestQuestionsRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate typed assignment questions for the AssignmentEditor (MCQ, fill-blank, etc.)."""
    ai = AIService()
    questions = await ai.generate_assignment_questions(
        topic=payload.topic,
        subject=payload.subject,
        grade=payload.grade,
        mcq_count=payload.mcqCount,
        fib_count=payload.fibCount,
        short_answer_count=payload.shortAnswerCount,
        true_false_count=payload.trueFalseCount,
        match_count=payload.matchCount,
        difficulty=payload.difficulty or "medium",
    )
    return {"questions": questions}
