import uuid
import logging
from typing import Optional
from fastapi import APIRouter, status, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.models.ai import AiChat, AiChatMessage, AiChatSetting, AiInteractionHistory
from app.models.content import UserLibraryItem, DocChunk
from app.models.public_library import PublicFile, PublicFileChunk
from app.schemas.ai import (
    AiChatCreate, AiChatUpdate, AiChatResponse, AiMessageResponse,
    SendMessageRequest, ChatSettingsUpdate, ChatSettingsResponse,
    FollowUpRequest, FollowUpResponse, VideoRefsRequest, VideoRefsResponse,
    NextStepsRequest, NextStepsResponse, MindMapRequest, MindMapResponse,
    InfographicRequest, InfographicResponse,
    PracticeExercisesRequest, PracticeExercise, PracticeExercisesResponse,
    GeneratePracticeAssessmentRequest, GeneratedQuestionsResponse,
)
from app.core.exceptions import NotFoundException
from app.services.ai_service import AIService
from app.services.faiss_service import FAISSService
from app.services.points_service import PointsService
from app.models.subscription import PointCost
from app.config import settings

router = APIRouter()

# Maps chat_settings keys to point_cost action names for sub-feature billing
ENHANCEMENT_COST_MAP = {
    "followup":    "chat_followup",
    "next_steps":  "chat_next_steps",
    "video_refs":  "chat_video_refs",
    "mind_map":    "chat_mindmap",
    "infographic": "chat_infographic",
    "practice":    "chat_practice",
}


def _parse_org_id(org_id: str | None) -> uuid.UUID | None:
    if not org_id or org_id == "personal":
        return None
    try:
        return uuid.UUID(org_id)
    except ValueError:
        return None


def _apply_org_filter(q, column, org_id_param: str | None):
    parsed = _parse_org_id(org_id_param)
    if parsed is None:
        return q.where(column.is_(None))
    return q.where(column == parsed)


async def _build_rag_context(
    user_id: str,
    question: str,
    file_ids: list[str],
    db,
    ai: AIService,
    k: int = 15,
) -> str:
    """Return the most relevant chunk texts for the given question + file selection.

    1. Tries FAISS cosine-similarity search (fast, semantic).
    2. Falls back to ordered DB fetch when no FAISS index exists yet.
    """
    if not file_ids:
        return ""

    file_uuids = [uuid.UUID(fid) for fid in file_ids]
    faiss_svc = FAISSService(settings.STORAGE_ROOT)

    # --- FAISS path ---
    query_embedding = await ai.generate_query_embedding(question)
    if query_embedding and faiss_svc.user_has_index(user_id):
        ranked_ids = faiss_svc.search(user_id=user_id, query_embedding=query_embedding, k=k)
        if ranked_ids:
            chunk_uuids = [uuid.UUID(cid) for cid in ranked_ids]
            result = await db.execute(
                select(DocChunk)
                .join(UserLibraryItem, DocChunk.library_item_id == UserLibraryItem.id)
                .where(
                    DocChunk.id.in_(chunk_uuids),
                    UserLibraryItem.user_id == uuid.UUID(user_id),
                    UserLibraryItem.id.in_(file_uuids),
                )
            )
            by_id = {str(c.id): c for c in result.scalars().all()}
            chunks = [by_id[cid] for cid in ranked_ids if cid in by_id]
            if chunks:
                return "\n\n".join(c.chunk_text for c in chunks)

    # --- Fallback: recency / order-based ---
    result = await db.execute(
        select(DocChunk)
        .join(UserLibraryItem, DocChunk.library_item_id == UserLibraryItem.id)
        .where(
            UserLibraryItem.user_id == uuid.UUID(user_id),
            UserLibraryItem.id.in_(file_uuids),
        )
        .order_by(DocChunk.library_item_id, DocChunk.chunk_order)
        .limit(20)
    )
    chunks = result.scalars().all()
    return "\n\n".join(c.chunk_text for c in chunks)


def _inject_rag(messages: list[dict], question: str, rag_text: str) -> list[dict]:
    """Replace the last user message with a RAG-enriched version."""
    enriched_question = (
        "Use the following excerpts from the user's selected documents to answer accurately.\n"
        "If the answer is not in the excerpts, say so and answer from general knowledge.\n\n"
        f"--- DOCUMENT EXCERPTS ---\n{rag_text}\n--- END OF EXCERPTS ---\n\n"
        f"Question: {question}"
    )
    updated = list(messages)
    if updated and updated[-1]["role"] == "user":
        updated[-1] = {"role": "user", "content": enriched_question}
    else:
        updated.append({"role": "user", "content": enriched_question})
    return updated


def _inject_inline_docs(messages: list[dict], question: str, inline_docs: list[dict]) -> list[dict]:
    """Inject device-attached file content into the last user message.

    inline_docs is a list of {"filename": str, "text": str} dicts sent by the
    frontend when the user attaches a file from their device (not the vault).
    The original question is preserved for display; only the AI sees the file content.
    """
    doc_block = "\n\n".join(
        f"[ATTACHED FILE: {doc['filename']}]\n{doc['text']}\n[END OF FILE]"
        for doc in inline_docs
        if doc.get("text")
    )
    if not doc_block:
        return messages

    enriched = (
        "The user has attached the following file(s). "
        "Use the content to answer their question accurately.\n\n"
        f"{doc_block}\n\n"
        f"Question: {question}"
    )
    updated = list(messages)
    if updated and updated[-1]["role"] == "user":
        updated[-1] = {"role": "user", "content": enriched}
    else:
        updated.append({"role": "user", "content": enriched})
    return updated


LIBRARY_FAISS_KEY = "public_library"


async def _build_library_rag_context(
    question: str,
    library_file_id: str,
    db,
    ai: AIService,
    k: int = 15,
) -> str:
    """Return the most relevant chunk texts from a public library file.

    Searches the shared public_library FAISS index, then filters results
    to only chunks belonging to the specified file.
    """
    import logging
    log = logging.getLogger(__name__)

    faiss_svc = FAISSService(settings.STORAGE_ROOT)
    file_uuid = uuid.UUID(library_file_id)

    # FAISS path — search with a larger k since we'll filter by file_id
    query_embedding = await ai.generate_query_embedding(question)
    has_index = faiss_svc.user_has_index(LIBRARY_FAISS_KEY)
    log.info("Library RAG: file=%s, has_index=%s, has_embedding=%s", library_file_id, has_index, bool(query_embedding))

    if query_embedding and has_index:
        ranked_ids = faiss_svc.search(
            user_id=LIBRARY_FAISS_KEY,
            query_embedding=query_embedding,
            k=k * 3,  # over-fetch since we filter to one file
        )
        log.info("Library RAG: FAISS returned %d results", len(ranked_ids) if ranked_ids else 0)
        if ranked_ids:
            chunk_uuids = [uuid.UUID(cid) for cid in ranked_ids]
            result = await db.execute(
                select(PublicFileChunk)
                .where(
                    PublicFileChunk.id.in_(chunk_uuids),
                    PublicFileChunk.file_id == file_uuid,
                )
            )
            by_id = {str(c.id): c for c in result.scalars().all()}
            log.info("Library RAG: %d chunks matched file_id after filter", len(by_id))
            # Maintain FAISS ranking order, limited to k results
            chunks = [by_id[cid] for cid in ranked_ids if cid in by_id][:k]
            if chunks:
                text = "\n\n".join(c.chunk_text for c in chunks)
                log.info("Library RAG: returning %d chars from %d FAISS chunks", len(text), len(chunks))
                return text

    # Fallback: ordered DB fetch
    result = await db.execute(
        select(PublicFileChunk)
        .where(PublicFileChunk.file_id == file_uuid)
        .order_by(PublicFileChunk.chunk_order)
        .limit(20)
    )
    chunks = result.scalars().all()
    log.info("Library RAG: fallback DB fetch returned %d chunks for file %s", len(chunks), library_file_id)
    return "\n\n".join(c.chunk_text for c in chunks)


def _inject_library_rag(messages: list[dict], question: str, rag_text: str, book_title: str) -> list[dict]:
    """Replace the last user message with library-book-RAG-enriched version."""
    enriched_question = (
        f"The user is studying from the book '{book_title}' in the public library.\n"
        "Use the following excerpts from this book to answer accurately.\n"
        "If the answer is not in the excerpts, say so and answer from general knowledge.\n\n"
        f"--- BOOK EXCERPTS ---\n{rag_text}\n--- END OF EXCERPTS ---\n\n"
        f"Question: {question}"
    )
    updated = list(messages)
    if updated and updated[-1]["role"] == "user":
        updated[-1] = {"role": "user", "content": enriched_question}
    else:
        updated.append({"role": "user", "content": enriched_question})
    return updated


@router.post("/chats", response_model=AiChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(payload: AiChatCreate, current_user: CurrentUser, db: DBSession):
    chat = AiChat(
        user_id=current_user.id,
        org_id=_parse_org_id(payload.org_id),
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
async def list_chats(current_user: CurrentUser, db: DBSession, scope: str | None = Query(None), org_id: str | None = Query(None)):
    q = select(AiChat).where(AiChat.user_id == current_user.id)
    if org_id is not None:
        q = _apply_org_filter(q, AiChat.org_id, org_id)
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

    # Extract org_id from context for org-workspace point deduction
    org_id = _parse_org_id((payload.context or {}).get("org_id"))

    # Check feature access + usage limits
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ai_chat", db=db, org_id=org_id)

    # NOTE: Points deduction is deferred until after chat_settings are resolved (below)
    # so we can calculate the total cost including enabled enhancements.

    # Save user message — store any attached filenames in sources_json so the
    # frontend can render them as file chips above the message bubble.
    _inline_docs_meta = (payload.context or {}).get("inline_docs") or []
    user_msg = AiChatMessage(
        chat_id=chat_id,
        role="user",
        content=payload.message,
        sources_json={"attached_files": [d["filename"] for d in _inline_docs_meta if d.get("filename")]} if _inline_docs_meta else None,
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
            "infographic": db_settings.infographic,
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

    # Calculate total cost: base chat + enabled enhancement sub-features
    enabled_actions = [
        action for key, action in ENHANCEMENT_COST_MAP.items()
        if chat_settings.get(key)
    ]
    base_result = await db.execute(select(PointCost).where(PointCost.action == "basic_chat"))
    base_pc = base_result.scalars().first()
    if not base_pc:
        logger.warning("PointCost row for 'basic_chat' not found, defaulting to cost=1, xp=0")
    total_cost = base_pc.cost if base_pc else 1
    total_xp = (base_pc.xp_reward or 0) if base_pc else 0
    if enabled_actions:
        enh_result = await db.execute(select(PointCost).where(PointCost.action.in_(enabled_actions)))
        for pc in enh_result.scalars().all():
            total_cost += pc.cost
            total_xp += pc.xp_reward or 0
    await points_service.deduct_custom(
        user_id=current_user.id, action="basic_chat", db=db,
        cost_override=total_cost, xp_override=total_xp, org_id=org_id,
    )

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

    # RAG: enrich messages with vault content when files are selected
    selected_files: list[str] = (
        payload.selected_files
        or (payload.context or {}).get("selected_files")
        or []
    )
    if selected_files:
        try:
            rag_text = await _build_rag_context(
                user_id=str(current_user.id),
                question=payload.message,
                file_ids=selected_files,
                db=db,
                ai=ai,
            )
            if rag_text:
                messages = _inject_rag(messages, payload.message, rag_text)
        except Exception as e:
            print(f"[StreamChat] RAG context build failed: {e}", flush=True)

    # Library RAG: use public library book embeddings
    library_file_id: str | None = (payload.context or {}).get("library_file_id")
    if library_file_id:
        try:
            # Fetch book title for prompt context
            lib_file = await db.get(PublicFile, uuid.UUID(library_file_id))
            lib_rag_text = await _build_library_rag_context(
                question=payload.message,
                library_file_id=library_file_id,
                db=db,
                ai=ai,
            )
            if lib_rag_text:
                messages = _inject_library_rag(
                    messages, payload.message, lib_rag_text,
                    book_title=lib_file.title if lib_file else "Unknown Book",
                )
        except Exception as e:
            print(f"[StreamChat] Library RAG context build failed: {e}", flush=True)

    # Inline docs: device-attached files sent from the frontend (not saved to vault)
    inline_docs: list[dict] = (payload.context or {}).get("inline_docs") or []
    if inline_docs:
        messages = _inject_inline_docs(messages, payload.message, inline_docs)

    # Detect if files are involved (vault selection, inline upload, or library book)
    has_files = bool(selected_files) or bool(inline_docs) or bool(library_file_id)

    async def event_stream():
        full_response = ""
        try:
            async for chunk in ai.stream_chat(messages=messages, context=payload.context, chat_settings=chat_settings or None, has_files=has_files):
                full_response += chunk
                # Encode newlines so the SSE "data:" line stays intact (decoded by client)
                encoded = chunk.replace('\n', '\\n')
                yield f"data: {encoded}\n\n"
        except Exception as e:
            import traceback
            print(f"[StreamChat] Streaming error: {e}", flush=True)
            traceback.print_exc()
            # Send a user-visible error so the chat doesn't appear blank
            if not full_response:
                error_msg = "Sorry, I encountered an issue generating a response. Please try again."
                yield f"data: {error_msg}\n\n"
                full_response = error_msg

        # Save assistant message after streaming completes
        try:
            async with db as session:
                assistant_msg = AiChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=full_response,
                    enhancements_json={"settings_snapshot": chat_settings} if chat_settings else None,
                )
                session.add(assistant_msg)
                chat_obj = await session.get(AiChat, chat_id)
                if chat_obj:
                    from datetime import datetime, timezone
                    chat_obj.updated_at = datetime.now(timezone.utc)
                # Track interaction for analytics (study-time, activity-heatmap)
                ctx = payload.context or {}
                session.add(AiInteractionHistory(
                    user_id=current_user.id,
                    service="ai-assistant",
                    query=payload.message[:500] if payload.message else None,
                    response_summary=full_response[:200] if full_response else None,
                    context_snapshot={
                        "subject":  ctx.get("subject"),
                        "grade":    ctx.get("grade"),
                        "board":    ctx.get("board"),
                        "language": ctx.get("language"),
                    },
                ))
                await session.commit()
        except Exception as e:
            print(f"[StreamChat] DB save error: {e}", flush=True)

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

    # Extract org_id from context for org-workspace point deduction
    org_id = _parse_org_id((payload.context or {}).get("org_id"))

    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ai_chat", db=db, org_id=org_id)

    # Calculate dynamic cost based on enabled enhancements
    cs = payload.chat_settings or {}
    enabled_actions = [action for key, action in ENHANCEMENT_COST_MAP.items() if cs.get(key)]
    base_result = await db.execute(select(PointCost).where(PointCost.action == "basic_chat"))
    base_pc = base_result.scalars().first()
    if not base_pc:
        logger.warning("PointCost row for 'basic_chat' not found, defaulting to cost=1, xp=0")
    total_cost = base_pc.cost if base_pc else 1
    total_xp = (base_pc.xp_reward or 0) if base_pc else 0
    if enabled_actions:
        enh_result = await db.execute(select(PointCost).where(PointCost.action.in_(enabled_actions)))
        for pc in enh_result.scalars().all():
            total_cost += pc.cost
            total_xp += pc.xp_reward or 0
    await points_service.deduct_custom(
        user_id=current_user.id, action="basic_chat", db=db,
        cost_override=total_cost, xp_override=total_xp, org_id=org_id,
    )

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

    # RAG: enrich messages with vault content when files are selected
    selected_files: list[str] = (
        payload.selected_files
        or (payload.context or {}).get("selected_files")
        or []
    )
    if selected_files:
        try:
            rag_text = await _build_rag_context(
                user_id=str(current_user.id),
                question=payload.message,
                file_ids=selected_files,
                db=db,
                ai=ai,
            )
            if rag_text:
                messages = _inject_rag(messages, payload.message, rag_text)
        except Exception as e:
            print(f"[SendMessage] RAG context build failed: {e}", flush=True)

    # Library RAG: use public library book embeddings
    library_file_id_sync: str | None = (payload.context or {}).get("library_file_id")
    if library_file_id_sync:
        try:
            lib_file_sync = await db.get(PublicFile, uuid.UUID(library_file_id_sync))
            lib_rag_text_sync = await _build_library_rag_context(
                question=payload.message,
                library_file_id=library_file_id_sync,
                db=db,
                ai=ai,
            )
            if lib_rag_text_sync:
                messages = _inject_library_rag(
                    messages, payload.message, lib_rag_text_sync,
                    book_title=lib_file_sync.title if lib_file_sync else "Unknown Book",
                )
        except Exception as e:
            print(f"[SendMessage] Library RAG context build failed: {e}", flush=True)

    # Inline docs: device-attached files sent from the frontend (not saved to vault)
    inline_docs: list[dict] = (payload.context or {}).get("inline_docs") or []
    if inline_docs:
        messages = _inject_inline_docs(messages, payload.message, inline_docs)

    # Detect if files are involved (vault selection, inline upload, or library book)
    has_files = bool(selected_files) or bool(inline_docs) or bool(library_file_id_sync)

    response_text = await ai.chat(messages=messages, context=payload.context, has_files=has_files)

    assistant_msg = AiChatMessage(
        chat_id=chat_id,
        role="assistant",
        content=response_text,
    )
    db.add(assistant_msg)
    ctx = payload.context or {}
    db.add(AiInteractionHistory(
        user_id=current_user.id,
        service="ai-assistant",
        query=payload.message[:500] if payload.message else None,
        response_summary=response_text[:200] if response_text else None,
        context_snapshot={
            "subject":  ctx.get("subject"),
            "grade":    ctx.get("grade"),
            "board":    ctx.get("board"),
            "language": ctx.get("language"),
        },
    ))
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


async def _persist_enhancement(db, chat_id: uuid.UUID, key: str, value):
    """Atomically merge one enhancement key into the last assistant message's enhancements_json.

    Uses SQLAlchemy ORM update with PostgreSQL JSONB ``||`` merge so concurrent
    calls from parallel enhancement endpoints don't overwrite each other.
    Failures are logged but never propagated — the enhancement data has already
    been generated and will still be returned to the frontend.
    """
    import logging
    from sqlalchemy import update, type_coerce, func
    from sqlalchemy.dialects.postgresql import JSONB as JSONB_TYPE

    try:
        # Find the last assistant message id
        result = await db.execute(
            select(AiChatMessage.id)
            .where(AiChatMessage.chat_id == chat_id, AiChatMessage.role == "assistant")
            .order_by(AiChatMessage.created_at.desc())
            .limit(1)
        )
        msg_id = result.scalar_one_or_none()
        if not msg_id:
            return

        patch = {key: value}
        await db.execute(
            update(AiChatMessage)
            .where(AiChatMessage.id == msg_id)
            .values(
                enhancements_json=func.coalesce(
                    AiChatMessage.enhancements_json, type_coerce({}, JSONB_TYPE)
                ).concat(type_coerce(patch, JSONB_TYPE))
            )
        )
        await db.commit()
    except Exception:
        logging.getLogger(__name__).warning("Failed to persist enhancement '%s' for chat %s", key, chat_id, exc_info=True)
        try:
            await db.rollback()
        except Exception:
            pass


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
    await _persist_enhancement(db, chat_id, "follow_ups", questions)
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
        grade=payload.grade,
        student_mode=payload.student_mode or False,
    )

    from app.services.youtube_service import YouTubeService
    yt = YouTubeService()
    print(f"youtube query is {search_query}")
    videos = await yt.search_videos(query=search_query, max_results=3)
    await _persist_enhancement(db, chat_id, "video_refs", [v if isinstance(v, dict) else v.model_dump() for v in videos])
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
    await _persist_enhancement(db, chat_id, "next_steps", steps)
    return NextStepsResponse(steps=steps)


@router.post("/chats/{chat_id}/practice-exercises", response_model=PracticeExercisesResponse)
async def generate_chat_practice_exercises(
    chat_id: uuid.UUID,
    payload: PracticeExercisesRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate dynamic practice exercises based on the last Q&A exchange."""
    chat_result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise NotFoundException("Chat not found")

    ai = AIService()
    raw_exercises = await ai.generate_practice_exercises(
        user_message=payload.message,
        ai_response=payload.response,
        count=payload.count,
        grade=payload.grade,
        difficulty=payload.difficulty,
    )
    exercises = [PracticeExercise(**ex) for ex in raw_exercises]
    await _persist_enhancement(
        db, chat_id, "practice_exercises",
        [ex.model_dump() for ex in exercises],
    )
    return PracticeExercisesResponse(exercises=exercises)


@router.post("/chats/{chat_id}/mindmap", response_model=MindMapResponse)
async def generate_chat_mindmap(
    chat_id: uuid.UUID,
    payload: MindMapRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate a visual mind map from the last Q&A exchange."""
    chat_result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise NotFoundException("Chat not found")

    ai = AIService()
    mindmap = await ai.generate_chat_mindmap(
        user_message=payload.message,
        ai_response=payload.response,
        grade=payload.grade,
        language=payload.language,
    )
    await _persist_enhancement(db, chat_id, "mind_map", mindmap)
    return MindMapResponse(mindmap=mindmap)


@router.post("/chats/{chat_id}/infographic", response_model=InfographicResponse)
async def generate_chat_infographic(
    chat_id: uuid.UUID,
    payload: InfographicRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate a structured infographic from the last Q&A exchange."""
    chat_result = await db.execute(
        select(AiChat).where(AiChat.id == chat_id, AiChat.user_id == current_user.id)
    )
    if not chat_result.scalar_one_or_none():
        raise NotFoundException("Chat not found")

    ai = AIService()
    infographic = await ai.generate_chat_infographic(
        user_message=payload.message,
        ai_response=payload.response,
        grade=payload.grade,
        language=payload.language,
    )
    await _persist_enhancement(db, chat_id, "infographic", infographic)
    return InfographicResponse(infographic=infographic)


@router.post("/generate-practice-assessment", response_model=GeneratedQuestionsResponse)
async def generate_practice_assessment_preview(
    payload: GeneratePracticeAssessmentRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate AI practice questions for review — does NOT save to DB."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="create_assessment", db=db, org_id=org_id)
    await points_service.deduct(user_id=current_user.id, action="generate_assessment", db=db, org_id=org_id)

    # Resolve topic list: prefer explicit multi_topics, fall back to splitting topic string
    if payload.multi_topics:
        topics = payload.multi_topics
    elif payload.topic:
        topics = [t.strip() for t in payload.topic.split(",") if t.strip()]
    else:
        topics = None

    # Resolve source text: fetch vault file chunks when source_ref_id is provided
    resolved_source_text = payload.source_text
    if payload.source_ref_id and not resolved_source_text:
        try:
            from app.models.content import DocChunk, UserLibraryItem
            chunks_result = await db.execute(
                select(DocChunk)
                .join(UserLibraryItem, DocChunk.library_item_id == UserLibraryItem.id)
                .where(
                    UserLibraryItem.id == uuid.UUID(payload.source_ref_id),
                    UserLibraryItem.user_id == current_user.id,
                )
                .order_by(DocChunk.chunk_order)
            )
            chunks = chunks_result.scalars().all()
            if chunks:
                resolved_source_text = " ".join(c.chunk_text for c in chunks)
        except Exception:
            pass  # fall back to topic-based generation if fetch fails

    # Resolve library file content via FAISS RAG (takes priority when library_file_ids are provided)
    import logging
    _log = logging.getLogger(__name__)
    if payload.library_file_ids:
        _log.info("Assessment RAG: library_file_ids=%s", payload.library_file_ids)
        ai_for_rag = AIService()
        topic_query = payload.topic or payload.subject or "general content"
        _log.info("Assessment RAG: topic_query=%r", topic_query)
        rag_parts: list[str] = []
        for lib_fid in payload.library_file_ids:
            part = await _build_library_rag_context(
                question=topic_query,
                library_file_id=lib_fid,
                db=db,
                ai=ai_for_rag,
                k=20,
            )
            _log.info("Assessment RAG: file=%s returned %d chars", lib_fid, len(part) if part else 0)
            if part:
                rag_parts.append(part)
        if rag_parts:
            resolved_source_text = "\n\n---\n\n".join(rag_parts)
            _log.info("Assessment RAG: total resolved_source_text=%d chars", len(resolved_source_text))
        else:
            _log.warning("Assessment RAG: NO content resolved from library files")

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
        source_text=resolved_source_text,
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


@router.post("/generate-practice-assessment-stream")
async def generate_practice_assessment_stream(
    payload: GeneratePracticeAssessmentRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate AI practice questions with SSE progress streaming.

    Primary: Celery worker (offloads LLM work from FastAPI process).
    Fallback: In-process async (if Celery/Redis dispatch fails).
    """
    import asyncio
    import json
    from typing import AsyncGenerator

    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.check_and_increment_usage(
        user_id=current_user.id, feature_key="create_assessment", db=db, org_id=org_id,
    )
    await points_service.deduct(
        user_id=current_user.id, action="generate_assessment", db=db, org_id=org_id,
    )

    # Resolve topics
    if payload.multi_topics:
        topics = payload.multi_topics
    elif payload.topic:
        topics = [t.strip() for t in payload.topic.split(",") if t.strip()]
    else:
        topics = None

    # Resolve source text
    resolved_source_text = payload.source_text
    if payload.source_ref_id and not resolved_source_text:
        try:
            chunks_result = await db.execute(
                select(DocChunk)
                .join(UserLibraryItem, DocChunk.library_item_id == UserLibraryItem.id)
                .where(
                    UserLibraryItem.id == uuid.UUID(payload.source_ref_id),
                    UserLibraryItem.user_id == current_user.id,
                )
                .order_by(DocChunk.chunk_order)
            )
            chunks = chunks_result.scalars().all()
            if chunks:
                resolved_source_text = " ".join(c.chunk_text for c in chunks)
        except Exception:
            pass

    # Resolve library file content via FAISS RAG
    if payload.library_file_ids:
        ai_for_rag = AIService()
        topic_query = payload.topic or payload.subject or "general content"
        rag_parts: list[str] = []
        for lib_fid in payload.library_file_ids:
            part = await _build_library_rag_context(
                question=topic_query, library_file_id=lib_fid,
                db=db, ai=ai_for_rag, k=20,
            )
            if part:
                rag_parts.append(part)
        if rag_parts:
            resolved_source_text = "\n\n---\n\n".join(rag_parts)

    allowed_types = {t.lower() for t in (payload.question_types or ["mcq"])}

    task_params = {
        "subject": payload.subject,
        "topics": topics,
        "grade": payload.grade,
        "board": payload.board,
        "difficulty": payload.difficulty,
        "question_count": payload.question_count,
        "question_types": payload.question_types,
        "mode": payload.mode,
        "blooms_level": payload.blooms_level or "mixed",
        "mcq_subtypes": payload.mcq_subtypes,
        "type_weightage": payload.type_weightage,
        "topic_weightage": payload.topic_weightage,
        "negative_marking": payload.negative_marking,
        "source_text": resolved_source_text,
        "allowed_types": list(allowed_types),
        "user_id": str(current_user.id),
        "org_id": str(org_id) if org_id else None,
    }

    # ── Try Celery dispatch (primary path) ──
    celery_ok = False
    redis_channel = None
    pubsub = None
    aio_redis = None

    try:
        import redis.asyncio as aioredis
        from app.tasks.assessment_tasks import generate_assessment_task

        task_id = str(uuid.uuid4())
        redis_channel = f"assessment:progress:{task_id}"
        aio_redis = aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = aio_redis.pubsub()
        await pubsub.subscribe(redis_channel)

        generate_assessment_task.apply_async(args=[task_params], task_id=task_id)
        celery_ok = True
        print(f"[Assessment] Celery task dispatched: {task_id}", flush=True)
    except Exception as e:
        print(f"[Assessment] Celery dispatch failed, falling back to in-process: {e}", flush=True)
        if pubsub:
            try:
                await pubsub.unsubscribe()
                await pubsub.close()
            except Exception:
                pass
        if aio_redis:
            try:
                await aio_redis.close()
            except Exception:
                pass

    if celery_ok:
        async def celery_event_generator() -> AsyncGenerator[str, None]:
            try:
                yield f"data: {json.dumps({'stage': 'points_done', 'progress': 10, 'message': 'Points deducted successfully'})}\n\n"
                while True:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
                    if msg and msg["type"] == "message":
                        try:
                            event = json.loads(msg["data"])
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if event.get("stage") == "__DONE__":
                            yield "data: [DONE]\n\n"
                            break
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("stage") in ("complete", "error"):
                            yield "data: [DONE]\n\n"
                            break
                    else:
                        yield ": heartbeat\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                await pubsub.unsubscribe(redis_channel)
                await pubsub.close()
                await aio_redis.close()

        generator = celery_event_generator()
    else:
        # ── FALLBACK: in-process async ──
        async def inprocess_event_generator() -> AsyncGenerator[str, None]:
            try:
                yield f"data: {json.dumps({'stage': 'points_done', 'progress': 10, 'message': 'Points deducted successfully'})}\n\n"
                yield f"data: {json.dumps({'stage': 'preparing', 'progress': 20, 'message': 'Preparing assessment configuration...'})}\n\n"
                yield f"data: {json.dumps({'stage': 'generating', 'progress': 30, 'message': f'Generating {payload.question_count} questions with AI...'})}\n\n"

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
                    source_text=resolved_source_text,
                )

                yield f"data: {json.dumps({'stage': 'processing', 'progress': 80, 'message': 'Processing and validating questions...'})}\n\n"

                import uuid as _uuid
                question_json = []
                answer_key_json = []
                for q in raw:
                    qid = q.get("id") or str(_uuid.uuid4())
                    q_type = (q.get("type") or "mcq").lower()
                    if q_type not in allowed_types:
                        continue
                    opts = q.get("options")
                    if isinstance(opts, dict):
                        opts = list(opts.values())
                    question_json.append({
                        "id": qid, "type": q_type, "subtype": q.get("subtype"),
                        "text": q.get("text") or q.get("question", ""),
                        "options": opts, "pairs": q.get("pairs"),
                        "points": q.get("marks") or q.get("points") or 1,
                        "blooms_level": q.get("blooms_level"),
                    })
                    answer_key_json.append({
                        "id": qid,
                        "correctAnswer": q.get("correct_answer") or q.get("correctAnswer", ""),
                        "explanation": q.get("explanation", ""),
                    })

                yield f"data: {json.dumps({'stage': 'complete', 'progress': 100, 'message': f'{len(question_json)} questions generated successfully!', 'question_json': question_json, 'answer_key_json': answer_key_json})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'stage': 'error', 'progress': 0, 'message': str(exc)})}\n\n"
                yield "data: [DONE]\n\n"

        generator = inprocess_event_generator()

    return StreamingResponse(generator, media_type="text/event-stream")


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
    lesson_plan_id: Optional[str] = None
    rubric_id: Optional[str] = None
    source_text: Optional[str] = None  # Full text from a vault file for document-grounded generation


@router.post("/suggest-questions")
async def suggest_questions_for_assignment(
    payload: SuggestQuestionsRequest,
    current_user: CurrentUser,
    db: DBSession,
):
    """Generate typed assignment questions, optionally enriched by lesson plan and/or rubric context."""
    import uuid as _uuid
    from sqlalchemy import select as _select
    from app.models.classes import LessonPlan, Rubric

    # --- Fetch lesson plan context ---
    lesson_plan_context: dict | None = None
    if payload.lesson_plan_id:
        try:
            lp_result = await db.execute(
                _select(LessonPlan).where(LessonPlan.id == _uuid.UUID(payload.lesson_plan_id))
            )
            lp = lp_result.scalar_one_or_none()
            if lp:
                lesson_plan_context = {
                    "title": lp.title,
                    "topic": lp.topic,
                    "objectives": lp.objectives or [],
                    "steps": lp.steps or [],
                    "practice_tasks": lp.practice_tasks or [],
                    "formative_check": lp.formative_check or "",
                }
        except Exception:
            pass  # proceed without lesson plan if fetch fails

    # --- Fetch rubric criteria ---
    rubric_criteria: list | None = None
    if payload.rubric_id:
        try:
            rb_result = await db.execute(
                _select(Rubric).where(Rubric.id == _uuid.UUID(payload.rubric_id))
            )
            rb = rb_result.scalar_one_or_none()
            if rb:
                # Strip __meta sentinel entries before passing to AI
                rubric_criteria = [
                    c for c in (rb.criteria or [])
                    if isinstance(c, dict) and not c.get("__meta")
                ]
        except Exception:
            pass  # proceed without rubric if fetch fails

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
        lesson_plan_context=lesson_plan_context,
        rubric_criteria=rubric_criteria,
        source_text=payload.source_text,
    )
    return {"questions": questions}
