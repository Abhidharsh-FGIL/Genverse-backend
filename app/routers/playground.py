import uuid
from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sqlalchemy import select

from app.dependencies import DBSession, CurrentUser
from app.schemas.ai import PlaygroundRequest
from app.services.ai_service import AIService, get_ai_service
from app.services.points_service import PointsService
from app.models.subscription import PointCost
from app.models.ai import AiContextSession

router = APIRouter()


def _parse_org_id(raw: Any) -> uuid.UUID | None:
    """Convert a raw org_id string to UUID, returning None for personal workspace."""
    if not raw or raw == "personal":
        return None
    try:
        return uuid.UUID(str(raw))
    except (ValueError, AttributeError):
        return None

VALID_MODES = {"experiment", "play", "challenge", "imagine"}


def _resolve_role_grade(payload, current_user) -> tuple[str, int | None]:
    """Use the authenticated user's profile role & grade when the frontend sends defaults/nulls."""
    role = current_user.role or payload.role
    grade = payload.grade if payload.grade is not None else getattr(current_user, "grade", None)
    return role, grade


async def _resolve_context(payload, current_user, db) -> tuple[str, int | None, str | None, bool]:
    """Resolve role, grade, board, and student_mode with layered fallbacks.

    Precedence (highest first) for grade / board:
      1. The value the frontend sent on the payload.
      2. The user's saved AI context session (same table ``GlobalContextBar`` writes to).
         Looked up for the workspace implied by ``payload.org_id`` (or ``"personal"``).
      3. The user's profile grade / board_preference.

    This lets the backend recover grade/board even when the frontend's AI context
    hasn't finished loading before a Playground mode auto-fires its initial
    request — which was causing ``grade: null`` and missing ``board`` in the
    ``/match-pairs`` payload.

    ``student_mode`` defaults to True. When False, grade and board are cleared
    so the prompt is fully generic.
    """
    role = current_user.role or getattr(payload, "role", "student")
    student_mode = getattr(payload, "student_mode", True)
    if student_mode is None:
        student_mode = True

    grade = getattr(payload, "grade", None)
    board = getattr(payload, "board", None)

    if grade is None or board is None:
        # Resolve workspace id in the same way the frontend stores context.
        org_id_raw = getattr(payload, "org_id", None)
        context_dict = getattr(payload, "context", None) or {}
        ws_raw = org_id_raw or context_dict.get("org_id")
        workspace_id = str(ws_raw) if ws_raw and ws_raw != "personal" else "personal"

        # 1) Try the active workspace row first.
        ctx_result = await db.execute(
            select(AiContextSession).where(
                AiContextSession.user_id == current_user.id,
                AiContextSession.workspace_id == workspace_id,
            )
        )
        ctx = ctx_result.scalar_one_or_none()
        if ctx is not None:
            if grade is None:
                grade = ctx.grade
            if board is None:
                board = ctx.board

        # 2) If still missing, look across every AiContextSession the user owns
        #    (picks up the case where board was saved in a different workspace —
        #    e.g. during signup in "personal" but the user is now in an org).
        if grade is None or board is None:
            any_result = await db.execute(
                select(AiContextSession).where(
                    AiContextSession.user_id == current_user.id,
                )
            )
            for any_ctx in any_result.scalars().all():
                if grade is None and any_ctx.grade is not None:
                    grade = any_ctx.grade
                if board is None and any_ctx.board:
                    board = any_ctx.board
                if grade is not None and board:
                    break

    # Final fallback: the user's profile.
    if grade is None:
        grade = getattr(current_user, "grade", None)
    if board is None:
        board = getattr(current_user, "board_preference", None)

    if not student_mode:
        grade = None
        board = None
    return role, grade, board, bool(student_mode)

# Map playground mode → feature_key for plan gating
MODE_FEATURE_KEY = {
    "experiment": "playground_rapid_mcq",
    "play": "playground_concept_connect",
    "challenge": "playground_roleplay",
    "imagine": "playground_imagine",
}


# ── New mode-specific request schemas ─────────────────────────────────────────

class SetupRequest(BaseModel):
    topic: str


class ControlDef(BaseModel):
    id: str
    label: str
    unit: str = "%"
    icon: str = "Activity"
    default: int = 50
    description: str = ""


class SimulateRequest(BaseModel):
    topic: str
    variables: dict[str, Any] = {}
    controls: list[ControlDef] = []   # topic-specific control definitions for AI context


class FlashcardsRequest(BaseModel):
    topic: str
    role: str = "student"       # student | teacher | normal_user | guardian
    grade: int | None = None    # 1-12 or None for adults
    board: str | None = None    # CBSE | ICSE | State | etc. Ignored when student_mode is False.
    student_mode: bool = True   # When False, ignore grade/board and generate generically.
    org_id: Optional[str] = None


class ImagineEvalRequest(BaseModel):
    topic: str
    question: str
    answer: str
    max_points: int = 20
    role: str = "student"
    grade: int | None = None
    board: str | None = None
    student_mode: bool = True
    org_id: Optional[str] = None


class QuestRequest(BaseModel):
    topic: str
    history: list[str] = []
    choice: str | None = None


class MirrorChatRequest(BaseModel):
    topic: str
    persona: str = "The Ancient Scholar"
    messages: list[dict] = []


@router.post("/explore/stream")
async def playground_explore_stream(
    payload: PlaygroundRequest, current_user: CurrentUser, db: DBSession
):
    """
    Interactive topic exploration with SSE streaming.
    Modes: experiment | play | challenge | imagine
    """
    if payload.mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Mode must be one of: {', '.join(VALID_MODES)}")

    org_id = _parse_org_id((payload.context or {}).get("org_id"))
    points_service = PointsService()
    feature_key = MODE_FEATURE_KEY.get(payload.mode, "playground_rapid_mcq")
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key=feature_key, db=db, org_id=org_id)

    # Calculate cost: base + harder_mode bonus
    base_r = await db.execute(select(PointCost).where(PointCost.action == "playground_explore"))
    base_pc = base_r.scalars().first()
    total_cost = base_pc.cost if base_pc else 2
    total_xp = (base_pc.xp_reward or 0) if base_pc else 0
    if payload.harder_mode:
        h_r = await db.execute(select(PointCost).where(PointCost.action == "playground_harder"))
        h_pc = h_r.scalars().first()
        if h_pc:
            total_cost += h_pc.cost
            total_xp += h_pc.xp_reward or 0
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=total_cost, xp_override=total_xp, org_id=org_id,
    )

    ai = get_ai_service()

    # Resolve grade/board/student_mode with the same session-backed fallback
    # used by the non-streaming mode endpoints, then merge them onto context
    # so stream_playground sees consistent values regardless of where the
    # caller placed them on the request.
    _role, resolved_grade, resolved_board, resolved_sm = await _resolve_context(payload, current_user, db)
    merged_context = dict(payload.context or {})
    merged_context["student_mode"] = resolved_sm
    if resolved_board is not None:
        merged_context["board"] = resolved_board

    async def event_stream():
        async for chunk in ai.stream_playground(
            topic=payload.topic,
            mode=payload.mode,
            messages=payload.messages or [],
            grade=resolved_grade,
            harder_mode=payload.harder_mode,
            context=merged_context,
        ):
            # Encode newlines so they survive SSE line-splitting on the client
            encoded = chunk.replace('\n', '\\n')
            yield f"data: {encoded}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/explore")
async def playground_explore(
    payload: PlaygroundRequest, current_user: CurrentUser, db: DBSession
):
    """Non-streaming playground exploration."""
    if payload.mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Mode must be one of: {', '.join(VALID_MODES)}")

    org_id = _parse_org_id((payload.context or {}).get("org_id"))
    points_service = PointsService()
    feature_key = MODE_FEATURE_KEY.get(payload.mode, "playground_rapid_mcq")
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key=feature_key, db=db, org_id=org_id)

    # Calculate cost: base + harder_mode bonus
    base_r = await db.execute(select(PointCost).where(PointCost.action == "playground_explore"))
    base_pc = base_r.scalars().first()
    total_cost = base_pc.cost if base_pc else 2
    total_xp = (base_pc.xp_reward or 0) if base_pc else 0
    if payload.harder_mode:
        h_r = await db.execute(select(PointCost).where(PointCost.action == "playground_harder"))
        h_pc = h_r.scalars().first()
        if h_pc:
            total_cost += h_pc.cost
            total_xp += h_pc.xp_reward or 0
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=total_cost, xp_override=total_xp, org_id=org_id,
    )

    ai = get_ai_service()
    _role, resolved_grade, resolved_board, resolved_sm = await _resolve_context(payload, current_user, db)
    merged_context = dict(payload.context or {})
    merged_context["student_mode"] = resolved_sm
    if resolved_board is not None:
        merged_context["board"] = resolved_board

    response = await ai.playground_explore(
        topic=payload.topic,
        mode=payload.mode,
        messages=payload.messages or [],
        grade=resolved_grade,
        harder_mode=payload.harder_mode,
        context=merged_context,
    )
    return {"response": response, "mode": payload.mode, "topic": payload.topic}


# ── The Sandbox — /setup ─────────────────────────────────────────────────────

@router.post("/setup")
async def playground_setup(
    payload: SetupRequest, current_user: CurrentUser, db: DBSession
):
    """Topic interpreter: returns topic-specific controls + scene setup for the Sandbox."""
    ai = get_ai_service()
    return await ai.playground_setup(payload.topic)


# ── The Sandbox — /simulate ───────────────────────────────────────────────────

@router.post("/simulate")
async def playground_simulate(
    payload: SimulateRequest, current_user: CurrentUser, db: DBSession
):
    """Sandbox: send slider variables + control context → AI returns analysis + visual_directives."""
    ai = get_ai_service()
    controls_raw = [c.model_dump() for c in payload.controls] if payload.controls else None
    return await ai.playground_simulate(payload.topic, payload.variables, controls_raw)


# ── Flash-Duel — /cards ───────────────────────────────────────────────────────

@router.post("/cards")
async def playground_cards(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Flash-Duel: generate 5 gamified flashcards (question/answer/hint/points)."""
    ai = get_ai_service()
    cards = await ai.playground_flashcards(payload.topic)
    return {"cards": cards, "topic": payload.topic}


# ── Quest-Line — /quest ───────────────────────────────────────────────────────

@router.post("/quest")
async def playground_quest(
    payload: QuestRequest, current_user: CurrentUser, db: DBSession
):
    """Quest-Line: advance a branching narrative RPG. Pass history + current choice."""
    ai = get_ai_service()
    return await ai.playground_quest(payload.topic, payload.history, payload.choice)


# ── The Mirror — /chat ────────────────────────────────────────────────────────

@router.post("/chat")
async def playground_mirror_chat(
    payload: MirrorChatRequest, current_user: CurrentUser, db: DBSession
):
    """The Mirror: LLM adopts an educational persona and chats in character."""
    ai = get_ai_service()
    response = await ai.playground_mirror_chat(
        payload.topic, payload.persona, payload.messages
    )
    return {"response": response, "persona": payload.persona}


# ── Match Mania + Concept Connect — /match-pairs ─────────────────────────────

@router.post("/match-pairs")
async def playground_match_pairs(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate 6 term-definition pairs for Match Mania and Concept Connect. Cost: 2 pts."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    pairs = await ai.playground_match_pairs(payload.topic, role, grade, board, student_mode)
    return {"pairs": pairs, "topic": payload.topic}


# ── Swipe & Sort — /swipe-facts ──────────────────────────────────────────────

@router.post("/swipe-facts")
async def playground_swipe_facts(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate 10 true/false facts for Swipe & Sort. Cost: 2 pts."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    facts = await ai.playground_swipe_facts(payload.topic, role, grade, board, student_mode)
    return {"facts": facts, "topic": payload.topic}


# ── Speed Blitz — /speed-quiz ────────────────────────────────────────────────

@router.post("/speed-quiz")
async def playground_speed_quiz(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate 10 MCQ questions for Speed Blitz. Cost: 2 pts. Requires playground_rapid_mcq."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="playground_rapid_mcq", db=db, org_id=org_id)
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    result = await ai.playground_speed_quiz(payload.topic, role, grade, board, student_mode)
    return {"questions": result.get("questions", []),
            "challenger": result.get("challenger", {}),
            "topic": payload.topic}


# ── RolePlay Challenge — /roleplay ──────────────────────────────────────────

@router.post("/roleplay")
async def playground_roleplay(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate an immersive roleplay scenario. Cost: 2 pts. Requires playground_roleplay."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="playground_roleplay", db=db, org_id=org_id)
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    scenario = await ai.playground_roleplay(payload.topic, role, grade, board, student_mode)
    return {"scenario": scenario, "topic": payload.topic}


# ── Imagine — /imagine + /imagine-evaluate ──────────────────────────────────

@router.post("/imagine")
async def playground_imagine(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate a creative 'What if?' scenario. Cost: 2 pts. Requires playground_imagine."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="playground_imagine", db=db, org_id=org_id)
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    scenario = await ai.playground_imagine(payload.topic, role, grade, board, student_mode)
    return {"scenario": scenario, "topic": payload.topic}


@router.post("/imagine-evaluate")
async def playground_imagine_evaluate(
    payload: ImagineEvalRequest, current_user: CurrentUser, db: DBSession
):
    """Evaluate a creative open-ended answer. Cost: 2 pts. Requires playground_imagine."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="playground_imagine", db=db, org_id=org_id)
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    evaluation = await ai.playground_imagine_evaluate(
        payload.topic, payload.question, payload.answer,
        payload.max_points, role, grade, board, student_mode,
    )
    return {"evaluation": evaluation}


# ── Myth Busters — /mythbusters ─────────────────────────────────────────────

@router.post("/mythbusters")
async def playground_mythbusters(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate a myth + 8 evidence cards for Myth Busters. Cost: 2 pts."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    result = await ai.playground_mythbusters(payload.topic, role, grade, board, student_mode)
    return result


# ── Cascade Quiz — /cascade-quiz ────────────────────────────────────────────

@router.post("/cascade-quiz")
async def playground_cascade_quiz(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate a branching knowledge tree (depth 3, 7 questions). Cost: 2 pts."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    result = await ai.playground_cascade_quiz(payload.topic, role, grade, board, student_mode)
    return result


# ── Time Warp — /timeline ───────────────────────────────────────────────────

@router.post("/timeline")
async def playground_timeline(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate a timeline with 6-8 events + challenges. Cost: 2 pts."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    result = await ai.playground_timeline(payload.topic, role, grade, board, student_mode)
    return result


# ── Build-a-Diagram — /diagram ──────────────────────────────────────────────

@router.post("/diagram")
async def playground_diagram(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate a concept diagram with nodes, connections, and zones. Cost: 2 pts."""
    org_id = _parse_org_id(payload.org_id)
    points_service = PointsService()
    await points_service.deduct_custom(
        user_id=current_user.id, action="playground_explore", db=db,
        cost_override=2, xp_override=5, org_id=org_id,
    )
    role, grade, board, student_mode = await _resolve_context(payload, current_user, db)
    ai = get_ai_service()
    result = await ai.playground_diagram(payload.topic, role, grade, board, student_mode)
    return result
