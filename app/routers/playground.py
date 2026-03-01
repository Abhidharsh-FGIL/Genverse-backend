from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.dependencies import DBSession, CurrentUser
from app.schemas.ai import PlaygroundRequest
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()

VALID_MODES = {"experiment", "play", "challenge", "imagine"}


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


class ImagineEvalRequest(BaseModel):
    topic: str
    question: str
    answer: str
    max_points: int = 20
    role: str = "student"
    grade: int | None = None


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

    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="playground_explore", db=db)

    ai = AIService()

    async def event_stream():
        async for chunk in ai.stream_playground(
            topic=payload.topic,
            mode=payload.mode,
            messages=payload.messages or [],
            grade=payload.grade,
            harder_mode=payload.harder_mode,
            context=payload.context,
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

    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="playground_explore", db=db)

    ai = AIService()
    response = await ai.playground_explore(
        topic=payload.topic,
        mode=payload.mode,
        messages=payload.messages or [],
        grade=payload.grade,
        harder_mode=payload.harder_mode,
        context=payload.context,
    )
    return {"response": response, "mode": payload.mode, "topic": payload.topic}


# ── The Sandbox — /setup ─────────────────────────────────────────────────────

@router.post("/setup")
async def playground_setup(
    payload: SetupRequest, current_user: CurrentUser, db: DBSession
):
    """Topic interpreter: returns topic-specific controls + scene setup for the Sandbox."""
    ai = AIService()
    return await ai.playground_setup(payload.topic)


# ── The Sandbox — /simulate ───────────────────────────────────────────────────

@router.post("/simulate")
async def playground_simulate(
    payload: SimulateRequest, current_user: CurrentUser, db: DBSession
):
    """Sandbox: send slider variables + control context → AI returns analysis + visual_directives."""
    ai = AIService()
    controls_raw = [c.model_dump() for c in payload.controls] if payload.controls else None
    return await ai.playground_simulate(payload.topic, payload.variables, controls_raw)


# ── Flash-Duel — /cards ───────────────────────────────────────────────────────

@router.post("/cards")
async def playground_cards(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Flash-Duel: generate 5 gamified flashcards (question/answer/hint/points)."""
    ai = AIService()
    cards = await ai.playground_flashcards(payload.topic)
    return {"cards": cards, "topic": payload.topic}


# ── Quest-Line — /quest ───────────────────────────────────────────────────────

@router.post("/quest")
async def playground_quest(
    payload: QuestRequest, current_user: CurrentUser, db: DBSession
):
    """Quest-Line: advance a branching narrative RPG. Pass history + current choice."""
    ai = AIService()
    return await ai.playground_quest(payload.topic, payload.history, payload.choice)


# ── The Mirror — /chat ────────────────────────────────────────────────────────

@router.post("/chat")
async def playground_mirror_chat(
    payload: MirrorChatRequest, current_user: CurrentUser, db: DBSession
):
    """The Mirror: LLM adopts an educational persona and chats in character."""
    ai = AIService()
    response = await ai.playground_mirror_chat(
        payload.topic, payload.persona, payload.messages
    )
    return {"response": response, "persona": payload.persona}


# ── Match Mania + Concept Connect — /match-pairs ─────────────────────────────

@router.post("/match-pairs")
async def playground_match_pairs(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate 6 term-definition pairs for Match Mania and Concept Connect."""
    ai = AIService()
    pairs = await ai.playground_match_pairs(payload.topic, payload.role, payload.grade)
    return {"pairs": pairs, "topic": payload.topic}


# ── Swipe & Sort — /swipe-facts ──────────────────────────────────────────────

@router.post("/swipe-facts")
async def playground_swipe_facts(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate 10 true/false facts for Swipe & Sort."""
    ai = AIService()
    facts = await ai.playground_swipe_facts(payload.topic, payload.role, payload.grade)
    return {"facts": facts, "topic": payload.topic}


# ── Speed Blitz — /speed-quiz ────────────────────────────────────────────────

@router.post("/speed-quiz")
async def playground_speed_quiz(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate 10 MCQ questions for Speed Blitz."""
    ai = AIService()
    result = await ai.playground_speed_quiz(payload.topic, payload.role, payload.grade)
    return {"questions": result.get("questions", []),
            "challenger": result.get("challenger", {}),
            "topic": payload.topic}


# ── RolePlay Challenge — /roleplay ──────────────────────────────────────────

@router.post("/roleplay")
async def playground_roleplay(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate an immersive roleplay scenario with character and dialogue choices."""
    ai = AIService()
    scenario = await ai.playground_roleplay(payload.topic, payload.role, payload.grade)
    return {"scenario": scenario, "topic": payload.topic}


# ── Imagine — /imagine + /imagine-evaluate ──────────────────────────────────

@router.post("/imagine")
async def playground_imagine(
    payload: FlashcardsRequest, current_user: CurrentUser, db: DBSession
):
    """Generate a creative 'What if?' scenario with open-ended prompts."""
    ai = AIService()
    scenario = await ai.playground_imagine(payload.topic, payload.role, payload.grade)
    return {"scenario": scenario, "topic": payload.topic}


@router.post("/imagine-evaluate")
async def playground_imagine_evaluate(
    payload: ImagineEvalRequest, current_user: CurrentUser, db: DBSession
):
    """Evaluate a creative open-ended answer."""
    ai = AIService()
    evaluation = await ai.playground_imagine_evaluate(
        payload.topic, payload.question, payload.answer,
        payload.max_points, payload.role, payload.grade
    )
    return {"evaluation": evaluation}
