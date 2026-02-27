from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.dependencies import DBSession, CurrentUser
from app.schemas.ai import PlaygroundRequest
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()

VALID_MODES = {"experiment", "play", "challenge", "imagine"}


@router.post("/explore/stream")
async def playground_explore_stream(
    payload: PlaygroundRequest, current_user: CurrentUser, db: DBSession
):
    """
    Interactive topic exploration with SSE streaming.
    Modes: experiment | play | challenge | imagine
    """
    if payload.mode not in VALID_MODES:
        from fastapi import HTTPException
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
        from fastapi import HTTPException
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
