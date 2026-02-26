from fastapi import APIRouter
from app.dependencies import DBSession, CurrentUser
from app.schemas.ai import AudioQARequest, AudioQAResponse
from app.services.ai_service import AIService
from app.services.points_service import PointsService

router = APIRouter()


@router.post("/qa", response_model=AudioQAResponse)
async def audio_qa(payload: AudioQARequest, current_user: CurrentUser, db: DBSession):
    """
    Ask a question and get an AI response (text + optional audio).
    The response can be used for audio playback on the frontend.
    """
    points_service = PointsService()
    await points_service.deduct(user_id=current_user.id, action="basic_chat", db=db)

    ai = AIService()
    response_text = await ai.chat(
        messages=[{"role": "user", "content": payload.question}],
        context=payload.context,
    )

    return AudioQAResponse(
        text_response=response_text,
        audio_path=None,  # Audio TTS would be generated here in a real implementation
        language=payload.language,
        points_used=1,
    )
