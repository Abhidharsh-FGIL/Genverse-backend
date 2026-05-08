from io import BytesIO

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from app.dependencies import DBSession, CurrentUser
from app.schemas.ai import AudioQARequest, AudioQAResponse
from app.services.ai_service import AIService, get_ai_service
from app.services.points_service import PointsService
from app.services.audiobook_service import (
    clean_narration_text, _resolve_voice,
    _synthesize_edge_tts, _synthesize_gtts_fallback,
    VOICE_MAP,
)

router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    voice_profile: str = "female_professional"
    rate: str = "+35%"
    pitch: str = "+0Hz"


@router.post("/tts")
async def text_to_speech(payload: TTSRequest, current_user: CurrentUser):
    """Synthesize text to MP3 audio using edge-tts (or gTTS fallback).

    Returns raw MP3 bytes with Content-Type: audio/mpeg.
    Used by the AI assistant's "Read Aloud" feature.
    """
    # Clean markdown/HTML from the text for natural narration
    cleaned = clean_narration_text(payload.text)
    if not cleaned.strip():
        return Response(content=b"", media_type="audio/mpeg", status_code=204)

    # Resolve voice
    voice_id = _resolve_voice(payload.language, payload.voice_profile)

    # Synthesize
    try:
        audio_bytes = await _synthesize_edge_tts(
            text=cleaned,
            voice=voice_id,
            rate=payload.rate,
            pitch=payload.pitch,
        )
    except Exception:
        # Fallback to gTTS
        gtts_map = {
            "en": "en", "hi": "hi", "ta": "ta", "te": "te",
            "es": "es", "fr": "fr", "de": "de", "ar": "ar",
            "zh": "zh-CN", "ja": "ja", "ko": "ko", "pt": "pt",
        }
        try:
            audio_bytes = await _synthesize_gtts_fallback(
                cleaned, gtts_map.get(payload.language, "en")
            )
        except Exception:
            return Response(content=b"", media_type="audio/mpeg", status_code=500)

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Length": str(len(audio_bytes))},
    )


@router.post("/qa", response_model=AudioQAResponse)
async def audio_qa(payload: AudioQARequest, current_user: CurrentUser, db: DBSession):
    """
    Ask a question and get an AI response (text + optional audio).
    The response can be used for audio playback on the frontend.
    """
    points_service = PointsService()
    await points_service.check_and_increment_usage(user_id=current_user.id, feature_key="ai_chat", db=db)
    await points_service.deduct(user_id=current_user.id, action="basic_chat", db=db)

    ai = get_ai_service()
    response_text = await ai.chat(
        messages=[{"role": "user", "content": payload.question}],
        context=payload.context,
    )

    return AudioQAResponse(
        text_response=response_text,
        audio_path=None,
        language=payload.language,
        points_used=1,
    )
