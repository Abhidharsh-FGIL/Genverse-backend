"""
vision_ocr_service.py

Google Cloud Vision OCR — used for non-English and auto-detected-language images.
Vision's document_text_detection is a dedicated OCR model with per-language hints
and broad handwriting support (including Tamil and other Indic scripts), so it is
both faster (a single call) and more accurate for those cases than routing every
image through a general-purpose Gemini vision prompt. English stays on the
existing Gemini path in ai_service.py, which is already good at reconstructing
markdown/tables/equations.

Reuses the same GCP service account already configured for Google Cloud Storage
(GCS_CREDENTIALS_PATH) — the Cloud Vision API must be enabled on that GCP project
for this to work; see the project's Vision API console page if calls start
failing with a SERVICE_DISABLED / PermissionDenied error.
"""
from pathlib import Path

from google.cloud import vision

from app.config import settings

# Resolves to genverse-backend/ (the project root, same directory as .env)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_client: vision.ImageAnnotatorClient | None = None


def _resolve_credentials_path() -> str:
    p = Path(settings.GCS_CREDENTIALS_PATH)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p)


def _get_client() -> vision.ImageAnnotatorClient:
    global _client
    if _client is None:
        _client = vision.ImageAnnotatorClient.from_service_account_json(_resolve_credentials_path())
    return _client


def ocr_image(image_bytes: bytes, language_hint: str | None = None) -> tuple[str, str]:
    """Run Vision's document text detection on an image.

    Args:
        image_bytes: raw image data (JPEG/PNG/etc).
        language_hint: ISO 639-1 code (e.g. "ta", "hi") to bias recognition, or
            None to let Vision auto-detect the language.

    Returns:
        (extracted_text, detected_language_code).

    Raises:
        RuntimeError if the Vision API itself reports an error (including if the
        API isn't enabled on the GCP project yet) — callers should catch this and
        fall back to another OCR path rather than surfacing it to the user.
    """
    image = vision.Image(content=image_bytes)
    image_context = vision.ImageContext(language_hints=[language_hint]) if language_hint else None
    response = _get_client().document_text_detection(image=image, image_context=image_context)
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    annotation = response.full_text_annotation
    text = annotation.text or ""

    detected = language_hint or ""
    if annotation.pages and annotation.pages[0].property.detected_languages:
        detected = annotation.pages[0].property.detected_languages[0].language_code

    return text, detected
