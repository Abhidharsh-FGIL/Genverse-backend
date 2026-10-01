"""Typed, normalised AI Assistant payloads (audit I1, I7; plan §1).

The wire format is unchanged: SendMessageRequest still carries `context` and
`chat_settings` as plain JSON objects, so old clients keep working. Routers
call `normalize_context()` / `normalize_settings()` immediately, and every
downstream function receives these typed models instead of raw dicts.

Normalisation rules:
  - documented legacy values are mapped silently (brief→small, medium→summary,
    tutor/helpful→mentor, language names → codes);
  - any other unrecognised value falls back to the field default and logs a
    warning — never silently.
"""
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from app.prompts.learner import resolve_board_key

logger = logging.getLogger(__name__)

Language = Literal["en", "hi", "ta", "ar", "te", "fr", "de", "es", "zh", "pt"]
Difficulty = Literal["easy", "medium", "hard", "expert"]
ContentLength = Literal["small", "summary", "detailed", "deep-dive"]
Personality = Literal["mentor", "coach", "professor", "friend", "technical-expert"]

LANGUAGE_CODES: tuple[str, ...] = Language.__args__  # type: ignore[attr-defined]
DIFFICULTIES: tuple[str, ...] = Difficulty.__args__  # type: ignore[attr-defined]
CONTENT_LENGTHS: tuple[str, ...] = ContentLength.__args__  # type: ignore[attr-defined]
PERSONALITIES: tuple[str, ...] = Personality.__args__  # type: ignore[attr-defined]

# Single source of truth for defaults (audit I1, plan §1.3). Mirrored in
# FE/hooks/use-chat-settings.ts, BE/models/ai.py and the e1f2a3b4c5d6 migration.
CHAT_SETTINGS_DEFAULTS: dict[str, Any] = {
    "student_mode": True,
    "difficulty": "medium",
    "content_length": "summary",
    "personality": "mentor",
    "explain_3ways": False,
    "examples": False,
    "output_mode": "text",
    "mind_map": False,
    "infographic": False,
    "video_refs": False,
    "practice": False,
    "followup": False,
    "next_steps": False,
}

_LEGACY_CONTENT_LENGTH = {"brief": "small", "medium": "summary"}
_LEGACY_PERSONALITY = {"tutor": "mentor", "helpful": "mentor"}
_LANGUAGE_NAME_TO_CODE = {
    "english": "en", "hindi": "hi", "tamil": "ta", "arabic": "ar", "telugu": "te",
    "french": "fr", "german": "de", "spanish": "es", "chinese": "zh", "portuguese": "pt",
}


def _warn_unknown(field: str, value: Any, fallback: Any) -> Any:
    logger.warning("ai_payload.unknown_value field=%s value=%r fallback=%r", field, value, fallback)
    return fallback


def normalize_language(value: Any) -> Optional[str]:
    """Map a language code or name ("ta", "Tamil", "Tamil (தமிழ்)") to a code; None if absent, "en" if unknown."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    # Unknown → English instruction + warning, never "no instruction" (Phase 2 addition 3).
    if not isinstance(value, str):
        return _warn_unknown("language", value, "en")
    v = value.strip().lower()
    if v in LANGUAGE_CODES:
        return v
    base = v.split("(")[0].strip()
    if base in _LANGUAGE_NAME_TO_CODE:
        return _LANGUAGE_NAME_TO_CODE[base]
    return _warn_unknown("language", value, "en")


def _enum(field: str, value: Any, allowed: tuple[str, ...], default: str, legacy: dict[str, str] | None = None) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in allowed:
            return v
        if legacy and v in legacy:
            return legacy[v]
    return _warn_unknown(field, value, default)


def _bool(field: str, value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return _warn_unknown(field, value, default)


class InlineDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    filename: str = ""
    text: str = ""


class AIContextPayload(BaseModel):
    """Global learning context sent by the frontend as `context`."""
    model_config = ConfigDict(extra="ignore")

    student_mode: bool = True
    grade: Optional[int] = None
    board: Optional[str] = None          # raw label as the user chose it
    subject: Optional[str] = None
    # None = not sent; the builder then uses English for user-facing calls.
    language: Optional[str] = None
    selected_files: list[str] = []
    library_file_id: Optional[str] = None
    inline_docs: list[InlineDoc] = []
    org_id: Optional[str] = None
    workspace_type: Optional[str] = None
    # Deprecated for chat (plan D2 / audit I3): accepted, never used in prompts.
    tone: Optional[str] = None
    difficulty: Optional[str] = None
    output_mode: Optional[str] = None

    @field_validator("student_mode", mode="before")
    @classmethod
    def _v_student_mode(cls, v):
        return _bool("context.student_mode", v, True)

    @field_validator("grade", mode="before")
    @classmethod
    def _v_grade(cls, v):
        if v is None or v == "":
            return None
        try:
            g = int(v)
        except (TypeError, ValueError):
            return _warn_unknown("context.grade", v, None)
        if 1 <= g <= 12:
            return g
        return _warn_unknown("context.grade", v, None)

    @field_validator("board", "subject", mode="before")
    @classmethod
    def _v_text(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return s[:100] or None

    @field_validator("language", mode="before")
    @classmethod
    def _v_language(cls, v):
        return normalize_language(v)

    @field_validator("selected_files", mode="before")
    @classmethod
    def _v_files(cls, v):
        return [str(x) for x in v] if isinstance(v, list) else []

    @field_validator("inline_docs", mode="before")
    @classmethod
    def _v_inline(cls, v):
        return v if isinstance(v, list) else []

    @property
    def board_key(self) -> Optional[str]:
        """Registry key for the board (CBSE/ICSE/STATE/TN_STATE/IB/CAMBRIDGE) or None."""
        return resolve_board_key(self.board)


class ChatSettingsPayload(BaseModel):
    """Per-chat settings, merged from the DB row and the client overlay."""
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    student_mode: bool = CHAT_SETTINGS_DEFAULTS["student_mode"]
    difficulty: str = CHAT_SETTINGS_DEFAULTS["difficulty"]
    content_length: str = CHAT_SETTINGS_DEFAULTS["content_length"]
    personality: str = CHAT_SETTINGS_DEFAULTS["personality"]
    explain_3ways: bool = False
    examples: bool = False
    output_mode: str = "text"            # kept, not used in prompts from Phase 2 (plan D3)
    mind_map: bool = False
    infographic: bool = False
    video_refs: bool = False
    practice: bool = False
    followup: bool = False
    next_steps: bool = False

    @field_validator("difficulty", mode="before")
    @classmethod
    def _v_difficulty(cls, v):
        return _enum("difficulty", v, DIFFICULTIES, CHAT_SETTINGS_DEFAULTS["difficulty"])

    @field_validator("content_length", mode="before")
    @classmethod
    def _v_length(cls, v):
        return _enum("content_length", v, CONTENT_LENGTHS, CHAT_SETTINGS_DEFAULTS["content_length"], _LEGACY_CONTENT_LENGTH)

    @field_validator("personality", mode="before")
    @classmethod
    def _v_personality(cls, v):
        return _enum("personality", v, PERSONALITIES, CHAT_SETTINGS_DEFAULTS["personality"], _LEGACY_PERSONALITY)

    @field_validator("output_mode", mode="before")
    @classmethod
    def _v_output_mode(cls, v):
        return _enum("output_mode", v, ("text", "structured", "bullets"), "text")

    @field_validator(
        "student_mode", "explain_3ways", "examples", "mind_map", "infographic",
        "video_refs", "practice", "followup", "next_steps", mode="before",
    )
    @classmethod
    def _v_bools(cls, v, info):
        return _bool(info.field_name, v, CHAT_SETTINGS_DEFAULTS[info.field_name])


def normalize_context(raw: Any) -> Optional[AIContextPayload]:
    """Raw `context` (dict / payload / None) → AIContextPayload, or None when absent or empty."""
    if raw is None:
        return None
    if isinstance(raw, AIContextPayload):
        return raw
    if not isinstance(raw, dict) or not raw:
        return None
    return AIContextPayload.model_validate(raw)


def normalize_settings(raw: Any, base: Optional[ChatSettingsPayload] = None) -> Optional[ChatSettingsPayload]:
    """Raw `chat_settings` → ChatSettingsPayload.

    `base` (usually the DB row) is overlaid with the raw values, so a partial
    client dict only changes the keys it contains. Returns None when there is
    neither a base nor a non-empty raw dict.
    """
    if isinstance(raw, ChatSettingsPayload):
        return raw
    overlay = raw if isinstance(raw, dict) else {}
    if base is None and not overlay:
        return None
    merged = {**(base.model_dump() if base else {}), **overlay}
    return ChatSettingsPayload.model_validate(merged)
