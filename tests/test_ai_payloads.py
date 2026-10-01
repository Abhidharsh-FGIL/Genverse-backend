"""Typed AI payload normalisation (plan §1; audit I1, I7, G4).

Pure functions — no DB, no network.
"""
import logging
import uuid

import pytest
from sqlalchemy.dialects import postgresql

from app.models.ai import AiChatSetting
from app.prompts.learner import BOARD_REGISTRY, board_profile_key, resolve_board_key
from app.schemas.ai_payloads import (
    CHAT_SETTINGS_DEFAULTS,
    AIContextPayload,
    ChatSettingsPayload,
    normalize_context,
    normalize_language,
    normalize_settings,
)


# ── legacy values map silently ───────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [("brief", "small"), ("medium", "summary"), ("deep-dive", "deep-dive")])
def test_legacy_content_length(raw, expected, caplog):
    with caplog.at_level(logging.WARNING):
        assert normalize_settings({"content_length": raw}).content_length == expected
    assert not caplog.records, "documented legacy values must not warn"


@pytest.mark.parametrize("raw", ["tutor", "helpful", "Helpful"])
def test_legacy_personality(raw, caplog):
    with caplog.at_level(logging.WARNING):
        assert normalize_settings({"personality": raw}).personality == "mentor"
    assert not caplog.records


# ── unknown values fall back and warn (never silent) ─────────────────────────

@pytest.mark.parametrize("field, value", [
    ("difficulty", "insane"),
    ("content_length", "novel"),
    ("personality", "pirate"),
    ("output_mode", "haiku"),
])
def test_unknown_setting_falls_back_with_warning(field, value, caplog):
    with caplog.at_level(logging.WARNING, logger="app.schemas.ai_payloads"):
        cs = normalize_settings({field: value})
    assert getattr(cs, field) == ChatSettingsPayload().__getattribute__(field)
    assert any("ai_payload.unknown_value" in r.getMessage() and field in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("grade", [0, 13, -1, "twelve"])
def test_out_of_range_grade_dropped_with_warning(grade, caplog):
    with caplog.at_level(logging.WARNING, logger="app.schemas.ai_payloads"):
        assert normalize_context({"grade": grade}).grade is None
    assert caplog.records


def test_unknown_language_falls_back_to_english_with_warning(caplog):
    """Phase 2 addition 3: English instruction + warning, never no instruction."""
    with caplog.at_level(logging.WARNING, logger="app.schemas.ai_payloads"):
        assert normalize_language("klingon") == "en"
        assert normalize_context({"language": "ml"}).language == "en"
    assert len(caplog.records) == 2


# ── language codes and names (audit I9) ──────────────────────────────────────

@pytest.mark.parametrize("raw, code", [
    ("ta", "ta"), ("TA", "ta"), ("Tamil", "ta"), ("Tamil (தமிழ்)", "ta"),
    ("English", "en"), ("hindi", "hi"), ("Arabic (عربي)", "ar"), (None, None), ("", None),
])
def test_language_normalisation(raw, code):
    assert normalize_language(raw) == code


# ── valid values pass through unchanged ──────────────────────────────────────

def test_valid_settings_are_identity():
    raw = {"difficulty": "expert", "content_length": "deep-dive", "personality": "technical-expert",
           "explain_3ways": True, "examples": True, "student_mode": False, "output_mode": "bullets"}
    cs = normalize_settings(raw)
    for k, v in raw.items():
        assert getattr(cs, k) == v


def test_string_booleans_accepted():
    cs = normalize_settings({"explain_3ways": "true", "student_mode": "false"})
    assert cs.explain_3ways is True and cs.student_mode is False


# ── None / empty handling (callers that send nothing get no prompt section) ──

def test_absent_payloads_are_none():
    assert normalize_context(None) is None
    assert normalize_context({}) is None
    assert normalize_settings(None) is None
    assert normalize_settings({}) is None


def test_overlay_on_db_base():
    base = ChatSettingsPayload(personality="professor", content_length="detailed")
    merged = normalize_settings({"content_length": "small"}, base)
    assert merged.personality == "professor"          # untouched key kept from base
    assert merged.content_length == "small"           # overlay wins


def test_settings_from_orm_row_with_legacy_values():
    row = AiChatSetting(personality="helpful", content_length="medium", difficulty="hard",
                        examples=True, student_mode=False)
    cs = ChatSettingsPayload.model_validate(row)
    assert (cs.personality, cs.content_length, cs.difficulty) == ("mentor", "summary", "hard")
    assert cs.examples is True and cs.student_mode is False


def test_unflushed_orm_row_gets_payload_defaults():
    cs = ChatSettingsPayload.model_validate(AiChatSetting())   # python defaults apply only at flush
    assert cs.model_dump() == CHAT_SETTINGS_DEFAULTS


# ── defaults are aligned everywhere (audit I1) ───────────────────────────────

def test_payload_defaults_match_declared_defaults():
    assert ChatSettingsPayload().model_dump() == CHAT_SETTINGS_DEFAULTS


def test_model_column_defaults_match_payload_defaults():
    cols = AiChatSetting.__table__.c
    for key, value in CHAT_SETTINGS_DEFAULTS.items():
        assert cols[key].default.arg == value, f"AiChatSetting.{key} default drifted"


def test_deprecated_context_fields_accepted():
    ctx = normalize_context({"tone": "formal", "difficulty": "hard", "output_mode": "default", "language": "en"})
    assert isinstance(ctx, AIContextPayload) and ctx.language == "en"


# ── board registry (amendment 6) ─────────────────────────────────────────────

@pytest.mark.parametrize("label, key", [
    ("CBSE", "CBSE"), ("cbse (ncert)", "CBSE"), ("ICSE", "ICSE"), ("State Board", "STATE"),
    ("IB", "IB"), ("Cambridge", "CAMBRIDGE"),
    ("Tamil Nadu State Board", "TN_STATE"), ("Samacheer Kalvi", "TN_STATE"),
    ("Kerala Board", None), (None, None), ("", None),
])
def test_board_resolution(label, key):
    assert resolve_board_key(label) == key
    assert normalize_context({"board": label}) is None or normalize_context({"board": label}).board_key == key


def test_tn_state_registered_before_generic_state():
    keys = [b.key for b in BOARD_REGISTRY]
    assert keys.index("TN_STATE") < keys.index("STATE")
    assert board_profile_key("TN_STATE") == "TN_STATE"   # own Samacheer Kalvi overlay (Phase 3)


# ── G4: settings are only ever loaded through the owning chat ────────────────

def test_owned_settings_query_filters_by_user():
    from app.routers.ai_assistant import _owned_settings_query
    sql = str(_owned_settings_query(uuid.uuid4(), uuid.uuid4()).compile(dialect=postgresql.dialect()))
    assert "JOIN ai_chats" in sql
    assert "ai_chats.user_id" in sql
    assert "ai_chat_settings.chat_id" in sql
