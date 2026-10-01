"""Router-level Phase 1 checks with a fake async DB session (no Postgres).

Covers: settings derived from the model + client overlay (audit §7.4, I1),
ownership enforcement on settings GET/PATCH (audit G4), and the GET creating a
missing default row so the frontend needs no racy "ensure" PATCH (audit G1).
"""
import uuid
from types import SimpleNamespace

import pytest

from app.core.exceptions import NotFoundException
from app.models.ai import AiChat, AiChatSetting
from app.routers import ai_assistant as router_mod
from app.schemas.ai import ChatSettingsUpdate

pytestmark = pytest.mark.asyncio


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class FakeDB:
    """Returns queued results for execute() in order; records adds/commits."""

    def __init__(self, *results):
        self._results = list(results)
        self.added = []
        self.commits = 0

    async def execute(self, _stmt):
        return _Result(self._results.pop(0))

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1

    async def refresh(self, _obj):
        pass


USER = SimpleNamespace(id=uuid.uuid4())


async def test_resolve_settings_overlays_client_on_db_row():
    row = AiChatSetting(personality="professor", content_length="detailed", difficulty="hard",
                        examples=False, student_mode=True, explain_3ways=False)
    db = FakeDB(row)
    cs = await router_mod._resolve_chat_settings(uuid.uuid4(), {"content_length": "brief", "explain_3ways": True}, db)
    assert cs.personality == "professor"          # from DB
    assert cs.difficulty == "hard"                # from DB
    assert cs.content_length == "small"           # client overlay, legacy-mapped
    assert cs.explain_3ways is True               # client overlay


async def test_resolve_settings_none_when_nothing_known():
    assert await router_mod._resolve_chat_settings(uuid.uuid4(), None, FakeDB(None)) is None


async def test_get_settings_rejects_chat_owned_by_someone_else():
    db = FakeDB(None)   # _get_owned_chat finds no chat for this user
    with pytest.raises(NotFoundException):
        await router_mod.get_chat_settings(uuid.uuid4(), USER, db)


async def test_patch_settings_rejects_chat_owned_by_someone_else():
    db = FakeDB(None)
    with pytest.raises(NotFoundException):
        await router_mod.update_chat_settings(uuid.uuid4(), ChatSettingsUpdate(personality="coach"), USER, db)
    assert db.added == [] and db.commits == 0


async def test_get_settings_creates_missing_default_row():
    chat_id = uuid.uuid4()
    db = FakeDB(AiChat(id=chat_id, user_id=USER.id, title="t"), None)
    row = await router_mod.get_chat_settings(chat_id, USER, db)
    assert isinstance(row, AiChatSetting) and row.chat_id == chat_id
    assert db.added == [row] and db.commits == 1


async def test_patch_settings_normalises_values():
    chat_id = uuid.uuid4()
    existing = AiChatSetting(chat_id=chat_id, personality="mentor", content_length="summary")
    db = FakeDB(AiChat(id=chat_id, user_id=USER.id, title="t"), existing)
    row = await router_mod.update_chat_settings(
        chat_id, ChatSettingsUpdate(personality="tutor", content_length="nonsense"), USER, db,
    )
    assert row.personality == "mentor"        # legacy mapped
    assert row.content_length == "summary"    # unknown → default (with warning)
