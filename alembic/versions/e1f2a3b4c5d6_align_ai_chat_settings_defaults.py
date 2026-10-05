"""align AI chat-settings / context defaults with the frontend

Plan PLAN_AI_TAILORING.md Phase 1:
  - audit I1  — one default per field (FE and BE previously disagreed):
                ai_chat_settings.personality=mentor, content_length=summary,
                student_mode=true; ai_context_sessions.student_mode=true.
  - audit P2  — ai_chat_settings.examples defaults to false, and existing rows
                are set to false (there was never a UI for it, so no user chose
                true — plan D6).
  - plan D8   — legacy values rewritten: personality helpful/tutor → mentor,
                content_length medium → summary, brief → small. The API keeps
                accepting the legacy values (app/schemas/ai_payloads.py).

Downgrade restores the previous server defaults only; the data rewrite is not
reversed (the old values are still accepted and mapped on read).

Revision ID: e1f2a3b4c5d6
Revises: d4e5f6a7b8c9
Create Date: 2026-10-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('ai_chat_settings', 'personality', server_default='mentor')
    op.alter_column('ai_chat_settings', 'content_length', server_default='summary')
    op.alter_column('ai_chat_settings', 'difficulty', server_default='medium')
    op.alter_column('ai_chat_settings', 'examples', server_default=sa.false())
    op.alter_column('ai_chat_settings', 'student_mode', server_default=sa.true())
    op.alter_column('ai_context_sessions', 'student_mode', server_default=sa.true())

    op.execute("UPDATE ai_chat_settings SET examples = false WHERE examples IS DISTINCT FROM false")
    op.execute("UPDATE ai_chat_settings SET personality = 'mentor' WHERE personality IN ('helpful', 'tutor')")
    op.execute("UPDATE ai_chat_settings SET content_length = 'summary' WHERE content_length = 'medium'")
    op.execute("UPDATE ai_chat_settings SET content_length = 'small' WHERE content_length = 'brief'")


def downgrade() -> None:
    op.alter_column('ai_context_sessions', 'student_mode', server_default=None)
    op.alter_column('ai_chat_settings', 'student_mode', server_default=None)
    op.alter_column('ai_chat_settings', 'examples', server_default=None)
    op.alter_column('ai_chat_settings', 'difficulty', server_default=None)
    op.alter_column('ai_chat_settings', 'content_length', server_default=None)
    op.alter_column('ai_chat_settings', 'personality', server_default=None)
