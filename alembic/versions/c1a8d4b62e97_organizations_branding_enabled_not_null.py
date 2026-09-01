"""organizations.branding_enabled: backfill nulls, add server default, enforce NOT NULL

Revision ID: c1a8d4b62e97
Revises: b4c7e2f19a83
Create Date: 2026-09-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = 'c1a8d4b62e97'
down_revision: Union[str, None] = 'b4c7e2f19a83'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    # Existing rows created before this column had any default (or inserted
    # through a path that bypassed the ORM's client-side default=False) can
    # still be NULL — backfill before the column can safely become NOT NULL.
    conn.execute(text("UPDATE organizations SET branding_enabled = false WHERE branding_enabled IS NULL"))
    op.alter_column(
        'organizations', 'branding_enabled',
        existing_type=sa.Boolean(),
        server_default=text('false'),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        'organizations', 'branding_enabled',
        existing_type=sa.Boolean(),
        server_default=None,
        nullable=True,
    )
