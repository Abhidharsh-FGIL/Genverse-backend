"""add platform_stats and payment_intents tables

Revision ID: b4c7e2f19a83
Revises: e9f1a2b3c4d5
Create Date: 2026-09-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = 'b4c7e2f19a83'
down_revision: Union[str, None] = 'e9f1a2b3c4d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(conn, name: str) -> bool:
    return bool(conn.execute(text(
        "SELECT 1 FROM information_schema.tables WHERE table_name = :t"
    ), {"t": name}).scalar())


def upgrade() -> None:
    # Both tables can already exist here — app/database.py's own startup
    # routine self-creates them idempotently the moment the app boots on
    # the new code, which commonly happens before anyone gets around to
    # running this migration. Guard on existence so this migration is safe
    # to run whether or not that already fired.
    conn = op.get_bind()

    if not _table_exists(conn, 'platform_stats'):
        op.create_table(
            'platform_stats',
            sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('stats_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('id', name=op.f('pk_platform_stats')),
        )

    if not _table_exists(conn, 'payment_intents'):
        op.create_table(
            'payment_intents',
            sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('merchant_order_id', sa.String(length=80), nullable=False),
            sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column('gateway', sa.String(length=20), nullable=False),
            sa.Column('flow', sa.String(length=20), nullable=False),
            sa.Column('item_type', sa.String(length=30), nullable=False),
            sa.Column('item_id', sa.String(length=50), nullable=False),
            sa.Column('amount_inr', sa.Integer(), nullable=False),
            sa.Column('promo_code', sa.String(length=50), nullable=True),
            sa.Column('promo_discount', sa.Integer(), nullable=False),
            sa.Column('status', sa.String(length=20), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], name=op.f('fk_payment_intents_org_id_organizations'), ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['user_id'], ['profiles.id'], name=op.f('fk_payment_intents_user_id_profiles'), ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id', name=op.f('pk_payment_intents')),
            sa.UniqueConstraint('merchant_order_id', name='uq_payment_intents_merchant_order_id'),
        )
        # database.py's self-create path names these idx_* instead of ix_* —
        # only add the alembic-convention indexes when we're the one creating
        # the table, so we never collide with indexes that already exist.
        op.create_index(op.f('ix_payment_intents_merchant_order_id'), 'payment_intents', ['merchant_order_id'], unique=False)
        op.create_index(op.f('ix_payment_intents_status'), 'payment_intents', ['status'], unique=False)
        op.create_index(op.f('ix_payment_intents_user_id'), 'payment_intents', ['user_id'], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    if _table_exists(conn, 'payment_intents'):
        op.drop_table('payment_intents')
    if _table_exists(conn, 'platform_stats'):
        op.drop_table('platform_stats')
