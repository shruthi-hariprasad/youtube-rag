"""add transcript_segments, summary, suggested_questions, added_at to videos

These columns were previously added ad-hoc by _run_migrations() on every
startup. This migration replaces that approach.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-29
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use IF NOT EXISTS to be safe against partial prior migrations done via
    # the old raw ALTER TABLE approach.
    conn = op.get_bind()
    conn.execute(sa.text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS transcript_segments TEXT"))
    conn.execute(sa.text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS summary TEXT"))
    conn.execute(sa.text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS suggested_questions TEXT"))
    conn.execute(sa.text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS added_at TIMESTAMPTZ DEFAULT now()"))


def downgrade() -> None:
    with op.batch_alter_table("videos") as batch_op:
        batch_op.drop_column("added_at")
        batch_op.drop_column("suggested_questions")
        batch_op.drop_column("summary")
        batch_op.drop_column("transcript_segments")
