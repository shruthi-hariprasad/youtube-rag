"""initial schema — baseline for tables created before Alembic

This migration represents the schema as it existed before Alembic was
introduced. When first setting up Alembic on an existing DB, stamp at
this revision rather than running upgrade from scratch.

Revision ID: 0001
Revises:
Create Date: 2026-05-29
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("email", sa.String(), unique=True, index=True, nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "videos",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("youtube_video_id", sa.String(), index=True),
        sa.Column("title", sa.String()),
        sa.Column("channel_name", sa.String()),
        sa.Column("thumbnail_url", sa.String()),
        sa.Column("url", sa.String()),
        sa.Column("transcript_text", sa.Text()),
    )


def downgrade() -> None:
    op.drop_table("videos")
    op.drop_table("users")
