"""add team column to xt_results

Revision ID: b8f2a3c91d47
Revises: 4a771208632f
Create Date: 2026-03-30 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b8f2a3c91d47'
down_revision: Union[str, Sequence[str], None] = '4a771208632f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("xt_results", sa.Column("team", sa.String(120), nullable=True))
    # Backfill existing rows: infer team from the events table via match_id,
    # or fall back to 'Unknown' for season-level rows without a match_id.
    op.execute("""
        UPDATE xt_results
        SET team = sub.team
        FROM (
            SELECT DISTINCT ON (e.match_id) e.match_id, e.team
            FROM events e
        ) sub
        WHERE xt_results.match_id = sub.match_id
          AND xt_results.team IS NULL
    """)
    op.execute("""
        UPDATE xt_results
        SET team = 'Unknown'
        WHERE team IS NULL
    """)
    op.alter_column("xt_results", "team", nullable=False)


def downgrade() -> None:
    op.drop_column("xt_results", "team")
