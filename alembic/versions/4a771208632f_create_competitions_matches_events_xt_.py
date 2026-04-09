"""create competitions matches events xt_results tables

Revision ID: 4a771208632f
Revises:
Create Date: 2026-03-30 01:03:09.419092

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4a771208632f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Enum types referenced by columns
event_type_enum = sa.Enum("PASS", "CARRY", "SHOT", name="eventtype")
scope_enum = sa.Enum("MATCH", "SEASON", name="scope")


def upgrade() -> None:
    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("competition_name", sa.String(120), nullable=False),
        sa.Column("season_name", sa.String(60), nullable=False),
        sa.Column("statsbomb_competition_id", sa.Integer, nullable=False),
        sa.Column("statsbomb_season_id", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("competition_id", sa.Integer, sa.ForeignKey("competitions.id"), nullable=False),
        sa.Column("home_team", sa.String(120), nullable=False),
        sa.Column("away_team", sa.String(120), nullable=False),
        sa.Column("match_date", sa.Date, nullable=False),
        sa.Column("statsbomb_match_id", sa.Integer, unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("event_type", event_type_enum, nullable=False),
        sa.Column("team", sa.String(120), nullable=False),
        sa.Column("start_zone", sa.String(10)),
        sa.Column("end_zone", sa.String(10)),
        sa.Column("is_goal", sa.Boolean, default=False, nullable=False),
        sa.Column("is_loss", sa.Boolean, default=False, nullable=False),
        sa.Column("start_x", sa.Float),
        sa.Column("start_y", sa.Float),
        sa.Column("end_x", sa.Float),
        sa.Column("end_y", sa.Float),
    )

    op.create_table(
        "xt_results",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("competition_id", sa.Integer, sa.ForeignKey("competitions.id"), nullable=False),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=True),
        sa.Column("zone_name", sa.String(10), nullable=False),
        sa.Column("xt_value", sa.Float, nullable=False),
        sa.Column("direct_goal_pct", sa.Float, nullable=False),
        sa.Column("loss_pct", sa.Float, nullable=False),
        sa.Column("event_count", sa.Integer, nullable=False),
        sa.Column("scope", scope_enum, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("xt_results")
    op.drop_table("events")
    op.drop_table("matches")
    op.drop_table("competitions")
    event_type_enum.drop(op.get_bind())
    scope_enum.drop(op.get_bind())
