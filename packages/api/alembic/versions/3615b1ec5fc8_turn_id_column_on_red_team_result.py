"""turn id column on red team result

Every red-team probe (out_of_scope, injection, workspace_isolation) already
produces a real Turn per case, via _ask_in_fresh_conversation() in
scripts/run_eval.py: the question was actually asked, retrieval actually
ran, an answer was actually generated. Until now that Turn was discarded
the moment the pass/fail check read it, so a failure could be reported but
never inspected. Section 11 requires "every failure linked to its full
trace in the web app's /trace view", which needs a turn id to link to.

Nullable, unlike run_id in the prior migration: existing rows genuinely
cannot be backfilled (the Turn each one produced was never persisted as a
foreign key anywhere, only read and discarded), but that is a narrower,
disclosable gap for rows already written, not a reason to clear the table
again. Every row scripts/run_eval.py writes from this commit forward sets
this column; claims.py and the /eval page only ever read the latest run_id
anyway, so old rows with a null turn_id fall out of view on the next real
run rather than needing to be reconciled.

Revision ID: 3615b1ec5fc8
Revises: 285e78efc343
Create Date: 2026-09-22 03:17:48.672402
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3615b1ec5fc8"
down_revision: str | None = "285e78efc343"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None


def upgrade() -> None:
    op.add_column("red_team_result", sa.Column("turn_id", sa.Uuid(), nullable=True))
    op.create_index(
        op.f("ix_red_team_result_turn_id"), "red_team_result", ["turn_id"], unique=False
    )
    # Unnamed, matching every other foreign key in this project's migration
    # history (see 328787185122's own ForeignKeyConstraint calls): none of
    # them pass an explicit name, and this project's metadata has no
    # naming_convention configured, so Postgres assigns its own default
    # constraint name rather than this migration inventing one that
    # nothing else in the schema would match.
    op.create_foreign_key(
        None,
        "red_team_result",
        "turn",
        ["turn_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("red_team_result_turn_id_fkey", "red_team_result", type_="foreignkey")
    op.drop_index(op.f("ix_red_team_result_turn_id"), table_name="red_team_result")
    op.drop_column("red_team_result", "turn_id")
