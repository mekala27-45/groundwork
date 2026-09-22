"""run id column on eval run and red team result

scripts/run_eval.py (build order step 20) can be, and already has been,
run more than once against the same database: once for the real recorded
evaluation, and a second time after scripts/seed_demo_workspaces.py's own
cascade delete bug (see that script's docstring) was fixed and verified.
Neither table had any column identifying which invocation wrote a given
row, only an independent run_at timestamp per row, so a query with no
run-scoping filter silently aggregated across every run that had ever
executed. eval_run happened to hold only one run's worth of rows by the
time this was noticed, because EvalRun is workspace scoped and the
reseed's cascade delete (correctly) wipes it; red_team_result has no
workspace_id column at all, so it held two runs' worth, 58 rows where a
single real run produces 29, silently doubling every count a reader would
compute from it.

The existing rows in both tables cannot be assigned a real run_id
retroactively: for eval_run they are already one consistent run and would
only need a single backfilled value, but for red_team_result the two
runs' rows are interleaved with no reliable way to tell them apart after
the fact (both share the same case_id values, only run_at differs, and
run_at was never meant to be parsed as an identity). Since this is
reproducible eval data, not user data, generated fresh by re-running
scripts/seed_demo_workspaces.py, scripts/build_eval_questions.py, and
scripts/run_eval.py in that order, the correct fix is to clear both
tables here and let the next real run repopulate them properly tagged,
rather than inventing a backfill value that would misrepresent which
rows actually belong together.

Revision ID: 285e78efc343
Revises: 9c777f02d948
Create Date: 2026-09-22 02:43:28.952592
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "285e78efc343"
down_revision: str | None = "9c777f02d948"
branch_labels: Sequence[str] | str | None = None
depends_on: Sequence[str] | str | None = None


def upgrade() -> None:
    # Stale, run-undifferentiated rows: see the module docstring for why
    # these cannot be backfilled with a real run_id instead of cleared.
    op.execute("DELETE FROM red_team_result")
    op.execute("DELETE FROM eval_run")

    op.add_column("eval_run", sa.Column("run_id", sa.Uuid(), nullable=False))
    op.create_index(op.f("ix_eval_run_run_id"), "eval_run", ["run_id"], unique=False)

    op.add_column("red_team_result", sa.Column("run_id", sa.Uuid(), nullable=False))
    op.create_index(op.f("ix_red_team_result_run_id"), "red_team_result", ["run_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_red_team_result_run_id"), table_name="red_team_result")
    op.drop_column("red_team_result", "run_id")

    op.drop_index(op.f("ix_eval_run_run_id"), table_name="eval_run")
    op.drop_column("eval_run", "run_id")
