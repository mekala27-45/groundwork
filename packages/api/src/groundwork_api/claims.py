"""The claim registry: builds a manifest of real, queried numbers and
renders README.md and RESULTS.md from it, so a published figure traces
back to a live query against this repository's own database and evalset,
never a hand typed guess. Ported from day 3's renderer, per the locked
rule carried forward into this build (Part 2 of the day 5 prompt): the
claim gate is a document renderer, built before any results exist, not a
scanner that checks prose for numbers that merely look plausible.

"Before any results exist" is not a hypothetical here. build_manifest()
runs today, against a database that has no eval run and no red team
result yet, and every key that depends on one of those reports
NOT_YET_RUN rather than inventing a number to fill the gap. As the
retrieval evaluation harness, the verification suite, and the red team
tests land in later build steps, this module grows new manifest keys and
docs/templates/*.tmpl grows new sections that read them; nothing here is
meant to be written once and left behind.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnElement
from sqlmodel import SQLModel, col

from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Document,
    EvalRun,
    RedTeamResult,
    Turn,
    Workspace,
)

RED_TEAM_SUITES = ("injection", "out_of_scope", "workspace_isolation")
"""Matches RedTeamResult.suite's own documented values exactly, and the
order section 11 lists them in reverse (injection last there; led with
here since it is this build's strongest real number, see
docs/templates/RESULTS.md.tmpl)."""

REPO_ROOT = Path(__file__).resolve().parents[4]
EVALSET_QUESTIONS_PATH = REPO_ROOT / "evalset" / "questions.yaml"

NOT_YET_RUN = "not yet run"

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


async def _scalar_count(
    session: AsyncSession, model: type[SQLModel], *where: ColumnElement[bool]
) -> int:
    statement = select(func.count()).select_from(model)
    for clause in where:
        statement = statement.where(clause)
    result = await session.execute(statement)
    return int(result.scalar_one())


def _eval_questions() -> list[dict[str, Any]]:
    if not EVALSET_QUESTIONS_PATH.exists():
        return []
    loaded = yaml.safe_load(EVALSET_QUESTIONS_PATH.read_text(encoding="utf-8"))
    return list(loaded) if loaded else []


def _category_counts_label(questions: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for question in questions:
        category = question["category"]
        counts[category] = counts.get(category, 0) + 1
    # A fixed, spec-given order rather than whatever order dict insertion
    # or sorting happens to produce, so this label reads the same way
    # section 11 of the build prompt lists the six categories.
    order = ["direct", "boundary", "table", "cross_document", "out_of_scope", "injection"]
    return ", ".join(f"{counts.get(name, 0)} {name}" for name in order)


async def _workspace_names(session: AsyncSession) -> dict[UUID, str]:
    result = await session.execute(select(Workspace))
    return {workspace.id: workspace.name for workspace in result.scalars().all()}


async def _latest_eval_run_id(session: AsyncSession) -> UUID | None:
    """The run_id every row scripts/run_eval.py's most recent invocation
    wrote to EvalRun shares, or None before it has ever run. Every query
    below that reads EvalRun rows for publication filters to this value
    rather than the whole table: that script is safe to rerun (it appends
    rather than overwrites, see its own module docstring), so the table
    can hold more than one invocation's worth of rows, and only the most
    recent one is what a reader opening RESULTS.md today should see.
    """
    result = await session.execute(
        select(col(EvalRun.run_id)).order_by(col(EvalRun.run_at).desc()).limit(1)
    )
    return result.scalars().first()


async def _latest_red_team_run_id(session: AsyncSession) -> UUID | None:
    """Same reasoning as _latest_eval_run_id, against RedTeamResult's own
    run_id column instead. The two tables are written by the same script
    invocation and so share the same value in practice, but are looked up
    independently here rather than one being assumed from the other,
    since nothing in the schema itself enforces that only
    scripts/run_eval.py ever writes to either table.
    """
    result = await session.execute(
        select(col(RedTeamResult.run_id)).order_by(col(RedTeamResult.run_at).desc()).limit(1)
    )
    return result.scalars().first()


def _format_retrieval_metrics_table(runs: list[EvalRun], workspace_names: dict[UUID, str]) -> str:
    """Section 8's full comparison table, every cell EvalRun holds, not
    only a winning configuration: one row per workspace, configuration,
    and category actually scored, in the exact shape RedTeamResult and
    EvalRun's own docstrings describe. Returns a placeholder sentence
    instead of an empty table when no EvalRun rows exist yet, so a fresh
    checkout before scripts/run_eval.py has ever run renders something
    readable rather than a header over nothing.
    """
    if not runs:
        return "_No EvalRun rows yet. Run `scripts/run_eval.py` (build order step 20)._"
    lines = [
        "| Workspace | Config | Category | Recall@3 | Recall@5 | Precision@5 | MRR | N |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in sorted(
        runs, key=lambda r: (workspace_names.get(r.workspace_id, ""), r.config_label, r.category)
    ):
        name = workspace_names.get(run.workspace_id, str(run.workspace_id))
        lines.append(
            f"| {name} | {run.config_label} | {run.category} | {run.recall_at_3:.3f} | "
            f"{run.recall_at_5:.3f} | {run.precision_at_5:.3f} | {run.mrr:.3f} | {run.n_questions} |"
        )
    return "\n".join(lines)


def _format_boundary_comparison_table(runs: list[EvalRun], workspace_names: dict[UUID, str]) -> str:
    """Section 16's specific ask, "the chunking strategy comparison with
    the boundary spanning subset broken out", pulled out of the full
    retrieval table above into its own small, easy to scan view: exactly
    the category build_eval_questions.py's own boundary spanning
    questions were constructed to make naive chunking struggle with.
    """
    boundary_runs = [run for run in runs if run.category == "boundary"]
    if not boundary_runs:
        return "_No boundary category EvalRun rows yet._"
    lines = ["| Workspace | Config | Recall@3 | Recall@5 | MRR | N |", "|---|---|---|---|---|---|"]
    for run in sorted(
        boundary_runs, key=lambda r: (workspace_names.get(r.workspace_id, ""), r.config_label)
    ):
        name = workspace_names.get(run.workspace_id, str(run.workspace_id))
        lines.append(
            f"| {name} | {run.config_label} | {run.recall_at_3:.3f} | {run.recall_at_5:.3f} | "
            f"{run.mrr:.3f} | {run.n_questions} |"
        )
    return "\n".join(lines)


def _faithfulness_claim_counts(turns: list[Turn]) -> dict[str, int]:
    counts = {"entailed": 0, "contradicted": 0, "unsupported": 0}
    for turn in turns:
        for claim in turn.claims:
            label = claim.get("nli_label")
            if isinstance(label, str) and label in counts:
                counts[label] += 1
    return counts


async def _red_team_pass_counts(
    session: AsyncSession, *, run_id: UUID | None
) -> dict[str, tuple[int, int]]:
    statement = select(RedTeamResult)
    if run_id is not None:
        statement = statement.where(col(RedTeamResult.run_id) == run_id)
    result = await session.execute(statement)
    rows = list(result.scalars().all())
    counts: dict[str, tuple[int, int]] = {}
    for suite in RED_TEAM_SUITES:
        suite_rows = [row for row in rows if row.suite == suite]
        passed = sum(1 for row in suite_rows if row.passed)
        counts[suite] = (passed, len(suite_rows))
    return counts


async def _out_of_scope_failures_table(session: AsyncSession, *, run_id: UUID | None) -> str:
    """Every out_of_scope failure under naive chunking, in full, not a
    curated sample: section 18's own "no claim of zero hallucinations...
    let the rates speak" applies just as much to this category's real 0
    percent pass rate. Filtered to case ids starting "naive:" only
    (rather than showing all 16 rows) because build order step 20's real
    run found every one of these eight questions failed identically
    under structure aware chunking too; repeating each row twice would
    not add information, only length. detail already carries a
    real turn's actual answer text, written by scripts/run_eval.py at
    the moment the check failed, not reconstructed here.

    Also filtered to run_id, the same as every other red team query in
    this module: without it, a second scripts/run_eval.py invocation left
    this exact table showing every question twice, once per run, which is
    the real duplicate this project found and is precisely why the
    run_id column and this filter exist.
    """
    statement = select(RedTeamResult).where(
        col(RedTeamResult.suite) == "out_of_scope",
        col(RedTeamResult.passed).is_(False),
        col(RedTeamResult.case_id).like("naive:%"),
    )
    if run_id is not None:
        statement = statement.where(col(RedTeamResult.run_id) == run_id)
    result = await session.execute(statement.order_by(col(RedTeamResult.case_id)))
    rows = list(result.scalars().all())
    if not rows:
        return "_No failing out_of_scope cases under naive chunking in the current run._"
    lines = [
        "Every row below failed identically under structure aware chunking too "
        "(not repeated here; the full 16 row record is in the `red_team_result` table).",
        "",
        "| Question | What was wrongly treated as in scope (truncated) |",
        "|---|---|",
    ]
    for row in rows:
        question = row.case_id.split(": ", 1)[1] if ": " in row.case_id else row.case_id
        lines.append(f"| {question} | {_quoted_passage_snippet(row.detail)} |")
    return "\n".join(lines)


_QUOTED_PASSAGE_MARKER = 'verbatim. [1]\n\n"'
"""The exact boilerplate ExtractiveGenerator's own template ends with
(packages/generate/src/groundwork_generate/generate.py), right before
the actual quoted chunk text begins. Splitting on it, rather than just
truncating an out of scope failure's stored answer from character zero,
is what makes the failures table below show the passage that was wrongly
judged relevant instead of a row of identical boilerplate: "This is an
extractive answer: no language model was used..." alone is already
longer than a readable table cell."""


def _quoted_passage_snippet(detail: str | None, *, limit: int = 160) -> str:
    text = detail or ""
    marker_index = text.find(_QUOTED_PASSAGE_MARKER)
    if marker_index != -1:
        text = text[marker_index + len(_QUOTED_PASSAGE_MARKER) :]
    snippet = text.replace("\n", " ").replace("|", "/").strip()
    if len(snippet) > limit:
        snippet = snippet[:limit] + "..."
    return snippet


async def build_manifest(session: AsyncSession) -> dict[str, Any]:
    """Every key here is either the direct result of a query run inside
    this call, or read from evalset/questions.yaml, itself a committed
    file rather than a number carried over from a previous run. Nothing
    is cached between calls, so a manifest built right after seeding
    fresh data can never show yesterday's counts.

    Takes an already open session rather than opening its own, the same
    dependency injection shape index_document() and search_chunks() use
    in packages/retrieve: it keeps this function testable against
    whichever database a caller's own session is bound to (the real one,
    or an isolated test database truncated between tests) instead of
    always reaching for the process wide singleton engine.
    """
    manifest: dict[str, Any] = {}

    manifest["workspace_count"] = await _scalar_count(session, Workspace)
    manifest["document_count"] = await _scalar_count(session, Document)
    manifest["naive_chunk_count"] = await _scalar_count(
        session, Chunk, col(Chunk.strategy) == ChunkStrategy.NAIVE
    )
    manifest["structure_chunk_count"] = await _scalar_count(
        session, Chunk, col(Chunk.strategy) == ChunkStrategy.STRUCTURE
    )
    eval_run_count = await _scalar_count(session, EvalRun)
    red_team_result_count = await _scalar_count(session, RedTeamResult)

    questions = _eval_questions()
    manifest["eval_question_count"] = len(questions)
    manifest["eval_category_counts"] = _category_counts_label(questions)

    manifest["eval_run_count"] = eval_run_count
    manifest["retrieval_metrics_status"] = "computed" if eval_run_count else NOT_YET_RUN

    manifest["red_team_result_count"] = red_team_result_count
    manifest["red_team_status"] = "computed" if red_team_result_count else NOT_YET_RUN

    workspace_names = await _workspace_names(session)
    latest_eval_run_id = await _latest_eval_run_id(session)
    eval_runs_statement = select(EvalRun)
    if latest_eval_run_id is not None:
        eval_runs_statement = eval_runs_statement.where(col(EvalRun.run_id) == latest_eval_run_id)
    eval_runs = list((await session.execute(eval_runs_statement)).scalars().all())
    manifest["retrieval_metrics_table"] = _format_retrieval_metrics_table(
        eval_runs, workspace_names
    )
    manifest["boundary_comparison_table"] = _format_boundary_comparison_table(
        eval_runs, workspace_names
    )

    # Turn has no run_id of its own: unlike EvalRun and RedTeamResult, a
    # Turn row is not only ever written by scripts/run_eval.py, it is the
    # same row a real chat request through /workspaces/{id}/ask produces,
    # so giving it a mandatory eval run_id would misdescribe what the
    # column means for every non-eval turn. Counting every Turn in the
    # table stays correct exactly as long as the documented run order
    # (seed, then build questions, then run_eval) is followed, since that
    # order clears prior Turn rows before the current run writes new
    # ones; see scripts/seed_demo_workspaces.py's own cascade delete.
    turns = list((await session.execute(select(Turn))).scalars().all())
    manifest["turn_count"] = len(turns)
    faithfulness_counts = _faithfulness_claim_counts(turns)
    manifest["faithfulness_claim_count"] = sum(faithfulness_counts.values())
    manifest["faithfulness_entailed_count"] = faithfulness_counts["entailed"]
    manifest["faithfulness_contradicted_count"] = faithfulness_counts["contradicted"]
    manifest["faithfulness_unsupported_count"] = faithfulness_counts["unsupported"]

    latest_red_team_run_id = await _latest_red_team_run_id(session)
    red_team_pass_counts = await _red_team_pass_counts(session, run_id=latest_red_team_run_id)
    for suite in RED_TEAM_SUITES:
        passed, total = red_team_pass_counts[suite]
        manifest[f"{suite}_pass_count"] = passed
        manifest[f"{suite}_total_count"] = total

    manifest["out_of_scope_failures_table"] = await _out_of_scope_failures_table(
        session, run_id=latest_red_team_run_id
    )

    return manifest


def render_template(template: str, manifest: dict[str, Any]) -> str:
    """Substitutes every {{key}} placeholder with manifest[key], str()'d.
    Raises on a placeholder the manifest does not cover, so a typo in a
    template fails the render loudly instead of shipping the literal
    "{{typo}}" text into a published document.
    """

    def _substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in manifest:
            raise KeyError(
                f"template references {{{{{key}}}}}, which build_manifest() does not provide"
            )
        return str(manifest[key])

    return _PLACEHOLDER.sub(_substitute, template)
