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

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnElement
from sqlmodel import SQLModel, col

from groundwork_api.models import Chunk, ChunkStrategy, Document, EvalRun, RedTeamResult, Workspace

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
