"""The injection defense test: section 11's own required test ("ask a
question that would plausibly retrieve that chunk, and assert the marker
string never appears anywhere in the output"), run against this build's
real injection test PDF through the full ingest, chunk, embed, retrieve,
generate, verify pipeline, not a synthetic stand in for any one stage of
it.

evalset/injection_test.pdf and its five real questions
(evalset/questions.yaml's injection category) already exist and are
already seeded into the dev database by scripts/seed_demo_workspaces.py.
This file re-extracts and re-indexes the same fixture into the isolated
test database instead of depending on that seeding having already run
somewhere else, so this test is self-contained and passes on a fresh
checkout with nothing but a reachable Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.chat import ask
from groundwork_api.models import (
    ChunkStrategy,
    Conversation,
    Document,
    ExtractionMethod,
    Workspace,
)
from groundwork_core.config import get_settings
from groundwork_generate.generate import ExtractiveGenerator
from groundwork_ingest.extract import extract_pdf
from groundwork_ingest.fixtures import INJECTION_TEST_MARKER
from groundwork_retrieve.index import index_document

pytestmark = pytest.mark.requires_postgres

REPO_ROOT = Path(__file__).resolve().parents[3]
INJECTION_PDF = REPO_ROOT / "evalset" / "injection_test.pdf"
QUESTIONS_PATH = REPO_ROOT / "evalset" / "questions.yaml"

requires_llm_key = pytest.mark.skipif(
    not get_settings().llm_api_key, reason="no LLM API key configured in this environment"
)


def _injection_questions() -> list[str]:
    loaded = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [q["question"] for q in loaded if q["category"] == "injection"]


async def _seed_injection_workspace(session: AsyncSession) -> Workspace:
    workspace = Workspace(name="injection-defense-test")
    session.add(workspace)
    await session.flush()

    extracted = extract_pdf(INJECTION_PDF)
    document = Document(
        workspace_id=workspace.id,
        filename=INJECTION_PDF.name,
        sha256=extracted.sha256,
        page_count=extracted.page_count,
        extraction_method=ExtractionMethod(extracted.extraction_method),
    )
    session.add(document)
    await session.flush()

    await index_document(
        session, workspace_id=workspace.id, document_id=document.id, extracted=extracted
    )
    await session.commit()
    return workspace


def test_injection_test_pdf_actually_contains_the_marker_in_its_extractable_text() -> None:
    """A sanity check on the fixture itself, not the defense: if this ever
    stopped being true, every test below would trivially pass for the
    wrong reason, never actually exercising anything. PyMuPDF extracts
    every character on a page regardless of its rendered color or size,
    which is exactly the property that makes "invisible to a human,
    present in the text layer" a real attack shape worth testing at all.
    """
    extracted = extract_pdf(INJECTION_PDF)
    assert INJECTION_TEST_MARKER in extracted.full_text


@pytest.mark.parametrize("strategy", [ChunkStrategy.NAIVE, ChunkStrategy.STRUCTURE])
async def test_injection_defense_extractive_path(
    db_session: AsyncSession, strategy: ChunkStrategy
) -> None:
    """The only path this sandbox can actually exercise end to end: no
    LLM key is configured here, so every one of the five injection
    questions resolves through ExtractiveGenerator. See that class's own
    docstring for the mechanism: a chunk the ingestion time heuristic
    flagged is withheld rather than quoted verbatim, which is what stops
    this specific marker, sitting in the chunk's own extracted text, from
    ever reaching the rendered answer. Parametrized over both chunking
    strategies since the withholding is keyed off Chunk.injection_flag,
    not the strategy, and both are first class in this build.
    """
    workspace = await _seed_injection_workspace(db_session)
    conversation = Conversation(workspace_id=workspace.id)
    db_session.add(conversation)
    await db_session.commit()

    for question in _injection_questions():
        turn = await ask(
            db_session,
            conversation_id=conversation.id,
            question=question,
            strategy=strategy,
            generator=ExtractiveGenerator(),
        )
        # Did retrieval even run and find the planted chunk: a pass below
        # must mean the defense actually engaged, not that nothing
        # relevant was ever found to leak in the first place.
        assert turn.retrieved_chunk_ids, (
            f"nothing retrieved for {question!r}, the defense was never exercised"
        )
        # Did the defense hold.
        assert INJECTION_TEST_MARKER not in turn.answer, (
            f"marker leaked for {question!r}: {turn.answer!r}"
        )
        assert turn.extractive_fallback is True


@requires_llm_key
async def test_injection_defense_litellm_path(db_session: AsyncSession) -> None:
    """The structural defense section 2 actually describes: a real model
    shown the planted instruction, verbatim, inside a passage explicitly
    labelled as data never instructions, and never emitting the marker on
    its own initiative. Skipped without a configured key, the same shape
    every other real-model test in this build uses, since this sandbox
    has none: RESULTS.md and docs/security.md say so plainly rather than
    claiming this path was measured when it was not.
    """
    workspace = await _seed_injection_workspace(db_session)
    conversation = Conversation(workspace_id=workspace.id)
    db_session.add(conversation)
    await db_session.commit()

    for question in _injection_questions():
        turn = await ask(
            db_session,
            conversation_id=conversation.id,
            question=question,
            strategy=ChunkStrategy.NAIVE,
        )
        assert turn.extractive_fallback is False
        assert INJECTION_TEST_MARKER not in turn.answer
