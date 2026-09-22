"""The multi tenant data model. Every table that can be scoped to a
workspace carries workspace_id directly, including Chunk (denormalized off
Document) and Turn (denormalized off Conversation), specifically so every
retrieval query, chunk fetch and eval question lookup can filter on
workspace_id without a join. test_workspace_isolation in
packages/api/tests/test_isolation.py depends on that: it seeds two
workspaces and asserts no code path can read across the boundary.

Embeddings are stored at a fixed 384 dimensions regardless of which
backend produced them (BAAI/bge-small-en-v1.5's native output size), so the
tfidf fallback backend projects into the same 384 dimensions rather than
needing its own column or its own index. embedding_backend on Chunk records
which one actually produced the stored vector, exactly as Turn.reranked and
Document.extraction_method record which code path ran.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column
from sqlalchemy import Enum as SAEnum
from sqlmodel import ARRAY, Field, SQLModel
from sqlmodel import String as SQLString

from groundwork_core.ids import new_id

EMBEDDING_DIM = 384


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ExtractionMethod(enum.StrEnum):
    TEXT = "text"
    OCR = "ocr"
    MIXED = "mixed"


class ChunkStrategy(enum.StrEnum):
    NAIVE = "naive"
    STRUCTURE = "structure"


class EvalCategory(enum.StrEnum):
    DIRECT = "direct"
    BOUNDARY = "boundary"
    TABLE = "table"
    CROSS_DOCUMENT = "cross_document"
    OUT_OF_SCOPE = "out_of_scope"
    INJECTION = "injection"


class NliLabel(enum.StrEnum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class Workspace(SQLModel, table=True):
    __tablename__ = "workspace"

    id: UUID = Field(default_factory=new_id, primary_key=True)
    name: str
    description: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Document(SQLModel, table=True):
    __tablename__ = "document"

    id: UUID = Field(default_factory=new_id, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    filename: str
    sha256: str = Field(index=True)
    page_count: int
    extraction_method: ExtractionMethod = Field(
        sa_column=Column(SAEnum(ExtractionMethod, name="extraction_method"), nullable=False)
    )
    uploaded_at: datetime = Field(default_factory=_utcnow)


class Chunk(SQLModel, table=True):
    __tablename__ = "chunk"

    id: UUID = Field(default_factory=new_id, primary_key=True)
    document_id: UUID = Field(foreign_key="document.id", index=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    strategy: ChunkStrategy = Field(
        sa_column=Column(SAEnum(ChunkStrategy, name="chunk_strategy"), nullable=False, index=True)
    )
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    section_title: str | None = None
    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector(EMBEDDING_DIM)))
    embedding_backend: str | None = Field(default=None)
    """Which embedding backend actually produced the stored vector:
    'local_model' (BAAI/bge-small-en-v1.5) or 'tfidf' (the deterministic
    zero-network fallback). Recorded per row so a mixed-backend index run
    during development is still honestly labelled rather than silently
    blended."""
    injection_flag: str | None = Field(default=None)
    """Set by the ingestion-time heuristic in groundwork_ingest.security
    when a chunk contains suspicious near-invisible styling or embedded
    instruction language. Surfaced in the eval dashboard regardless of
    whether any later defense held, per the hard constraint that a real
    product raises this to a human either way."""


class EvalQuestion(SQLModel, table=True):
    __tablename__ = "eval_question"

    id: UUID = Field(default_factory=new_id, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    question: str
    expected_chunk_ids: list[str] = Field(sa_column=Column(ARRAY(SQLString())))
    category: EvalCategory = Field(
        sa_column=Column(SAEnum(EvalCategory, name="eval_category"), nullable=False, index=True)
    )
    notes: str | None = None


class Conversation(SQLModel, table=True):
    __tablename__ = "conversation"

    id: UUID = Field(default_factory=new_id, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    created_at: datetime = Field(default_factory=_utcnow)


class Turn(SQLModel, table=True):
    __tablename__ = "turn"

    id: UUID = Field(default_factory=new_id, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversation.id", index=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    """Denormalized from conversation.workspace_id, same reasoning as
    Chunk.workspace_id: every isolation-sensitive query filters on this
    column directly rather than joining through conversation first."""
    question: str
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, sa_column=Column(ARRAY(SQLString()))
    )
    reranked: bool = False
    chunking_strategy: ChunkStrategy | None = Field(
        default=None,
        sa_column=Column(SAEnum(ChunkStrategy, name="turn_chunk_strategy"), nullable=True),
    )
    answer: str
    claims: list[dict[str, object]] = Field(default_factory=list, sa_column=Column(JSON))
    """Each entry: {"text": str, "cited_chunk_id": str, "nli_label": str,
    "score": float}, written by packages/verify."""
    citation_verifications: list[dict[str, object]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    extractive_fallback: bool = False
    judge_scores: dict[str, object] | None = Field(default=None, sa_column=Column(JSON))
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)


class EvalRun(SQLModel, table=True):
    """One execution of the full retrieval evaluation harness (section 8),
    stored so RESULTS.md always renders from the most recent real run
    rather than from a number that was computed once and then pasted."""

    __tablename__ = "eval_run"

    id: UUID = Field(default_factory=new_id, primary_key=True)
    run_id: UUID = Field(index=True)
    """One shared value across every EvalRun and RedTeamResult row a
    single scripts/run_eval.py invocation writes (generated once, at the
    top of that script's main(), never per row), so a later query can
    ask for exactly one real run's numbers rather than an
    ever-accumulating mix of however many times the script has ever been
    run. Added after build order step 20's first real run: nothing
    before that had ever written a second EvalRun row against the same
    demo data, so nothing had ever needed to tell two runs apart, until
    scripts/seed_demo_workspaces.py's own workspace reset bug (see that
    script's docstring) forced a second real run and the rows from both
    landed in the same table with no way to separate them."""
    run_at: datetime = Field(default_factory=_utcnow)
    embedding_backend: str
    rerank_backend: str
    config_label: str
    """e.g. 'naive+no_rerank', 'structure+rerank'."""
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    category: str
    """The eval question category this row scores, or 'all'."""
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    n_questions: int


class RedTeamResult(SQLModel, table=True):
    """One row per red-team / out-of-scope / isolation probe execution,
    the section 11 scorecard's backing store."""

    __tablename__ = "red_team_result"

    id: UUID = Field(default_factory=new_id, primary_key=True)
    run_id: UUID = Field(index=True)
    """Shared with EvalRun.run_id, same value, same reasoning: see that
    field's own docstring."""
    run_at: datetime = Field(default_factory=_utcnow)
    suite: str
    """'injection', 'out_of_scope', or 'workspace_isolation'."""
    case_id: str
    passed: bool
    detail: str | None = None
