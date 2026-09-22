"""Request and response schemas for the HTTP boundary. Every one of these
is a StrictModel, the same base class the stored records use, but they are
deliberately their own classes rather than the SQLModel table classes
themselves: a table row can carry columns (an internal id shape, a future
column added for a reason with nothing to do with the API) that should
never silently become part of a wire contract just because FastAPI is
willing to serialize whatever it is given. Keeping the two separate means
a schema migration and an API contract change are always two decisions,
never accidentally one.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from groundwork_api.models import ChunkStrategy, ExtractionMethod
from groundwork_core.model import StrictModel


class DocumentOut(StrictModel):
    id: UUID
    filename: str
    page_count: int
    extraction_method: ExtractionMethod
    uploaded_at: datetime


class ChunkOut(StrictModel):
    id: UUID
    document_id: UUID
    workspace_id: UUID
    strategy: ChunkStrategy
    text: str
    page_start: int
    page_end: int
    section_title: str | None
    injection_flag: str | None


class WorkspaceOut(StrictModel):
    id: UUID
    name: str
    description: str | None
    created_at: datetime


class WorkspaceDetailOut(WorkspaceOut):
    documents: list[DocumentOut]
    naive_chunk_count: int
    structure_chunk_count: int


class ConversationOut(StrictModel):
    id: UUID
    workspace_id: UUID
    created_at: datetime


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    strategy: ChunkStrategy = ChunkStrategy.NAIVE
    use_reranking: bool = False


class ClaimOut(StrictModel):
    text: str
    cited_chunk_id: str | None
    nli_label: str
    score: float


class CitationVerificationOut(StrictModel):
    chunk_id: str
    verified: bool
    reason: str | None


class TurnOut(StrictModel):
    id: UUID
    conversation_id: UUID
    workspace_id: UUID
    question: str
    retrieved_chunk_ids: list[str]
    reranked: bool
    chunking_strategy: ChunkStrategy | None
    answer: str
    claims: list[ClaimOut]
    citation_verifications: list[CitationVerificationOut]
    extractive_fallback: bool
    judge_scores: dict[str, object] | None
    latency_ms: float
    cost_usd: float
    created_at: datetime


class EvalRunOut(StrictModel):
    id: UUID
    run_id: UUID
    run_at: datetime
    embedding_backend: str
    rerank_backend: str
    config_label: str
    workspace_id: UUID
    category: str
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    n_questions: int


class RedTeamResultOut(StrictModel):
    id: UUID
    run_id: UUID
    run_at: datetime
    suite: str
    case_id: str
    passed: bool
    detail: str | None


class ErrorOut(StrictModel):
    detail: str


__all__ = [
    "AskRequest",
    "ChunkOut",
    "ClaimOut",
    "CitationVerificationOut",
    "ConversationOut",
    "DocumentOut",
    "ErrorOut",
    "EvalRunOut",
    "RedTeamResultOut",
    "TurnOut",
    "WorkspaceDetailOut",
    "WorkspaceOut",
]
