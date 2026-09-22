from __future__ import annotations

from typing import Literal

from groundwork_core import StrictModel
from groundwork_ingest.models import ExtractedTable

AtomKind = Literal["word", "table"]


class Atom(StrictModel):
    """The smallest unit either chunking strategy is allowed to split at.
    A table is never divided below the level of a whole Atom, which is
    what keeps "table rows are never split by the overlap logic" true by
    construction rather than by a check bolted on afterward."""

    kind: AtomKind
    text: str
    page_number: int
    is_section_start: bool = False
    table: ExtractedTable | None = None


class ChunkCandidate(StrictModel):
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    section_title: str | None = None
    strategy: Literal["naive", "structure"]
    tokenizer_backend: Literal["tiktoken", "approximate"]
    """Which backend counted tokens for this chunk: the real cl100k_base
    encoding or the deterministic offline approximation. Recorded per
    chunk, exactly as Chunk.embedding_backend records which embedding
    backend produced a stored vector, so a mixed-backend run stays
    honestly labelled rather than silently blended."""
