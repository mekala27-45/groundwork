"""Citation span verification: deterministic, no model call, a fact
checked against the database rather than a judgment. Section 10: "For
every cited chunk id, confirm it exists, belongs to the same workspace as
the conversation, and that its stored text is what was actually passed to
the model at generation time (a hash comparison, not a string search, so
a sha256 mismatch catches a citation whose underlying chunk changed after
the fact)."

This expects to run synchronously, immediately after generation, given
the exact Chunk objects a caller already retrieved and handed to
groundwork_generate.generate.generate_answer(), not as a later, out of
band re-check of some historical turn: Turn only ever persists chunk ids
(retrieved_chunk_ids), never a text snapshot, so "what was actually
passed to the model" has to be read from the caller's own in-memory
copy, taken at the moment it was true, while it is still in scope.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.models import Chunk
from groundwork_core.model import StrictModel
from groundwork_core.redaction import content_hash


class CitationVerification(StrictModel):
    chunk_id: UUID
    verified: bool
    reason: str | None = None
    """Set when verified is False: "chunk_not_found", "workspace_mismatch",
    or "text_hash_mismatch"."""


async def verify_citations(
    session: AsyncSession, *, workspace_id: UUID, cited_chunks: list[Chunk]
) -> list[CitationVerification]:
    """cited_chunks is the exact list of Chunk objects offered to the
    generator for this turn (GenerationResult.cited_chunk_ids' order,
    already resolved to full rows by the caller). For each one, this
    re-fetches that same chunk id fresh from session and confirms three
    things: the row still exists, it still belongs to workspace_id (the
    conversation's own workspace, the same boundary
    test_workspace_isolation checks elsewhere), and its current text
    hashes identically to the copy already in hand. That third check is a
    sha256 comparison, never a string search, so a chunk edited or
    re-indexed out from under a citation is caught even if its new text
    happens to still contain a matching phrase somewhere.
    """
    results: list[CitationVerification] = []
    for cited in cited_chunks:
        current = await session.get(Chunk, cited.id)
        if current is None:
            results.append(
                CitationVerification(chunk_id=cited.id, verified=False, reason="chunk_not_found")
            )
            continue
        if current.workspace_id != workspace_id:
            results.append(
                CitationVerification(chunk_id=cited.id, verified=False, reason="workspace_mismatch")
            )
            continue
        if content_hash(current.text) != content_hash(cited.text):
            results.append(
                CitationVerification(chunk_id=cited.id, verified=False, reason="text_hash_mismatch")
            )
            continue
        results.append(CitationVerification(chunk_id=cited.id, verified=True))
    return results
