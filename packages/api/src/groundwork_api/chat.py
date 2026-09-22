"""Chat orchestration: the one path that turns a question, asked against a
conversation, into a persisted Turn. Ties together retrieval
(groundwork_retrieve.search_chunks, optionally reranked), generation
(groundwork_generate.generate_answer, with its own extractive fallback),
independent verification (groundwork_verify.check_faithfulness and
verify_citations), and secondary quality scoring
(groundwork_verify.score_reply_quality, never the faithfulness gate) into
the single call every path that produces a real answer goes through:
packages/api/tests/test_isolation.py exercises this exact function to
prove the workspace boundary holds end to end, not only at the retrieval
query in isolation, and the /chat and /trace web pages (build order steps
22 and 23) call it directly.

workspace_id is deliberately never a parameter here. It is derived once,
by loading the Conversation row and reading conversation.workspace_id,
because that is how a real chat request actually arrives: a caller has a
conversation, not a workspace id it types in separately on every question.
Accepting workspace_id as its own argument would let a caller's own bug,
or a hostile request body, name one workspace's conversation while asking
retrieval and citation verification to scope to a different workspace_id
entirely, exactly the class of bug test_workspace_isolation exists to rule
out. Deriving it structurally closes that path rather than trusting every
caller to pass matching values.

The relevance gate. Section 11's out of scope category has only one
correct answer, a stated refusal, and section 9 places that judgement on
the generation model: "say plainly that the document does not cover it
when the retrieved chunks do not support an answer." That works when a
real LLM is configured. It does not work for the zero cost extractive
path, which has no way to judge relevance at all, it simply quotes
whatever search_chunks ranked first, however unrelated. Without a
relevance check, this build's own no-LLM-key default configuration could
never pass the out of scope category. So this function checks the top
ranked chunk's cosine similarity to the query itself, before handing
anything to a generator, using the same vectors retrieval already
computed: below Settings.relevance_threshold, retrieval is treated as
having found nothing usable, and generate_answer is called with an empty
chunk list, the same path an empty search result already takes. This is a
general property of the chat path, not a special case keyed to a
category label the running system never sees: any question the corpus
does not meaningfully address is refused the same way, which is
incidentally exactly what the out of scope category measures.
"""

from __future__ import annotations

import math
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.models import Chunk, Conversation, Turn
from groundwork_api.models import ChunkStrategy as DbChunkStrategy
from groundwork_core.config import get_settings
from groundwork_generate.generate import Generator, generate_answer
from groundwork_retrieve.embeddings import Embedder, get_embedder
from groundwork_retrieve.rerank import Reranker, get_reranker
from groundwork_retrieve.search import search_chunks
from groundwork_verify.citation import CitationVerification, verify_citations
from groundwork_verify.faithfulness import (
    ClaimVerification,
    FaithfulnessScorer,
    check_faithfulness,
)
from groundwork_verify.quality import Judge, QualityScore, score_reply_quality


class ConversationNotFoundError(ValueError):
    def __init__(self, conversation_id: UUID) -> None:
        super().__init__(f"conversation {conversation_id} not found")
        self.conversation_id = conversation_id


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _retrieve(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    strategy: DbChunkStrategy,
    question: str,
    use_reranking: bool,
    top_k: int,
    candidate_k: int,
    embedder: Embedder,
    reranker: Reranker | None,
) -> list[Chunk]:
    """Vector search, then an optional rerank stage that widens the
    candidate set and reorders it, mirroring
    groundwork_retrieve.evaluate.score_question's own composition of the
    same two stages (see that module's docstring for why the two are not
    shared code: the eval harness holds candidate_k and final_k fixed for
    comparability across a whole run, while a live chat request may
    reasonably want its own per-request values, and the two call sites are
    small and independent enough that a shared abstraction would serve
    only these two very different callers).
    """
    if not use_reranking:
        return await search_chunks(
            session,
            workspace_id=workspace_id,
            strategy=strategy,
            query=question,
            top_k=top_k,
            embedder=embedder,
        )

    candidates = await search_chunks(
        session,
        workspace_id=workspace_id,
        strategy=strategy,
        query=question,
        top_k=candidate_k,
        embedder=embedder,
    )
    active_reranker = reranker or get_reranker()
    if active_reranker is None or not candidates:
        return candidates[:top_k]

    scores = active_reranker.rerank(question, [chunk.text for chunk in candidates])
    ranked = [
        chunk
        for chunk, _ in sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
    ]
    return ranked[:top_k]


def _serialize_claims(verifications: list[ClaimVerification]) -> list[dict[str, object]]:
    return [
        {
            "text": verification.claim.text,
            "cited_chunk_id": (
                str(verification.claim.cited_chunk_id)
                if verification.claim.cited_chunk_id is not None
                else None
            ),
            "nli_label": verification.label.value,
            "score": verification.score,
        }
        for verification in verifications
    ]


def _serialize_citations(verifications: list[CitationVerification]) -> list[dict[str, object]]:
    return [
        {"chunk_id": str(v.chunk_id), "verified": v.verified, "reason": v.reason}
        for v in verifications
    ]


def _serialize_quality_score(score: QualityScore | None) -> dict[str, object] | None:
    return score.model_dump() if score is not None else None


async def ask(
    session: AsyncSession,
    *,
    conversation_id: UUID,
    question: str,
    strategy: DbChunkStrategy,
    use_reranking: bool = False,
    top_k: int | None = None,
    candidate_k: int | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
    scorer: FaithfulnessScorer | None = None,
    judge: Judge | None = None,
) -> Turn:
    """Retrieves, generates, verifies, scores, and persists one Turn.

    Every retrieval and citation check this function performs is scoped to
    conversation.workspace_id, looked up fresh from the database rather
    than accepted as an argument (see module docstring). Raises
    ConversationNotFoundError if conversation_id does not name a real,
    persisted Conversation, rather than silently scoping to nothing: a
    caller passing a stale or cross-workspace conversation id is exactly
    the kind of mistake this function should surface, not paper over.

    Turn.retrieved_chunk_ids always records what retrieval actually found,
    even on a turn the relevance gate goes on to refuse, so the /trace
    view can show a reader what was retrieved and why it was judged not
    relevant enough, rather than hiding it behind an empty list that looks
    identical to a corpus with nothing in it at all.

    Turn.judge_scores is section 10's secondary quality metric, written
    from groundwork_verify.quality.score_reply_quality and kept entirely
    separate from Turn.claims and Turn.citation_verifications, the
    faithfulness gate's own columns: a low quality score never affects
    faithfulness, a high one never excuses an unfaithful claim, and
    nothing here computes a pass or fail from it. None, not a synthetic
    number, when no LLM key is configured or the shared spend ledger is
    already exhausted, which is what this sandbox actually returns on
    every turn today.
    """
    settings = get_settings()
    resolved_top_k = top_k if top_k is not None else settings.final_k
    resolved_candidate_k = candidate_k if candidate_k is not None else settings.rerank_candidate_k

    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise ConversationNotFoundError(conversation_id)
    workspace_id = conversation.workspace_id

    active_embedder = embedder or get_embedder()
    retrieved = await _retrieve(
        session,
        workspace_id=workspace_id,
        strategy=strategy,
        question=question,
        use_reranking=use_reranking,
        top_k=resolved_top_k,
        candidate_k=resolved_candidate_k,
        embedder=active_embedder,
        reranker=reranker,
    )

    generation_chunks = retrieved
    if retrieved:
        [query_vector] = active_embedder.embed([question])
        top_chunk = retrieved[0]
        if (
            top_chunk.embedding is None
            or _cosine_similarity(query_vector, top_chunk.embedding) < settings.relevance_threshold
        ):
            generation_chunks = []

    generation = await generate_answer(question, generation_chunks, generator=generator)

    faithfulness = check_faithfulness(
        generation.answer,
        extractive_fallback=generation.extractive_fallback,
        chunks=generation_chunks,
        scorer=scorer,
    )

    chunk_by_id = {chunk.id: chunk for chunk in generation_chunks}
    cited_chunks = [
        chunk_by_id[chunk_id] for chunk_id in generation.cited_chunk_ids if chunk_id in chunk_by_id
    ]
    citation_verifications = await verify_citations(
        session, workspace_id=workspace_id, cited_chunks=cited_chunks
    )

    quality_score = await score_reply_quality(question, generation.answer, judge=judge)

    turn = Turn(
        conversation_id=conversation_id,
        workspace_id=workspace_id,
        question=question,
        retrieved_chunk_ids=[str(chunk.id) for chunk in retrieved],
        reranked=use_reranking,
        chunking_strategy=strategy,
        answer=generation.answer,
        claims=_serialize_claims(faithfulness),
        citation_verifications=_serialize_citations(citation_verifications),
        extractive_fallback=generation.extractive_fallback,
        judge_scores=_serialize_quality_score(quality_score),
        latency_ms=generation.latency_ms,
        cost_usd=generation.cost_usd,
    )
    session.add(turn)
    await session.flush()
    return turn
