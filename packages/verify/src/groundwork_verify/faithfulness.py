"""Faithfulness scoring: for each claim, checks whether its cited chunk
entails it, the independent method section 10 requires ("An independent,
locally run NLI model. A judge model may score a separate, clearly
labeled quality dimension only"). This is the same conflict of interest
Day 4 flagged for its safety gate, applied a second time: the generation
model grading its own output would share the exact blind spots that
produced a hallucination in the first place, so this never calls back
into groundwork_generate or LiteLLM at all.

Two FaithfulnessScorer implementations behind one interface, the same
auto-probe resilience shape embeddings.py and rerank.py use for their own
two backends: CrossEncoderFaithfulnessScorer, a local NLI cross encoder
model, and LexicalFaithfulnessScorer, a deterministic, zero network
fallback for when huggingface.co is unreachable, exactly as this
sandbox's own network policy has been throughout this build.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal, Protocol

from groundwork_api.models import Chunk, NliLabel
from groundwork_core.config import get_settings
from groundwork_core.model import StrictModel
from groundwork_core.network import can_reach_huggingface
from groundwork_verify.split import Claim, split_claims

FaithfulnessBackend = Literal["local_model", "lexical"]

ENTAILMENT_OVERLAP_THRESHOLD = 0.6
"""How much of a claim's vocabulary must be traceable to its cited chunk
before LexicalFaithfulnessScorer calls it supported at all (entailed or,
if polarity flips, contradicted) rather than unsupported. Chosen so a
claim built mostly from the chunk's own words, plus ordinary connecting
language, clears it, while a claim about something the chunk never
mentions does not."""

_TOKEN_PATTERN = re.compile(r"\w+")
_NEGATION_PATTERNS = (" not ", "n't", " never ", " no longer ", " no ", " cannot ", " without ")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _has_negation(text: str) -> bool:
    padded = f" {text.lower()} "
    return any(pattern in padded for pattern in _NEGATION_PATTERNS)


class ClaimVerification(StrictModel):
    claim: Claim
    label: NliLabel
    score: float


class FaithfulnessScorer(Protocol):
    def score(self, claim: str, chunk_text: str) -> tuple[NliLabel, float]:
        """Scores whether chunk_text entails claim: does the chunk, taken
        as true, support the claim. Returns the label plus a confidence
        for that label, never a fixed placeholder score."""
        ...


class CrossEncoderFaithfulnessScorer:
    """Wraps sentence-transformers' cross-encoder/nli-deberta-v3-base (or
    an equivalent open weight NLI cross encoder named by
    Settings.nli_model_name). Constructing this downloads and caches the
    model weights the first time; every predict() call after that runs
    locally, with no call back to whichever model produced the answer
    being checked.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        self._model = CrossEncoder(model_name or settings.nli_model_name)

    def score(self, claim: str, chunk_text: str) -> tuple[NliLabel, float]:
        # This model family's label order (its own model card's id2label):
        # 0 contradiction, 1 entailment, 2 neutral. "Does the chunk entail
        # the claim" is the premise, hypothesis direction section 10
        # specifies, so the chunk is the pair's first element and the
        # claim its second, matching how these models are trained.
        scores = self._model.predict([(chunk_text, claim)], apply_softmax=True)[0]
        labels = [NliLabel.CONTRADICTED, NliLabel.ENTAILED, NliLabel.UNSUPPORTED]
        best_index = int(scores.argmax())
        return labels[best_index], float(scores[best_index])


class LexicalFaithfulnessScorer:
    """Deterministic, zero network fallback: a real, if less
    sophisticated, substitute rather than a stub, the same honesty
    LexicalReranker's own docstring asks for about BM25 standing in for a
    cross encoder. Word overlap between a claim and its cited chunk
    stands in for entailment, since most of a genuinely faithful claim's
    words should be traceable to the chunk it is grounded in. A negation
    mismatch on an otherwise high overlap pair stands in for
    contradiction, since restating the same words with the opposite
    polarity is the single most common way a generated claim flips a
    source's meaning. Neither is a real semantic entailment model, and
    both are named accordingly rather than dressed up as equivalent to
    the real thing.
    """

    def score(self, claim: str, chunk_text: str) -> tuple[NliLabel, float]:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            return NliLabel.UNSUPPORTED, 0.0
        chunk_tokens = _tokenize(chunk_text)
        overlap_ratio = len(claim_tokens & chunk_tokens) / len(claim_tokens)

        if overlap_ratio < ENTAILMENT_OVERLAP_THRESHOLD:
            return NliLabel.UNSUPPORTED, overlap_ratio
        if _has_negation(claim) != _has_negation(chunk_text):
            return NliLabel.CONTRADICTED, overlap_ratio
        return NliLabel.ENTAILED, overlap_ratio


def resolve_faithfulness_backend() -> FaithfulnessBackend:
    choice = get_settings().faithfulness_backend
    if choice != "auto":
        return choice
    return "local_model" if can_reach_huggingface() else "lexical"


@lru_cache(maxsize=1)
def get_faithfulness_scorer() -> FaithfulnessScorer:
    """The process wide scorer, constructed once, the same cached
    singleton shape get_embedder() and get_reranker() use."""
    return (
        CrossEncoderFaithfulnessScorer()
        if resolve_faithfulness_backend() == "local_model"
        else LexicalFaithfulnessScorer()
    )


def reset_faithfulness_scorer_cache() -> None:
    """Test-only: clear the cached scorer instance."""
    get_faithfulness_scorer.cache_clear()


def check_faithfulness(
    answer: str,
    *,
    extractive_fallback: bool,
    chunks: list[Chunk],
    scorer: FaithfulnessScorer | None = None,
) -> list[ClaimVerification]:
    """The one entry point most callers use: splits answer into claims
    against chunks in the same order generate_answer() offered them
    (chunks[i] is what a "[i+1]" marker refers to, matching
    GenerationResult.cited_chunk_ids' own ordering) and scores each one.

    extractive_fallback answers skip the scorer entirely. Their one claim
    is chunks[0].text quoted verbatim, by construction
    (groundwork_generate.generate.ExtractiveGenerator never paraphrases),
    so it is faithful by construction too, not something a probabilistic
    entailment check needs to reconfirm. Text-parsing the rendered
    answer's own framing ("This is an extractive answer: no language
    model was used...") would also risk scoring that framing as an
    unsupported claim about the document, when it is a claim about how
    the system is presenting its answer, not about the document at all.

    chunks[0].injection_flag is not None is the one case where "quoted
    verbatim by construction" stops being true: ExtractiveGenerator
    withholds a flagged chunk's text instead of quoting it (see that
    class's own docstring), so there is no claim about the document to
    score at all, the same empty result an empty chunks list already
    returns, not a claim built from text the rendered answer never
    actually contains.
    """
    if extractive_fallback:
        if not chunks or chunks[0].injection_flag is not None:
            return []
        top = chunks[0]
        return [
            ClaimVerification(
                claim=Claim(text=top.text, cited_chunk_id=top.id),
                label=NliLabel.ENTAILED,
                score=1.0,
            )
        ]

    chunk_lookup = {chunk.id: chunk.text for chunk in chunks}
    claims = split_claims(answer, [chunk.id for chunk in chunks])
    scorer = scorer or get_faithfulness_scorer()

    results: list[ClaimVerification] = []
    for claim in claims:
        chunk_text = (
            chunk_lookup.get(claim.cited_chunk_id) if claim.cited_chunk_id is not None else None
        )
        if chunk_text is None:
            results.append(ClaimVerification(claim=claim, label=NliLabel.UNSUPPORTED, score=0.0))
            continue
        label, score = scorer.score(claim.text, chunk_text)
        results.append(ClaimVerification(claim=claim, label=label, score=score))
    return results
