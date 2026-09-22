"""Reranks a shortlist of already retrieved chunk texts against a query,
the same auto probe plus deterministic fallback shape embeddings.py uses:
a local sentence-transformers cross encoder when huggingface.co is
reachable, a zero network BM25 fallback otherwise.

The batch independence concern embeddings.py's module docstring spends a
paragraph on does not apply here. An embedding has to mean the same thing
whether it was produced during indexing (a whole document's chunks in one
batch) or during a later query (one string, alone); that is what forced
embed() to ignore whatever else shares its batch. Reranking has no such
split: every call scores one fixed candidate set against one query in one
step, there is no earlier "index time" batch its output has to stay
consistent with. So the lexical fallback below is free to compute
statistics, document frequency among the candidates chief among them, over
exactly the batch it is given. That is not a shortcut; it is what BM25 is
supposed to do.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Literal, Protocol

from groundwork_core.config import get_settings
from groundwork_core.network import can_reach_huggingface

RerankBackend = Literal["local_model", "lexical", "none"]

_TOKEN_PATTERN = re.compile(r"\w+")

BM25_K1 = 1.5
BM25_B = 0.75


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Returns one relevance score per candidate, same order as given,
        higher meaning more relevant. Never reorders candidates itself;
        the caller sorts, since the caller is the one who also needs to
        carry each score's chunk identity along with it."""
        ...


class CrossEncoderReranker:
    """Wraps sentence-transformers' cross-encoder/ms-marco-MiniLM-L-6-v2.
    Constructing this downloads and caches the model weights the first
    time; every predict() call after that runs locally."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        self._model = CrossEncoder(model_name or settings.reranker_model_name)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, candidate) for candidate in candidates]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]


class LexicalReranker:
    """Deterministic, zero network fallback: Okapi BM25 computed fresh over
    each call's own candidate set (see module docstring for why that is
    safe here and was not safe for embeddings.py's fallback). Standard
    k1=1.5, b=0.75 defaults; document frequency and average length are
    both scoped to the candidates passed to that one rerank() call, never
    cached or carried between calls.
    """

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(text)]

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        query_terms = self._tokenize(query)
        if not query_terms:
            return [0.0] * len(candidates)

        documents = [self._tokenize(candidate) for candidate in candidates]
        doc_lengths = [len(doc) for doc in documents]
        avg_doc_length = sum(doc_lengths) / len(documents) if documents else 0.0
        num_docs = len(documents)

        document_frequency: dict[str, int] = {}
        for doc in documents:
            for term in set(doc):
                document_frequency[term] = document_frequency.get(term, 0) + 1

        def idf(term: str) -> float:
            df = document_frequency.get(term, 0)
            return math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

        scores: list[float] = []
        for doc, doc_length in zip(documents, doc_lengths, strict=True):
            term_counts: dict[str, int] = {}
            for term in doc:
                term_counts[term] = term_counts.get(term, 0) + 1

            length_norm = (
                1.0 - BM25_B + BM25_B * (doc_length / avg_doc_length if avg_doc_length else 0.0)
            )
            score = 0.0
            for term in query_terms:
                term_freq = term_counts.get(term, 0)
                if term_freq == 0:
                    continue
                numerator = term_freq * (BM25_K1 + 1.0)
                denominator = term_freq + BM25_K1 * length_norm
                score += idf(term) * (numerator / denominator)
            scores.append(score)
        return scores


def resolve_rerank_backend() -> RerankBackend:
    choice = get_settings().rerank_backend
    if choice != "auto":
        return choice
    return "local_model" if can_reach_huggingface() else "lexical"


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    """The process wide reranker, constructed once. None means the
    resolved backend is "none": reranking is switched off and a caller
    should use the vector search order as is, not a signal to substitute
    some other reranker in its place."""
    backend = resolve_rerank_backend()
    if backend == "none":
        return None
    return CrossEncoderReranker() if backend == "local_model" else LexicalReranker()


def reset_reranker_cache() -> None:
    """Test-only: clear the cached reranker instance."""
    get_reranker.cache_clear()
