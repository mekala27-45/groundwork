"""Chunk and query embeddings, local sentence-transformers by default,
wired to the same "auto: probe once, cache, fall back if unreachable"
pattern documented in groundwork_core.config. Loading BAAI/bge-small-en-v1.5
for the first time is itself a huggingface.co fetch (sentence-transformers
downloads and caches the model weights), so "local" here means "runs
without further network calls once loaded," not "never touches the
network." This sandbox's own egress policy rejects that fetch exactly like
it rejects tiktoken's, which is what the fallback below actually runs
against in this build environment: see
groundwork_core.network.can_reach_huggingface.

The fallback is deliberately not full tfidf despite the Settings field's
name. A classic tfidf's idf term is a corpus statistic, fit over however
many texts happen to be in one embed() call; indexing a whole document's
chunks in one batch and embedding a single query in another would then
score idf over two differently sized "corpora", quietly changing the
vector space a query lands in relative to the chunks it is compared
against. Dropping idf and keeping hashed, signed, L2 normalized term
frequency (HashingVectorizer's own trick, no fitting required) makes
embed() a pure function of its input text, independent of whatever else
happens to be batched with it, which matters more here than a sharper but
batch-dependent baseline would.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Protocol

from sklearn.feature_extraction.text import HashingVectorizer

from groundwork_core.config import get_settings
from groundwork_core.network import can_reach_huggingface

EMBEDDING_DIM = 384

EmbeddingBackend = Literal["local_model", "tfidf"]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalModelEmbedder:
    """Wraps sentence-transformers' BAAI/bge-small-en-v1.5. Constructing
    this downloads and caches the model weights the first time; every
    encode() call after that runs locally, CPU friendly by design."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self._model = SentenceTransformer(model_name or settings.embedding_model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]


class TfidfEmbedder:
    """Deterministic, zero-network fallback: the hashing trick (Weinberger
    et al.), signed to reduce collision bias, L2 normalized so cosine
    distance behaves the same way it does against the real model's output.
    Stateless by construction, same as HashingVectorizer itself: nothing
    here is fit to a corpus, so embed() never depends on call history.
    """

    def __init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=EMBEDDING_DIM,
            alternate_sign=True,
            norm="l2",
            analyzer="word",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().tolist()  # type: ignore[no-any-return]


def resolve_embedding_backend() -> EmbeddingBackend:
    choice = get_settings().embedding_backend
    if choice != "auto":
        return choice
    return "local_model" if can_reach_huggingface() else "tfidf"


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """The process-wide embedder, constructed once. Loading the real model
    is expensive enough (and the fallback's whole point is to be cheap)
    that re-resolving the backend on every call, the way
    groundwork_chunk.tokenize.get_tokenizer_backend does, would be wrong
    here: this caches the constructed instance, not just the choice."""
    return LocalModelEmbedder() if resolve_embedding_backend() == "local_model" else TfidfEmbedder()


def reset_embedder_cache() -> None:
    """Test-only: clear the cached embedder instance."""
    get_embedder.cache_clear()
