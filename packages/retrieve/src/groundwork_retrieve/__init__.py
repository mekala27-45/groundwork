"""Local-first embeddings, pgvector backed indexing per chunking strategy,
and vector search, all built on the same auto-probe resilience pattern the
rest of this workspace uses. See embeddings.py for the backend choice,
index.py for turning chunk candidates into stored rows, and search.py for
querying them back out.
"""

from groundwork_retrieve.embeddings import (
    EMBEDDING_DIM,
    Embedder,
    EmbeddingBackend,
    LocalModelEmbedder,
    TfidfEmbedder,
    get_embedder,
    reset_embedder_cache,
    resolve_embedding_backend,
)
from groundwork_retrieve.index import index_document
from groundwork_retrieve.search import search_chunks

__all__ = [
    "EMBEDDING_DIM",
    "Embedder",
    "EmbeddingBackend",
    "LocalModelEmbedder",
    "TfidfEmbedder",
    "get_embedder",
    "index_document",
    "reset_embedder_cache",
    "resolve_embedding_backend",
    "search_chunks",
]
