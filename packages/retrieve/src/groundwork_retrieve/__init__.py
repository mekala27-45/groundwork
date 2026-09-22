"""Local-first embeddings, pgvector backed indexing per chunking strategy,
vector search, and cross encoder reranking, all built on the same
auto-probe resilience pattern the rest of this workspace uses. See
embeddings.py for the embedding backend choice, index.py for turning
chunk candidates into stored rows, search.py for querying them back out,
and rerank.py for reordering a shortlist against a query.
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
from groundwork_retrieve.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    RerankBackend,
    Reranker,
    get_reranker,
    reset_reranker_cache,
    resolve_rerank_backend,
)
from groundwork_retrieve.search import search_chunks

__all__ = [
    "EMBEDDING_DIM",
    "CrossEncoderReranker",
    "Embedder",
    "EmbeddingBackend",
    "LexicalReranker",
    "LocalModelEmbedder",
    "RerankBackend",
    "Reranker",
    "TfidfEmbedder",
    "get_embedder",
    "get_reranker",
    "index_document",
    "reset_embedder_cache",
    "reset_reranker_cache",
    "resolve_embedding_backend",
    "resolve_rerank_backend",
    "search_chunks",
]
