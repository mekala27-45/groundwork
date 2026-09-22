"""Local-first embeddings, pgvector backed indexing per chunking strategy,
vector search, cross encoder reranking, and the retrieval evaluation
harness, all built on the same auto-probe resilience pattern the rest of
this workspace uses. See embeddings.py for the embedding backend choice,
index.py for turning chunk candidates into stored rows, search.py for
querying them back out, rerank.py for reordering a shortlist against a
query, and evaluate.py for scoring all of the above against a golden
question set.
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
from groundwork_retrieve.evaluate import (
    CONFIGURATIONS,
    EvalQuestionRecord,
    MetricResult,
    QuestionScore,
    evaluate_all_configurations,
    evaluate_configuration,
    score_question,
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
    "CONFIGURATIONS",
    "EMBEDDING_DIM",
    "CrossEncoderReranker",
    "Embedder",
    "EmbeddingBackend",
    "EvalQuestionRecord",
    "LexicalReranker",
    "LocalModelEmbedder",
    "MetricResult",
    "QuestionScore",
    "RerankBackend",
    "Reranker",
    "TfidfEmbedder",
    "evaluate_all_configurations",
    "evaluate_configuration",
    "get_embedder",
    "get_reranker",
    "index_document",
    "reset_embedder_cache",
    "reset_reranker_cache",
    "resolve_embedding_backend",
    "resolve_rerank_backend",
    "score_question",
    "search_chunks",
]
