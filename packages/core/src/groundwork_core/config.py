"""Settings validated at import time, so a missing variable fails fast and
loudly rather than surfacing as an obscure error on the first request that
needs it.

Every backend choice below defaults to "auto", which means: probe the real
dependency once, cache the result for the process lifetime, and fall back
to the deterministic local implementation if the real one is unreachable.
This is the same shape as day 1's local-vs-docker sandbox backend and day
4's Meta-credentials-vs-simulator backend, applied here to embeddings,
reranking, faithfulness scoring, and chunk token counting
(groundwork_chunk.tokenize: tiktoken's cl100k_base ranks are themselves a
network fetch on first use, not a bundled file). Setting a variable
explicitly to "local_model", "lexical" or "tiktoken" / "approximate"
overrides the probe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingBackendChoice = Literal["auto", "local_model", "tfidf"]
RerankBackendChoice = Literal["auto", "local_model", "lexical", "none"]
FaithfulnessBackendChoice = Literal["auto", "local_model", "lexical"]
TokenizerBackendChoice = Literal["auto", "tiktoken", "approximate"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GROUNDWORK_", extra="ignore")

    database_url: str = Field(
        default="postgresql+psycopg://postgres:groundwork_dev_password@localhost:5432/groundwork_dev",
        description="Postgres connection string. Neon in production, local Postgres 16 + "
        "pgvector 0.6.0 in this build environment. Same engine, same extension either way.",
    )

    embedding_backend: EmbeddingBackendChoice = "auto"
    rerank_backend: RerankBackendChoice = "auto"
    faithfulness_backend: FaithfulnessBackendChoice = "auto"
    tokenizer_backend: TokenizerBackendChoice = "auto"

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"

    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    judge_model: str | None = None

    spend_ceiling_usd: float = 2.00

    default_top_k: int = 8
    rerank_candidate_k: int = 24
    final_k: int = 5

    ocr_density_threshold: float = 0.0001
    """Below this fraction of extractable-characters-per-page-area, a page
    is treated as scanned and routed to the OCR fallback."""

    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
