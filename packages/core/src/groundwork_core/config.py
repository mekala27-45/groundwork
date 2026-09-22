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

    relevance_threshold: float = 0.15
    """Below this cosine similarity between a query and the single most
    relevant chunk retrieval found, groundwork_api.chat.ask treats
    retrieval as having found nothing usable and answers with the same
    not-covered refusal an empty retrieval result produces, rather than
    handing a generator (real or extractive) a chunk that is not actually
    about the question. This is what lets out of scope refusal work on
    every backend, including the zero cost extractive path, which has no
    other way to judge relevance: the alternative, an LLM deciding on its
    own that retrieved passages do not answer the question, only exists
    when a real model is configured at all.

    Measured, not guessed, against the real 45 question eval set under
    this environment's actual embedding backend (tfidf: huggingface.co is
    unreachable here, so BAAI/bge-small-en-v1.5 was never in the running).
    That measurement's honest conclusion: lexical cosine similarity does
    not cleanly separate out_of_scope questions from genuinely answerable
    ones. Direct category top-1 similarity ranged 0.1455 to 0.5113;
    out_of_scope ranged 0.1612 to 0.4451, almost the same span. A
    question like "what is data leakage, as defined in the glossary"
    scores low because its wording barely overlaps the source chunk's,
    not because it is off topic, while an out_of_scope question sharing
    generic domain vocabulary ("the guide", "recommend", "training") with
    the corpus can score higher than a real question does. This value is
    deliberately not tuned to maximize this build's own out_of_scope pass
    rate on that specific 45 question set, which would be fitting the
    threshold to the eval rather than measuring against it: it sits just
    above where the most degenerate mismatches land (near zero or
    negative similarity, several of the injection category's off topic
    phrasings included) and leaves the genuinely ambiguous middle
    unresolved. RESULTS.md and docs/security.md report the real,
    honestly unimpressive out_of_scope pass rate this produces, and name
    a real embedding model or a real LLM key, either one, as what
    actually closes this gap, since both replace lexical overlap with an
    actual semantic or generative judgment this sandbox cannot make.
    """

    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
