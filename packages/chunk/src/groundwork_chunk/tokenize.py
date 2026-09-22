"""Token counting for chunk budgeting, in one place so every strategy
counts the same way. cl100k_base rather than a model-specific encoding:
this project's generation backend is swappable (packages/generate), so the
chunk token budget should not be pinned to one vendor's tokenizer either.

tiktoken does not bundle cl100k_base's BPE ranks with the package; loading
them fetches a small file from openaipublic.blob.core.windows.net on first
use. That fetch is unreachable from behind some egress policies, this
build's own sandbox included, so token counting gets the same "auto"
resilience treatment groundwork_core.config already gives embeddings,
reranking and faithfulness scoring: try the real tokenizer once, remember
whether it loaded, and fall back to a deterministic offline approximation
when it did not. GROUNDWORK_TOKENIZER_BACKEND can force either side
explicitly, the same way the other backend settings do; forcing "tiktoken"
somewhere the fetch genuinely cannot succeed is a loud RuntimeError rather
than a silent fallback, because forcing a backend is the caller asserting
it works, not asking to be rescued if it does not.

Unlike the Hugging Face probe in groundwork_core.network, this checks by
attempting the real load itself rather than a separate HEAD request to a
URL that is tiktoken's own internal implementation detail, not a
documented contract worth duplicating here.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import tiktoken

from groundwork_core.config import get_settings

TokenizerBackend = Literal["tiktoken", "approximate"]

_APPROX_CHARS_PER_TOKEN = 4
"""OpenAI's own commonly cited rule of thumb for English prose: roughly
one token per four characters. This is a deliberately simple approximation,
not a client side reimplementation of BPE; it exists only to keep the
offline chunk budget in the same rough regime as the real tokenizer, never
to be shown to a user or used to bill anything."""


@lru_cache(maxsize=1)
def _encoding_or_error() -> tiktoken.Encoding | Exception:
    """Attempt the real cl100k_base load exactly once per process and
    remember the outcome, success or failure, the same way
    can_reach_huggingface caches its probe. Keeping the exception, rather
    than collapsing straight to a bool, lets a forced "tiktoken" backend
    chain the real cause into its RuntimeError instead of just saying no.
    """
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # network fetch failed, proxy denial, and so on
        return exc


def reset_encoding_cache() -> None:
    """Test-only: clear the cached load attempt."""
    _encoding_or_error.cache_clear()


def get_tokenizer_backend() -> TokenizerBackend:
    """Which backend count_tokens() is actually using right now, honoring
    GROUNDWORK_TOKENIZER_BACKEND. Chunking records this per chunk
    (ChunkCandidate.tokenizer_backend) exactly as Chunk.embedding_backend
    records which embedding backend produced a stored vector, so a
    mixed-backend run is labelled honestly instead of silently blended."""
    choice = get_settings().tokenizer_backend
    if choice != "auto":
        return choice
    result = _encoding_or_error()
    return "approximate" if isinstance(result, Exception) else "tiktoken"


def _require_real_encoding() -> tiktoken.Encoding:
    result = _encoding_or_error()
    if isinstance(result, Exception):
        raise RuntimeError(
            "GROUNDWORK_TOKENIZER_BACKEND=tiktoken but the cl100k_base encoding "
            "could not be loaded. Use 'auto' to fall back to the offline "
            "approximation automatically, or 'approximate' to force it."
        ) from result
    return result


def _approximate_token_count(text: str) -> int:
    """A deterministic, offline stand-in for count_tokens() when the real
    cl100k_base ranks cannot be loaded. Not meant to match cl100k_base
    token for token, only to keep chunk boundaries in the same rough
    regime regardless of which backend produced them.
    """
    return math.ceil(len(text) / _APPROX_CHARS_PER_TOKEN)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if get_tokenizer_backend() == "approximate":
        return _approximate_token_count(text)
    return len(_require_real_encoding().encode(text))
