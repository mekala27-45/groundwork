"""Log redaction: document content never reaches a log aggregator.

Hard constraint (day 5 prompt, stack section): structlog output is JSON,
and no document content is ever logged at info level or above, only chunk
ids and content hashes. This module is the single choke point that every
logger call for chunk or document text must go through, so the rule is
enforced in one place rather than trusted to every call site.
"""

from __future__ import annotations

import hashlib
from typing import Any


def content_hash(text: str) -> str:
    """A stable, non-reversible fingerprint safe to log in place of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def redact_for_log(payload: dict[str, Any], *, text_fields: tuple[str, ...]) -> dict[str, Any]:
    """Return a copy of payload with each field in text_fields replaced by
    its length and content hash rather than its raw value.

    Any field not named in text_fields passes through unchanged, so callers
    must name every content-bearing field explicitly rather than relying on
    a deny list that a future field could silently miss.
    """
    out = dict(payload)
    for field in text_fields:
        if field in out and isinstance(out[field], str):
            value = out[field]
            out[field] = {"length": len(value), "sha256_16": content_hash(value)}
    return out
