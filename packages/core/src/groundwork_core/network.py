"""A single, cached probe for whether this process can reach the Hugging
Face model hub.

Every "auto" backend choice in config.py calls can_reach_huggingface() once
and reuses the answer for the rest of the process. The probe itself is
cheap (a HEAD request with a short timeout) and never raises: any
exception, timeout or non-2xx response is treated as "unreachable", because
the caller's job either way is to pick a backend and move on, not to
diagnose the network.
"""

from __future__ import annotations

import functools
import urllib.request

HF_PROBE_URL = "https://huggingface.co/api/models/BAAI/bge-small-en-v1.5"


@functools.lru_cache(maxsize=1)
def can_reach_huggingface(timeout_seconds: float = 3.0) -> bool:
    try:
        req = urllib.request.Request(HF_PROBE_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310
            status = int(resp.status)
            return 200 <= status < 300
    except Exception:
        return False


def reset_probe_cache() -> None:
    """Test-only: clear the cached probe result."""
    can_reach_huggingface.cache_clear()
