"""Monotonic UUIDv7 ids, ported from trajectory (day 1).

UUIDv7 embeds a millisecond timestamp in the high bits, so ids sort in
creation order without a separate created_at index scan. Python 3.12's
stdlib uuid module does not yet expose uuid7 directly (that lands in 3.14),
so this is the same small RFC 9562 compliant implementation carried forward
from day 1, with a per-process monotonic counter guarding against two ids
minted in the same millisecond from sorting out of order.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

_lock = threading.Lock()
_last_ms = 0
_seq = 0


def new_id() -> uuid.UUID:
    """Return a new monotonic UUIDv7."""
    global _last_ms, _seq
    with _lock:
        now_ms = time.time_ns() // 1_000_000
        if now_ms <= _last_ms:
            now_ms = _last_ms
            _seq += 1
        else:
            _last_ms = now_ms
            _seq = 0
        seq = _seq

    rand_bytes = os.urandom(8)
    seq_bits = seq & 0xFFF

    time_hi = (now_ms >> 16) & 0xFFFFFFFF
    time_lo = now_ms & 0xFFFF

    b = bytearray(16)
    b[0:4] = time_hi.to_bytes(4, "big")
    b[4:6] = time_lo.to_bytes(2, "big")
    b[6] = 0x70 | ((seq_bits >> 8) & 0x0F)
    b[7] = seq_bits & 0xFF
    b[8] = 0x80 | (rand_bytes[0] & 0x3F)
    b[9:16] = rand_bytes[1:8]

    return uuid.UUID(bytes=bytes(b))
