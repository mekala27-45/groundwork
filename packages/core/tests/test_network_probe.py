from __future__ import annotations

from groundwork_core.network import can_reach_huggingface, reset_probe_cache


def test_probe_result_is_cached_across_calls() -> None:
    reset_probe_cache()
    first = can_reach_huggingface()
    second = can_reach_huggingface()
    assert first is second, "the probe must not re-query the network on every call"


def test_probe_never_raises_and_returns_a_bool() -> None:
    reset_probe_cache()
    result = can_reach_huggingface(timeout_seconds=0.001)
    assert isinstance(result, bool)
