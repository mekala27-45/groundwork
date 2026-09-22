"""Resilience tests for tokenize.py. The real cl100k_base load is a network
fetch (see the module docstring) that this sandbox's own egress policy
rejects, so every test here drives the fallback deliberately through
mocking rather than depending on what a given environment's network allows
through on the day the suite runs.

Patched by dotted string ("tiktoken.get_encoding") rather than through
tokenize.tiktoken: tokenize.py merely imports the tiktoken module, it does
not re-export it, so reaching it as an attribute of tokenize from outside
is what mypy strict's no_implicit_reexport is there to catch. Patching
tiktoken's own name is also the more accurate description of what is being
faked: the real third party call, not this module's reference to it.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from groundwork_chunk import tokenize


@pytest.fixture(autouse=True)
def _reset_cache() -> Iterator[None]:
    tokenize.reset_encoding_cache()
    yield
    tokenize.reset_encoding_cache()


def test_empty_string_is_zero_tokens_without_loading_any_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = MagicMock(side_effect=AssertionError("must not load an encoding for empty input"))
    monkeypatch.setattr("tiktoken.get_encoding", spy)

    assert tokenize.count_tokens("") == 0
    spy.assert_not_called()


def test_auto_uses_the_real_encoding_when_it_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = MagicMock()
    stub.encode.return_value = [1, 2, 3]
    monkeypatch.setattr("tiktoken.get_encoding", MagicMock(return_value=stub))

    assert tokenize.get_tokenizer_backend() == "tiktoken"
    assert tokenize.count_tokens("three tokens please") == 3
    stub.encode.assert_called_once_with("three tokens please")


def test_auto_falls_back_when_the_real_encoding_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tiktoken.get_encoding",
        MagicMock(side_effect=ConnectionError("blocked by egress policy")),
    )

    assert tokenize.get_tokenizer_backend() == "approximate"
    assert tokenize.count_tokens("hello world") == tokenize._approximate_token_count("hello world")


def test_failed_load_is_cached_not_retried_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = MagicMock(side_effect=ConnectionError("blocked by egress policy"))
    monkeypatch.setattr("tiktoken.get_encoding", spy)

    tokenize.count_tokens("first call")
    tokenize.count_tokens("second call")
    tokenize.get_tokenizer_backend()

    spy.assert_called_once()


def test_forcing_approximate_never_touches_the_real_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_TOKENIZER_BACKEND", "approximate")
    spy = MagicMock(side_effect=AssertionError("forced approximate must not load tiktoken"))
    monkeypatch.setattr("tiktoken.get_encoding", spy)

    assert tokenize.get_tokenizer_backend() == "approximate"
    assert tokenize.count_tokens("some text here") > 0
    spy.assert_not_called()


def test_forcing_tiktoken_uses_it_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDWORK_TOKENIZER_BACKEND", "tiktoken")
    stub = MagicMock()
    stub.encode.return_value = [1, 2]
    monkeypatch.setattr("tiktoken.get_encoding", MagicMock(return_value=stub))

    assert tokenize.get_tokenizer_backend() == "tiktoken"
    assert tokenize.count_tokens("two") == 2


def test_forcing_tiktoken_raises_loudly_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDWORK_TOKENIZER_BACKEND", "tiktoken")
    original = ConnectionError("blocked by egress policy")
    monkeypatch.setattr("tiktoken.get_encoding", MagicMock(side_effect=original))

    with pytest.raises(RuntimeError, match="GROUNDWORK_TOKENIZER_BACKEND=tiktoken") as excinfo:
        tokenize.count_tokens("anything")
    assert excinfo.value.__cause__ is original


def test_approximate_counter_is_deterministic_and_zero_for_empty_text() -> None:
    text = "a slightly longer sentence to approximate"
    assert tokenize._approximate_token_count(text) == tokenize._approximate_token_count(text)
    assert tokenize._approximate_token_count(text) > 0
    assert tokenize._approximate_token_count("") == 0
