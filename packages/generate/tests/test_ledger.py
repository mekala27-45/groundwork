"""SpendLedger tests. No database involved: the ledger is a plain in
memory counter, so every test here runs unconditionally, unlike the
requires_postgres suites elsewhere in this workspace.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from groundwork_generate.ledger import SpendLedger, get_ledger, reset_ledger_cache


def test_has_budget_true_while_under_the_ceiling() -> None:
    ledger = SpendLedger(ceiling_usd=2.00)
    assert ledger.has_budget()


def test_has_budget_false_once_spend_reaches_the_ceiling() -> None:
    ledger = SpendLedger(ceiling_usd=1.00)
    ledger.record(1.00)
    assert not ledger.has_budget()


def test_has_budget_false_once_spend_exceeds_the_ceiling() -> None:
    ledger = SpendLedger(ceiling_usd=1.00)
    ledger.record(1.50)
    assert not ledger.has_budget()


def test_record_accumulates_across_multiple_calls() -> None:
    ledger = SpendLedger(ceiling_usd=1.00)
    ledger.record(0.25)
    ledger.record(0.40)
    assert ledger.spent_usd == pytest.approx(0.65)
    assert ledger.has_budget()


def test_remaining_usd_reflects_what_is_left() -> None:
    ledger = SpendLedger(ceiling_usd=2.00)
    ledger.record(0.75)
    assert ledger.remaining_usd == pytest.approx(1.25)


def test_remaining_usd_never_goes_negative_past_the_ceiling() -> None:
    ledger = SpendLedger(ceiling_usd=1.00)
    ledger.record(5.00)
    assert ledger.remaining_usd == 0.0


@pytest.fixture(autouse=True)
def _reset_ledger_cache() -> Iterator[None]:
    reset_ledger_cache()
    yield
    reset_ledger_cache()


def test_get_ledger_caches_the_constructed_instance() -> None:
    assert get_ledger() is get_ledger()


def test_get_ledger_uses_the_configured_spend_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDWORK_SPEND_CEILING_USD", "5.00")
    assert get_ledger().ceiling_usd == pytest.approx(5.00)


def test_reset_ledger_cache_produces_a_fresh_zero_spent_instance() -> None:
    get_ledger().record(1.00)
    reset_ledger_cache()
    assert get_ledger().spent_usd == 0.0
