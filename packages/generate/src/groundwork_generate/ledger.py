"""A shared spend ledger: every real LLM call checks remaining budget
before it runs and records its actual cost after it returns, the same
"check before, record after" shape the trajectory case study describes
for day 1's token ledger and the frontdesk prompt describes for
LiteLLMJudge's own budget capped runs ("same spend ceiling checked before
every call"). Checking only after a call, or only once at process start,
would let a run already past its ceiling keep spending on every question
asked after the moment it crossed the line.

This ledger is a soft, cooperative gate sized for a single process under
asyncio's single threaded, cooperative concurrency, the same scale this
whole demo runs at. It is not a hardened, cross process rate limiter: two
plain method calls (has_budget() then, later, record()) are not one atomic
transaction, so a genuinely multi process or multi threaded deployment
could still race past the ceiling between them. Stating that plainly here
is more honest than adding a lock that would not actually close the gap
between those two calls anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass
class SpendLedger:
    """Tracks cumulative USD spent against a fixed ceiling."""

    ceiling_usd: float
    spent_usd: float = 0.0

    def has_budget(self) -> bool:
        """Checked before every real call: is there any budget left at
        all. This never tries to predict one specific call's cost ahead of
        time, since LiteLLM only reports a call's real cost after it
        completes; it only ever refuses a new call once the ledger is
        already exhausted."""
        return self.spent_usd < self.ceiling_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent_usd)

    def record(self, cost_usd: float) -> None:
        """Adds one completed call's real, reported cost to the running
        total."""
        self.spent_usd += cost_usd


@lru_cache(maxsize=1)
def get_ledger() -> SpendLedger:
    """The process wide ledger, constructed once from
    Settings.spend_ceiling_usd, the same cached singleton shape
    get_embedder() and get_reranker() use. A fresh process means a fresh
    budget: nothing here persists spend across restarts."""
    from groundwork_core.config import get_settings

    return SpendLedger(ceiling_usd=get_settings().spend_ceiling_usd)


def reset_ledger_cache() -> None:
    """Test-only: clear the cached ledger instance."""
    get_ledger.cache_clear()
