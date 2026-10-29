"""Out of scope refusal scoring: section 11's own requirement, "scored as
a hard pass or fail, not a quality judgment... a question in that category
passes only if no citation is returned and the response matches a stated
refusal pattern, checked in code, not by asking a model whether the
refusal was good."

Deterministic means exact, not a heuristic match against words like "not"
or "sorry" that a confident wrong answer could just as easily contain.
Every code path in this build that legitimately has nothing to answer
from, groundwork_api.chat.ask's relevance gate, an empty retrieval
result, or a real LLM asked about chunks it was never given, already
funnels through groundwork_generate.generate.NOT_COVERED_MESSAGE, the one
fixed string, so exact equality against it is the correct check, not a
softer one.

A real model shown passages that do not answer the question used to be the
gap: told only to "say plainly" that the document does not cover it, it
could word that any number of ways, and every passage it was offered was
counted as cited, so a model refusal could never pass. Both are closed on
the model path now. The system prompt tells the model to reply with exactly
NOT_COVERED_MESSAGE and nothing else, and groundwork_api.chat.ask counts as
cited only the passages the model's markers name, so a refusal cites
nothing. The check itself stays exact: a model that paraphrases the
refusal anyway fails it, on purpose, since a softer match would also pass a
confident wrong answer that happens to apologize. MODEL-RESULTS.md reports
how the real model runs scored against it.
"""

from __future__ import annotations

from groundwork_generate.generate import NOT_COVERED_MESSAGE


def is_out_of_scope_refusal(answer: str, *, cited_chunk_count: int) -> bool:
    """True only when nothing was cited and the answer is exactly this
    build's own stated refusal message, not merely similar to it."""
    return cited_chunk_count == 0 and answer.strip() == NOT_COVERED_MESSAGE
