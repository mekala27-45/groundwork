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
softer one. The one gap this does not close: a real LLM shown chunks it
judges insufficient (the system prompt's own free-form "say plainly that
the document does not cover this question" instruction, section 9) can
phrase that refusal any number of ways, none of them this exact string,
and this function would not recognize it. This sandbox has no LLM key to
exercise that path at all, so the gap is real but unmeasured here;
RESULTS.md and docs/security.md say so rather than implying it was
checked.
"""

from __future__ import annotations

from groundwork_generate.generate import NOT_COVERED_MESSAGE


def is_out_of_scope_refusal(answer: str, *, cited_chunk_count: int) -> bool:
    """True only when nothing was cited and the answer is exactly this
    build's own stated refusal message, not merely similar to it."""
    return cited_chunk_count == 0 and answer.strip() == NOT_COVERED_MESSAGE
