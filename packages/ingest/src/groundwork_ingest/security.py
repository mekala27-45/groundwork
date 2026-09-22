"""The ingestion-time injection heuristic.

Two independent signals, either one enough to flag a page: near-invisible
styling (the delivery mechanism an attacker uses so a human skimming the
PDF never sees the instruction) and instruction-shaped language (the
payload itself). This runs on every ingested document, on every page, and
its verdict is stored and surfaced in the eval dashboard even when nothing
downstream ever acts on the flagged text, per the hard constraint that a
real product raises this to a human regardless of whether a later defense
held.

This is a heuristic, not the defense. The actual defense is structural:
retrieved chunk text is passed to the generation model labelled as data,
never as instructions (packages/generate/src/groundwork_generate/prompts.py),
and packages/verify checks whether that boundary held. Flagging here is a
second, independent signal a human reviewing the eval dashboard can use
even on a run where the structural defense worked and nothing "happened".
"""

from __future__ import annotations

import re

from groundwork_ingest.models import ExtractedSpan

# A page background is assumed white (the overwhelming common case for a
# generated or scanned PDF); a span within this distance of pure white in
# every channel is treated as invisible against it. 250 out of 255 catches
# "255,255,254" as well as literal white.
NEAR_WHITE_THRESHOLD = 250

MIN_TRUSTED_FONT_SIZE = 1.0
"""Below this point size, text is not legible to a human reading the
rendered page even though it is fully present in the extracted layer."""

_INSTRUCTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\bdisregard\s+(all\s+|any\s+)?(previous|prior|above)\b", re.I),
    re.compile(r"\breveal\s+(your\s+)?(system\s+prompt|instructions)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(a|an)\b", re.I),
    re.compile(r"\bnew\s+instructions?\s*:", re.I),
    re.compile(r"\boutput\s+(the\s+following|exactly)\b", re.I),
    re.compile(r"\bdo\s+not\s+(tell|inform|mention)\s+the\s+user\b", re.I),
    re.compile(r"\bthis\s+is\s+a\s+system\s+message\b", re.I),
]


def _is_near_invisible(span: ExtractedSpan) -> bool:
    r, g, b = span.color_rgb
    near_white = (
        r >= NEAR_WHITE_THRESHOLD and g >= NEAR_WHITE_THRESHOLD and b >= NEAR_WHITE_THRESHOLD
    )
    tiny = span.font_size < MIN_TRUSTED_FONT_SIZE
    return near_white or tiny


def _matches_instruction_language(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INSTRUCTION_PATTERNS)


def flag_suspicious_content(spans: list[ExtractedSpan]) -> str | None:
    """Return a short machine readable reason string, or None.

    Deliberately returns at most one reason (the first match) rather than
    a list: the eval dashboard renders this as a single flag chip per
    chunk, and a page can be reflagged with more detail later without
    changing this function's contract.
    """
    invisible_spans = [s for s in spans if _is_near_invisible(s)]
    for span in invisible_spans:
        if _matches_instruction_language(span.text):
            return "near_invisible_instruction_language"
    if invisible_spans and any(len(s.text.strip()) > 20 for s in invisible_spans):
        return "near_invisible_text"
    full_text = " ".join(s.text for s in spans)
    if _matches_instruction_language(full_text):
        return "instruction_language"
    return None
