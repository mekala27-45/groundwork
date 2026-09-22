"""Claim splitting: turns a generated answer's raw text into sentence
level claims, each resolved back to the chunk its citation marker points
at, per section 10 of the day 5 prompt ("Split each generated answer into
individual sentence level claims").

Pure text parsing, nothing else: this module knows nothing about
extractive_fallback or how an answer was produced, only how to read the
"[N]" style markers groundwork_generate.generate's system prompt asks a
real model to write, matching the numbering
groundwork_generate.generate.GenerationResult.cited_chunk_ids already
uses (cited_chunk_ids[0] is what "[1]" refers to). Whether an extractive
answer even needs this treatment at all is a decision
groundwork_verify.faithfulness.check_faithfulness makes on its own; see
that module's docstring for why it is answered there rather than here.
"""

from __future__ import annotations

import re
from uuid import UUID

from groundwork_core.model import StrictModel

_CITATION_MARKER = re.compile(r"\[(\d+)\]")
_WHITESPACE = re.compile(r"\s+")

# Splits after sentence ending punctuation followed by whitespace, same as
# any plain sentence splitter, with one addition: a [N] marker directly
# after that punctuation is not itself a new sentence, so the split is
# withheld until after the marker instead (matched by the second
# alternative, whitespace right after a closing bracket). Without this, "
# Sessions run fifty minutes. [1]" would split into "Sessions run fifty
# minutes." and the marker-only fragment "[1]", stranding the marker with
# no sentence text left to attach it to once brackets are stripped.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\[\d+\])|(?<=\])\s+")


class Claim(StrictModel):
    text: str
    """The claim's sentence text, with its citation marker(s) removed."""
    cited_chunk_id: UUID | None
    """None when no [N] marker was found for this sentence, at or after
    it, anywhere in the rest of the answer: an uncited claim, which
    faithfulness checking should flag rather than silently skip."""


def _clean(sentence: str) -> str:
    return _WHITESPACE.sub(" ", _CITATION_MARKER.sub("", sentence)).strip()


def split_claims(answer: str, cited_chunk_ids: list[UUID]) -> list[Claim]:
    """Splits answer into sentences and resolves each one's citation from
    the nearest [N] marker at or after it, not only a marker typeset
    directly against that one sentence. A citation commonly closes out a
    short run of sentences that together make one point ("Sessions run
    fifty minutes. They are held over video call. [1]") rather than only
    the clause it immediately follows, so an unmarked sentence inherits
    the next marker found after it. A trailing run with no marker
    anywhere after it resolves to cited_chunk_id=None.

    A marker naming a number outside cited_chunk_ids' range (an index a
    real model hallucinated, or simply off by one) resolves the same way,
    None, rather than raising: a malformed citation is exactly the kind
    of thing this whole verification layer exists to catch and report,
    not to crash on.
    """
    raw_sentences = [s for s in _SENTENCE_SPLIT.split(answer.strip()) if s.strip()]

    parsed: list[tuple[str, int | None]] = []
    for raw in raw_sentences:
        markers = _CITATION_MARKER.findall(raw)
        text = _clean(raw)
        if not text:
            continue
        marker_index = int(markers[-1]) - 1 if markers else None
        parsed.append((text, marker_index))

    claims: list[Claim] = []
    pending_index: int | None = None
    for text, marker_index in reversed(parsed):
        if marker_index is not None:
            pending_index = marker_index
        cited_chunk_id = (
            cited_chunk_ids[pending_index]
            if pending_index is not None and 0 <= pending_index < len(cited_chunk_ids)
            else None
        )
        claims.append(Claim(text=text, cited_chunk_id=cited_chunk_id))
    claims.reverse()
    return claims
