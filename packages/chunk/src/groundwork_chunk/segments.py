"""Turns an ExtractedDocument into a flat sequence of Atoms: the shared
input both chunking strategies pack into windows. Doing header detection
here, once, rather than inside the structure-aware strategy, keeps naive
and structure chunking operating on literally the same units and means the
only difference between the two strategies is how they group atoms, not
what counts as an atom.
"""

from __future__ import annotations

import statistics

from groundwork_chunk.models import Atom
from groundwork_ingest.models import ExtractedDocument, ExtractedSpan, ExtractedTable

HEADER_SIZE_RATIO = 1.25
"""A span at least this many times the document's median body font size is
a header candidate on size alone."""

HEADER_BOLD_SIZE_RATIO = 1.05
"""A bold span only needs to clear this lower ratio, since boldness is
already a second independent signal."""

MAX_HEADER_CHARS = 120
"""Longer than this and it reads as a sentence, not a heading, regardless
of font size."""


def _median_font_size(doc: ExtractedDocument) -> float:
    sizes = [s.font_size for p in doc.pages for s in p.spans if s.text.strip()]
    if not sizes:
        return 11.0
    return statistics.median(sizes)


def _looks_like_header(span: ExtractedSpan, median_size: float) -> bool:
    text = span.text.strip()
    if not text or len(text) > MAX_HEADER_CHARS:
        return False
    if span.font_size >= median_size * HEADER_SIZE_RATIO:
        return True
    return bool(span.is_bold and span.font_size >= median_size * HEADER_BOLD_SIZE_RATIO)


def build_atoms(doc: ExtractedDocument) -> list[Atom]:
    median_size = _median_font_size(doc)
    atoms: list[Atom] = []

    for page in doc.pages:
        for span in page.spans:
            words = span.text.split()
            if not words:
                continue
            is_header = _looks_like_header(span, median_size)
            for word in words:
                atoms.append(
                    Atom(
                        kind="word",
                        text=word,
                        page_number=page.page_number,
                        is_section_start=is_header,
                    )
                )
        for table in page.tables:
            atoms.append(
                Atom(
                    kind="table",
                    text=_serialize_table(table),
                    page_number=table.page_number,
                    table=table,
                )
            )
    return atoms


def _serialize_table(table: ExtractedTable) -> str:
    lines = [" | ".join(cell or "" for cell in row) for row in table.rows]
    return "\n".join(lines)
