"""Property tests held in common across every chunking strategy (section
15 of the day 5 build prompt): no chunk exceeds the token budget unless it
is a single oversized atom, every atom in the source document survives
into at least one chunk, and a table is never split across the chunk
boundary that contains it. Parametrized over STRATEGIES so a new strategy
is held to the identical bar by adding it to that list, not by writing a
parallel set of tests for it.
"""

from __future__ import annotations

from typing import Protocol

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from groundwork_chunk import ChunkCandidate, chunk_naive
from groundwork_chunk.models import Atom
from groundwork_chunk.packing import build_canonical_text
from groundwork_chunk.segments import build_atoms
from groundwork_chunk.tokenize import count_tokens
from groundwork_ingest.models import (
    ExtractedDocument,
    ExtractedPage,
    ExtractedSpan,
    ExtractedTable,
)


class ChunkStrategy(Protocol):
    """The signature chunk_naive and chunk_structure both share. Typing
    the parametrized chunk_fn against this, rather than leaving it
    untyped, is what lets mypy strict actually check the bodies below."""

    def __call__(
        self,
        doc: ExtractedDocument,
        *,
        chunk_size_tokens: int = ...,
        overlap_tokens: int = ...,
    ) -> list[ChunkCandidate]: ...


STRATEGIES: list[ChunkStrategy] = [chunk_naive]
"""Extended to include chunk_structure once structure aware chunking
lands; every property below is meant to hold for any chunking strategy
built on top of packing.pack_atom_range, not just this one."""

_LOWERCASE = list("abcdefghijklmnopqrstuvwxyz")
_ALNUM = list("abcdefghijklmnopqrstuvwxyz0123456789")


def _span(text: str, *, page: int = 1) -> ExtractedSpan:
    return ExtractedSpan(
        text=text,
        page_number=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        font_size=11.0,
        color_rgb=(0, 0, 0),
        font_name="Helvetica",
        is_bold=False,
    )


def _table(rows: list[list[str | None]], *, page: int = 1) -> ExtractedTable:
    return ExtractedTable(page_number=page, bbox=(0.0, 0.0, 10.0, 10.0), rows=list(rows))


def _page(
    *, page_number: int, spans: list[ExtractedSpan], tables: list[ExtractedTable]
) -> ExtractedPage:
    text = " ".join(s.text for s in spans)
    return ExtractedPage(
        page_number=page_number,
        text=text,
        spans=spans,
        tables=tables,
        char_count=len(text),
        page_area_pt2=1000.0,
        text_density=0.05,
        used_ocr=False,
    )


@st.composite
def documents(draw: st.DrawFn) -> ExtractedDocument:
    """One to three pages of plain words, each page optionally carrying
    one small table. No header sized spans: that is structure-aware
    chunking's own concern, exercised once it joins STRATEGIES."""
    n_pages = draw(st.integers(min_value=1, max_value=3))
    pages: list[ExtractedPage] = []
    for page_num in range(1, n_pages + 1):
        words = draw(
            st.lists(st.text(alphabet=_LOWERCASE, min_size=1, max_size=9), min_size=1, max_size=60)
        )
        spans = [_span(w, page=page_num) for w in words]

        tables: list[ExtractedTable] = []
        if draw(st.booleans()):
            n_rows = draw(st.integers(min_value=1, max_value=3))
            n_cols = draw(st.integers(min_value=1, max_value=3))
            cell = st.text(alphabet=_ALNUM, min_size=1, max_size=6)
            rows: list[list[str | None]] = [
                [draw(cell) for _ in range(n_cols)] for _ in range(n_rows)
            ]
            tables.append(_table(rows, page=page_num))

        pages.append(_page(page_number=page_num, spans=spans, tables=tables))
    return ExtractedDocument(
        filename="prop.pdf", sha256="0" * 64, page_count=len(pages), pages=pages
    )


def _atom_range(
    atoms: list[Atom], spans: list[tuple[int, int]], chunk: ChunkCandidate
) -> tuple[int, int]:
    """The [start, end) atom index range a chunk was built from, recovered
    from its char_start/char_end against an independently recomputed atom
    sequence for the same document. Safe because build_atoms and
    build_canonical_text are pure functions of the document: calling them
    again here reproduces exactly the spans the strategy under test built
    internally."""
    start_idx = next(i for i, (s, _e) in enumerate(spans) if s == chunk.char_start)
    end_idx = next(i for i, (_s, e) in enumerate(spans) if e == chunk.char_end)
    return start_idx, end_idx + 1


_budget = st.integers(min_value=1, max_value=40)
_overlap = st.integers(min_value=0, max_value=40)


@pytest.mark.parametrize("chunk_fn", STRATEGIES)
@settings(max_examples=50, deadline=None)
@given(doc=documents(), chunk_size_tokens=_budget, overlap_tokens=_overlap)
def test_no_chunk_exceeds_the_token_budget_unless_it_is_a_single_atom(
    chunk_fn: ChunkStrategy, doc: ExtractedDocument, chunk_size_tokens: int, overlap_tokens: int
) -> None:
    atoms = build_atoms(doc)
    _, spans = build_canonical_text(atoms)
    chunks = chunk_fn(doc, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)

    for chunk in chunks:
        start_idx, end_idx = _atom_range(atoms, spans, chunk)
        is_single_atom = (end_idx - start_idx) == 1
        assert count_tokens(chunk.text) <= chunk_size_tokens or is_single_atom, (
            f"chunk over budget ({count_tokens(chunk.text)} > {chunk_size_tokens}) "
            f"and not a single atom: {chunk.text!r}"
        )


@pytest.mark.parametrize("chunk_fn", STRATEGIES)
@settings(max_examples=50, deadline=None)
@given(doc=documents(), chunk_size_tokens=_budget, overlap_tokens=_overlap)
def test_every_atom_survives_into_at_least_one_chunk(
    chunk_fn: ChunkStrategy, doc: ExtractedDocument, chunk_size_tokens: int, overlap_tokens: int
) -> None:
    atoms = build_atoms(doc)
    _, spans = build_canonical_text(atoms)
    chunks = chunk_fn(doc, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)

    covered = [False] * len(atoms)
    for chunk in chunks:
        start_idx, end_idx = _atom_range(atoms, spans, chunk)
        for i in range(start_idx, end_idx):
            covered[i] = True

    assert all(covered), "every atom in the source document must land in at least one chunk"


@pytest.mark.parametrize("chunk_fn", STRATEGIES)
@settings(max_examples=50, deadline=None)
@given(doc=documents(), chunk_size_tokens=_budget, overlap_tokens=_overlap)
def test_a_table_is_never_split_across_the_chunks_that_contain_it(
    chunk_fn: ChunkStrategy, doc: ExtractedDocument, chunk_size_tokens: int, overlap_tokens: int
) -> None:
    atoms = build_atoms(doc)
    _, spans = build_canonical_text(atoms)
    chunks = chunk_fn(doc, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)
    ranges = [(chunk, *_atom_range(atoms, spans, chunk)) for chunk in chunks]

    for table_idx, atom in enumerate(atoms):
        if atom.kind != "table":
            continue
        containing = [chunk for chunk, start, end in ranges if start <= table_idx < end]
        assert containing, "a table atom must be covered by at least one chunk"
        for chunk in containing:
            assert atom.text in chunk.text, (
                "a table's serialized text must appear whole in any chunk that "
                "contains it, never truncated by the overlap logic"
            )
