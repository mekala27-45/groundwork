"""Structure aware strategy specific tests: split on detected section
headers first, then further split any section that still exceeds the
token budget. The three properties shared with naive chunking (token
budget, lossless atom coverage, tables never split) live in
test_chunking_properties.py, parametrized over both strategies; this file
covers what only structure aware chunking does: header detection driving
section boundaries and title propagation.
"""

from __future__ import annotations

import pytest

from groundwork_chunk import chunk_naive
from groundwork_chunk.structure import chunk_structure
from groundwork_chunk.tokenize import get_tokenizer_backend
from groundwork_ingest.models import ExtractedDocument, ExtractedPage, ExtractedSpan

BODY_SIZE = 11.0
HEADER_SIZE = 20.0
"""Comfortably clears segments.HEADER_SIZE_RATIO (1.25) against BODY_SIZE,
so a span built at this size is unambiguously a header regardless of what
else shares the page."""


@pytest.fixture(autouse=True)
def _force_approximate_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDWORK_TOKENIZER_BACKEND", "approximate")


def _span(text: str, *, size: float = BODY_SIZE, bold: bool = False) -> ExtractedSpan:
    return ExtractedSpan(
        text=text,
        page_number=1,
        bbox=(0.0, 0.0, 10.0, 10.0),
        font_size=size,
        color_rgb=(0, 0, 0),
        font_name="Helvetica-Bold" if bold else "Helvetica",
        is_bold=bold,
    )


def _doc(spans: list[ExtractedSpan]) -> ExtractedDocument:
    text = " ".join(s.text for s in spans)
    page = ExtractedPage(
        page_number=1,
        text=text,
        spans=spans,
        tables=[],
        char_count=len(text),
        page_area_pt2=1000.0,
        text_density=0.05,
        used_ocr=False,
    )
    return ExtractedDocument(filename="structure.pdf", sha256="0" * 64, page_count=1, pages=[page])


def test_empty_document_produces_no_chunks() -> None:
    empty_page = ExtractedPage(
        page_number=1,
        text="",
        spans=[],
        tables=[],
        char_count=0,
        page_area_pt2=1000.0,
        text_density=0.0,
        used_ocr=False,
    )
    doc = ExtractedDocument(filename="empty.pdf", sha256="0" * 64, page_count=1, pages=[empty_page])

    assert chunk_structure(doc) == []


def test_document_with_no_headers_is_a_single_untitled_section() -> None:
    doc = _doc([_span(w) for w in ["plain", "body", "text", "only"]])

    chunks = chunk_structure(doc, chunk_size_tokens=100, overlap_tokens=0)

    assert len(chunks) == 1
    assert chunks[0].section_title is None
    assert chunks[0].strategy == "structure"


def test_leading_content_before_the_first_header_has_no_title() -> None:
    spans = [
        _span("intro", size=BODY_SIZE),
        _span("text", size=BODY_SIZE),
        _span("Findings", size=HEADER_SIZE),
        _span("the", size=BODY_SIZE),
        _span("results", size=BODY_SIZE),
    ]
    doc = _doc(spans)

    chunks = chunk_structure(doc, chunk_size_tokens=100, overlap_tokens=0)

    assert len(chunks) == 2
    assert chunks[0].section_title is None
    assert chunks[0].text == "intro text"
    assert chunks[1].section_title == "Findings"
    assert chunks[1].text == "Findings the results"


def test_multi_word_header_becomes_the_full_section_title() -> None:
    spans = [
        _span("Executive", size=HEADER_SIZE),
        _span("Summary", size=HEADER_SIZE),
        _span("body", size=BODY_SIZE),
        _span("text", size=BODY_SIZE),
    ]
    doc = _doc(spans)

    chunks = chunk_structure(doc, chunk_size_tokens=100, overlap_tokens=0)

    assert len(chunks) == 1
    assert chunks[0].section_title == "Executive Summary"


def test_each_header_starts_its_own_section() -> None:
    spans = [
        _span("Alpha", size=HEADER_SIZE),
        _span("one", size=BODY_SIZE),
        _span("Beta", size=HEADER_SIZE),
        _span("two", size=BODY_SIZE),
    ]
    doc = _doc(spans)

    chunks = chunk_structure(doc, chunk_size_tokens=100, overlap_tokens=0)

    assert [c.section_title for c in chunks] == ["Alpha", "Beta"]
    assert [c.text for c in chunks] == ["Alpha one", "Beta two"]


def test_a_bold_span_only_slightly_larger_than_body_is_still_a_header() -> None:
    # HEADER_BOLD_SIZE_RATIO (1.05) is a much lower bar than
    # HEADER_SIZE_RATIO (1.25), specifically because boldness is already a
    # second independent signal, per segments.py. Padded with enough plain
    # body words that the one bold span cannot drag the document's own
    # median font size up with it: real documents have far more body text
    # than header text, and the test document needs that same shape or it
    # is not testing the ratio it claims to.
    spans = [
        _span("Notes", size=BODY_SIZE * 1.1, bold=True),
        *[_span(w) for w in ["body", "text", "goes", "here", "plainly"]],
    ]
    doc = _doc(spans)

    chunks = chunk_structure(doc, chunk_size_tokens=100, overlap_tokens=0)

    assert chunks[0].section_title == "Notes"


def test_long_section_is_further_split_by_budget_but_keeps_its_title() -> None:
    words = ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff"]
    spans = [_span("Section", size=HEADER_SIZE), *[_span(w) for w in words]]
    doc = _doc(spans)

    chunks = chunk_structure(doc, chunk_size_tokens=3, overlap_tokens=0)

    assert len(chunks) > 1
    assert all(c.section_title == "Section" for c in chunks)
    assert all(c.strategy == "structure" for c in chunks)


def test_records_which_tokenizer_backend_produced_each_chunk() -> None:
    doc = _doc([_span("Title", size=HEADER_SIZE), _span("body")])

    chunks = chunk_structure(doc, chunk_size_tokens=100, overlap_tokens=0)

    backend = get_tokenizer_backend()
    assert all(c.tokenizer_backend == backend for c in chunks)


def test_naive_ignores_headers_that_structure_splits_on() -> None:
    """Not a claim that the two strategies always disagree (the retrieval
    evaluation in section 8 is where that gets measured for real), just
    that naive chunking is provably blind to the header structure
    structure aware chunking uses: a header sized span is just another
    word to it."""
    spans = [
        _span("Alpha", size=HEADER_SIZE),
        _span("one", size=BODY_SIZE),
        _span("Beta", size=HEADER_SIZE),
        _span("two", size=BODY_SIZE),
    ]
    doc = _doc(spans)

    naive_chunks = chunk_naive(doc, chunk_size_tokens=100, overlap_tokens=0)
    structure_chunks = chunk_structure(doc, chunk_size_tokens=100, overlap_tokens=0)

    assert len(naive_chunks) == 1
    assert naive_chunks[0].section_title is None
    assert len(structure_chunks) == 2
