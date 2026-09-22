"""Naive strategy specific tests: fixed token windows, no structure
awareness. The three properties shared with structure-aware chunking
(token budget, lossless atom coverage, tables never split) live in
test_chunking_properties.py so both strategies are held to the identical
bar; this file covers naive-only behavior: empty input, single-chunk
documents, the overlap window, page span, and backend labelling.
"""

from __future__ import annotations

import pytest

from groundwork_chunk import chunk_naive
from groundwork_chunk.tokenize import get_tokenizer_backend
from groundwork_ingest.models import ExtractedDocument, ExtractedPage, ExtractedSpan


@pytest.fixture(autouse=True)
def _force_approximate_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exact chunk boundaries below are hand traced against the approximate
    # backend's ceil(len/4) rule. Forcing it here decouples this file from
    # whether the sandbox running it can reach tiktoken's blob host.
    monkeypatch.setenv("GROUNDWORK_TOKENIZER_BACKEND", "approximate")


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


def _words_doc(words: list[str], *, page: int = 1) -> ExtractedDocument:
    spans = [_span(w, page=page) for w in words]
    text = " ".join(words)
    page_obj = ExtractedPage(
        page_number=page,
        text=text,
        spans=spans,
        tables=[],
        char_count=len(text),
        page_area_pt2=1000.0,
        text_density=0.05,
        used_ocr=False,
    )
    return ExtractedDocument(filename="naive.pdf", sha256="0" * 64, page_count=1, pages=[page_obj])


def _empty_doc() -> ExtractedDocument:
    page_obj = ExtractedPage(
        page_number=1,
        text="",
        spans=[],
        tables=[],
        char_count=0,
        page_area_pt2=1000.0,
        text_density=0.0,
        used_ocr=False,
    )
    return ExtractedDocument(filename="empty.pdf", sha256="0" * 64, page_count=1, pages=[page_obj])


def test_empty_document_produces_no_chunks() -> None:
    assert chunk_naive(_empty_doc()) == []


def test_short_document_under_budget_is_a_single_chunk() -> None:
    doc = _words_doc(["a", "short", "document"])

    chunks = chunk_naive(doc, chunk_size_tokens=100, overlap_tokens=10)

    assert len(chunks) == 1
    assert chunks[0].text == "a short document"
    assert chunks[0].strategy == "naive"
    assert chunks[0].section_title is None


def test_overlap_shares_the_boundary_word_with_the_next_chunk() -> None:
    words = ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff"]
    doc = _words_doc(words)

    chunks = chunk_naive(doc, chunk_size_tokens=3, overlap_tokens=1)

    assert len(chunks) >= 2
    assert chunks[0].text.split()[0] == "aaaa"
    assert chunks[-1].text.split()[-1] == "ffff"
    for prev_chunk, next_chunk in zip(chunks, chunks[1:], strict=False):
        assert prev_chunk.text.split()[-1] == next_chunk.text.split()[0], (
            "consecutive chunks must share the overlap word at their boundary"
        )


def test_zero_overlap_produces_no_shared_words_between_chunks() -> None:
    words = ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff"]
    doc = _words_doc(words)

    chunks = chunk_naive(doc, chunk_size_tokens=2, overlap_tokens=0)

    assert len(chunks) >= 2
    seen: set[str] = set()
    for chunk in chunks:
        chunk_words = chunk.text.split()
        assert not (seen & set(chunk_words)), (
            "no word should repeat across chunks with zero overlap"
        )
        seen.update(chunk_words)


def test_records_which_tokenizer_backend_produced_each_chunk() -> None:
    doc = _words_doc(["alpha", "beta", "gamma"])

    chunks = chunk_naive(doc, chunk_size_tokens=100, overlap_tokens=0)

    backend = get_tokenizer_backend()
    assert all(c.tokenizer_backend == backend for c in chunks)


def test_page_span_is_a_single_page_within_one_page_and_spans_both_across_a_break() -> None:
    single_page_doc = _words_doc(["alpha", "beta", "gamma"], page=1)
    chunk = chunk_naive(single_page_doc, chunk_size_tokens=100, overlap_tokens=0)[0]
    assert chunk.page_start == chunk.page_end == 1

    page1 = ExtractedPage(
        page_number=1,
        text="alpha",
        spans=[_span("alpha", page=1)],
        tables=[],
        char_count=5,
        page_area_pt2=1000.0,
        text_density=0.05,
        used_ocr=False,
    )
    page2 = ExtractedPage(
        page_number=2,
        text="beta",
        spans=[_span("beta", page=2)],
        tables=[],
        char_count=4,
        page_area_pt2=1000.0,
        text_density=0.05,
        used_ocr=False,
    )
    two_page_doc = ExtractedDocument(
        filename="two-page.pdf", sha256="0" * 64, page_count=2, pages=[page1, page2]
    )
    chunk = chunk_naive(two_page_doc, chunk_size_tokens=100, overlap_tokens=0)[0]
    assert chunk.page_start == 1
    assert chunk.page_end == 2
