"""The page boundary test named directly in the day 5 prompt (section 6):
a sentence and a table row are each deliberately split across a page
break, and extraction must stitch both back into one coherent unit rather
than truncating, duplicating, or inserting a spurious break.
"""

from __future__ import annotations

from pathlib import Path

from groundwork_ingest.extract import extract_pdf
from groundwork_ingest.fixtures import build_page_boundary_test_pdf


def test_sentence_split_across_page_break_is_not_truncated_or_duplicated(tmp_path: Path) -> None:
    pdf_path = tmp_path / "boundary.pdf"
    build_page_boundary_test_pdf(pdf_path)

    doc = extract_pdf(pdf_path)
    assert doc.page_count == 2

    page1_text = doc.pages[0].text
    page2_text = doc.pages[1].text

    # Not truncated: the last fragment written to page 1 is present in full.
    assert "the sentence continues directly" in page1_text
    # Not duplicated: that fragment appears exactly once across both pages.
    assert (page1_text + page2_text).count("continues directly") == 1
    # Not lost: the continuation on page 2 is present and intact.
    assert "onto the next page without losing a single word" in page2_text
    # Not a spurious break: reading page 1's tail directly into page 2's
    # head (normalizing the page-break whitespace to a single space)
    # reconstructs the original, uninterrupted sentence.
    joined = (page1_text.rstrip() + " " + page2_text.lstrip()).replace("\n", " ")
    assert "continues directly onto the next page without losing a single word" in joined


def test_table_row_split_across_page_break_is_stitched(tmp_path: Path) -> None:
    pdf_path = tmp_path / "boundary.pdf"
    build_page_boundary_test_pdf(pdf_path)

    doc = extract_pdf(pdf_path)

    all_tables = doc.pages[0].tables + doc.pages[1].tables
    assert len(all_tables) == 1, "the two page-level table fragments must merge into one table"

    table = all_tables[0]
    assert table.page_number == 1
    assert table.page_number_end == 2
    row_texts = ["".join(cell or "" for cell in row) for row in table.rows]
    assert any("Package" in r and "Coverage" in r for r in row_texts), "header row present"
    assert any("core" in r and "94%" in r for r in row_texts), "row split-free on page 1 present"
    assert any("ingest" in r and "91%" in r for r in row_texts), "row continuing on page 2 present"


def test_no_character_lost_between_source_and_extracted_pages(tmp_path: Path) -> None:
    """A property closer to the underlying guarantee than any single
    string check: every word placed on the fixture PDF shows up somewhere
    in the extracted text, once."""
    pdf_path = tmp_path / "boundary.pdf"
    build_page_boundary_test_pdf(pdf_path)
    doc = extract_pdf(pdf_path)

    expected_words = {
        "Package",
        "Coverage",
        "Status",
        "core",
        "94%",
        "passing",
        "ingest",
        "91%",
    }
    full_text_words = set(doc.full_text.replace("\n", " ").split())
    missing = expected_words - full_text_words
    assert not missing, f"words dropped during extraction: {missing}"
