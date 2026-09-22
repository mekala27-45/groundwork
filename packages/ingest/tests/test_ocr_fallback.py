"""The OCR fallback test (section 6): a page with zero extractable text
layer must trigger the OCR path, and the text tesseract reads off the
raster image must contain the words that were actually rendered onto it.

Two separate tests, deliberately, per the carried-forward rule that a step
whose failure mode is silence needs a test asking whether it ran, apart
from whether its output looks right: test_ocr_path_is_invoked patches
ocr_page and asserts it was called at all, independent of what it returns.
test_ocr_extracts_expected_words then checks the real tesseract output on
the real fixture image, so a mocked "it ran" pass can never substitute for
a real "it worked" pass.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from groundwork_ingest.extract import extract_pdf
from groundwork_ingest.fixtures import build_ocr_only_test_pdf
from groundwork_ingest.ocr import is_tesseract_available, ocr_page

requires_tesseract = pytest.mark.skipif(
    not is_tesseract_available(), reason="tesseract binary not found on PATH"
)


@requires_tesseract
def test_low_density_page_triggers_ocr_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    build_ocr_only_test_pdf(pdf_path)

    with patch("groundwork_ingest.extract.ocr_page", wraps=ocr_page) as spy:
        doc = extract_pdf(pdf_path)

    spy.assert_called_once()
    assert doc.pages[0].used_ocr is True
    assert doc.pages[0].text_density < 0.02


def test_normal_text_page_never_calls_ocr(tmp_path: Path) -> None:
    from groundwork_ingest.fixtures import build_page_boundary_test_pdf

    pdf_path = tmp_path / "normal.pdf"
    build_page_boundary_test_pdf(pdf_path)

    with patch("groundwork_ingest.extract.ocr_page") as spy:
        doc = extract_pdf(pdf_path)

    spy.assert_not_called()
    assert all(not p.used_ocr for p in doc.pages)


@requires_tesseract
def test_ocr_extracts_the_words_actually_on_the_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    build_ocr_only_test_pdf(pdf_path, text_lines=["INVOICE NUMBER 88214", "Total due 412 dollars"])

    doc = extract_pdf(pdf_path)

    assert doc.extraction_method == "ocr"
    text = doc.pages[0].text.upper()
    assert "INVOICE" in text
    assert "88214" in text
    assert "412" in text
