"""Plain data shapes produced by extraction, independent of how a page's
text was obtained (native layer or OCR), so everything downstream
(chunking, the page boundary test) can treat both sources identically.
"""

from __future__ import annotations

from groundwork_core import StrictModel


class ExtractedSpan(StrictModel):
    """One PyMuPDF text span: a run of characters sharing one font, size
    and color. The unit security.py inspects for near-invisible styling."""

    text: str
    page_number: int
    bbox: tuple[float, float, float, float]
    font_size: float
    color_rgb: tuple[int, int, int]
    """0-255 per channel, decoded from PyMuPDF's packed sRGB integer."""
    font_name: str
    is_bold: bool


class ExtractedTable(StrictModel):
    page_number: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str | None]]
    page_number_end: int | None = None
    """Set only when stitch_cross_page_tables merged this table with one
    continuing at the top of the next page. None means the table lives
    entirely on page_number."""


class ExtractedPage(StrictModel):
    page_number: int
    text: str
    spans: list[ExtractedSpan]
    tables: list[ExtractedTable]
    char_count: int
    page_area_pt2: float
    text_density: float
    """char_count divided by page_area_pt2. Compared against
    Settings.ocr_density_threshold to decide whether this page needed OCR."""
    used_ocr: bool
    injection_flag: str | None = None


class ExtractedDocument(StrictModel):
    filename: str
    sha256: str
    page_count: int
    pages: list[ExtractedPage]

    @property
    def extraction_method(self) -> str:
        used_ocr = [p.used_ocr for p in self.pages]
        if not any(used_ocr):
            return "text"
        if all(used_ocr):
            return "ocr"
        return "mixed"

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)
