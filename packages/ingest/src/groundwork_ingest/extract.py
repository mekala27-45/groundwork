"""The main ingestion pass: PyMuPDF for text, layout and font metadata,
falling back to OCR per page when the native text layer is too sparse to
trust. This is the module the page boundary test and the OCR fallback
test both exercise directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf as fitz  # PyMuPDF; "fitz" is the deprecated import alias

from groundwork_ingest.models import ExtractedDocument, ExtractedPage, ExtractedSpan, ExtractedTable
from groundwork_ingest.ocr import ocr_page
from groundwork_ingest.security import flag_suspicious_content
from groundwork_ingest.tables import extract_tables, stitch_cross_page_tables

DEFAULT_OCR_DENSITY_THRESHOLD = 0.0001


def _decode_color(packed: int) -> tuple[int, int, int]:
    r = (packed >> 16) & 0xFF
    g = (packed >> 8) & 0xFF
    b = packed & 0xFF
    return r, g, b


def _spans_from_page(page: fitz.Page, page_number: int) -> tuple[list[ExtractedSpan], str]:
    # sort=True orders blocks by page position (top to bottom, then left to
    # right) rather than content-stream insertion order, which matters for
    # the page boundary test: the trailing sentence fragment must come out
    # last in reading order regardless of which order it was written in.
    raw = page.get_text("dict", sort=True)
    spans: list[ExtractedSpan] = []
    text_parts: list[str] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                line_parts.append(text)
                spans.append(
                    ExtractedSpan(
                        text=text,
                        page_number=page_number,
                        bbox=tuple(span["bbox"]),
                        font_size=float(span.get("size", 0.0)),
                        color_rgb=_decode_color(int(span.get("color", 0))),
                        font_name=span.get("font", ""),
                        is_bold=bool(span.get("flags", 0) & (1 << 4)),
                    )
                )
            if line_parts:
                text_parts.append("".join(line_parts))
    return spans, "\n".join(text_parts)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf(
    pdf_path: Path,
    *,
    ocr_density_threshold: float = DEFAULT_OCR_DENSITY_THRESHOLD,
) -> ExtractedDocument:
    doc = fitz.open(str(pdf_path))
    try:
        all_tables = extract_tables(pdf_path)
        all_tables = stitch_cross_page_tables(all_tables, page_sizes=_page_sizes(doc))
        tables_by_page: dict[int, list[ExtractedTable]] = {}
        for table in all_tables:
            tables_by_page.setdefault(table.page_number, []).append(table)

        pages: list[ExtractedPage] = []
        for i in range(1, doc.page_count + 1):
            page: fitz.Page = doc[i - 1]
            spans, text = _spans_from_page(page, i)
            area = float(page.rect.width * page.rect.height)
            char_count = len(text.strip())
            density = (char_count / area) if area > 0 else 0.0
            used_ocr = False

            if density < ocr_density_threshold:
                ocr_text = ocr_page(pdf_path, i)
                if len(ocr_text.strip()) > char_count:
                    text = ocr_text
                    char_count = len(text.strip())
                    used_ocr = True

            page_tables = tables_by_page.get(i, [])
            injection_flag = flag_suspicious_content(spans)

            pages.append(
                ExtractedPage(
                    page_number=i,
                    text=text,
                    spans=spans,
                    tables=page_tables,
                    char_count=char_count,
                    page_area_pt2=area,
                    text_density=density,
                    used_ocr=used_ocr,
                    injection_flag=injection_flag,
                )
            )

        return ExtractedDocument(
            filename=pdf_path.name,
            sha256=_file_sha256(pdf_path),
            page_count=len(pages),
            pages=pages,
        )
    finally:
        doc.close()


def _page_sizes(doc: fitz.Document) -> dict[int, tuple[float, float]]:
    return {
        i: (doc[i - 1].rect.width, doc[i - 1].rect.height) for i in range(1, doc.page_count + 1)
    }
