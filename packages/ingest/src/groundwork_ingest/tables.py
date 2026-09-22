"""The table-focused second pass. Run separately from the main PyMuPDF text
and layout pass because table extraction from raw text runs is unreliable
and rarely worth trying to fix: pdfplumber's line-detection based table
finder does a specific job the general text extractor cannot.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from groundwork_ingest.models import ExtractedTable

MARGIN_FRACTION = 0.15
"""A table hugging within this fraction of the page height from the
bottom (previous page) or top (next page) edge is a stitching candidate.
Page boundary layout rarely leaves a table that close to the edge unless
it is either starting or continuing, which is the signal used here rather
than any semantic understanding of the table's content."""


def stitch_cross_page_tables(
    tables: list[ExtractedTable], page_sizes: dict[int, tuple[float, float]]
) -> list[ExtractedTable]:
    """Merge a table ending near the bottom of page N with one starting
    near the top of page N+1, when both have the same column count.

    pdfplumber finds tables per page and has no native concept of a table
    spanning a page break, so a row split by the break comes back as the
    last row of one table and the first row of an unrelated-looking next
    one. This is a heuristic, not a guarantee: it looks at position and
    shape, not content, and test_page_boundary_table_row documents exactly
    what it catches and what a much fancier version could catch instead.
    """
    if not tables:
        return []

    by_page: dict[int, list[ExtractedTable]] = {}
    for t in tables:
        by_page.setdefault(t.page_number, []).append(t)

    consumed: set[int] = set()
    merged: list[ExtractedTable] = []

    for t in tables:
        if id(t) in consumed:
            continue
        page_h = page_sizes.get(t.page_number, (0.0, 0.0))[1]
        near_bottom = page_h > 0 and (page_h - t.bbox[3]) <= page_h * MARGIN_FRACTION
        next_page_tables = by_page.get(t.page_number + 1, [])
        stitched = False
        if near_bottom:
            for candidate in next_page_tables:
                if id(candidate) in consumed:
                    continue
                cand_page_h = page_sizes.get(candidate.page_number, (0.0, 0.0))[1]
                near_top = cand_page_h > 0 and candidate.bbox[1] <= cand_page_h * MARGIN_FRACTION
                same_width = t.rows and candidate.rows and len(t.rows[0]) == len(candidate.rows[0])
                if near_top and same_width:
                    merged.append(
                        ExtractedTable(
                            page_number=t.page_number,
                            page_number_end=candidate.page_number,
                            bbox=t.bbox,
                            rows=[*t.rows, *candidate.rows],
                        )
                    )
                    consumed.add(id(t))
                    consumed.add(id(candidate))
                    stitched = True
                    break
        if not stitched and id(t) not in consumed:
            merged.append(t)
            consumed.add(id(t))

    return merged


def extract_tables(pdf_path: Path) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            for table in page.find_tables():
                rows = table.extract()
                if not rows or not any(any(cell for cell in row) for row in rows):
                    continue
                tables.append(
                    ExtractedTable(
                        page_number=page_index,
                        bbox=(table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3]),
                        rows=rows,
                    )
                )
    return tables
