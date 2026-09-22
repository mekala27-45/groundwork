"""A small page-flowing helper for authoring the demo content PDFs
(evalset/meridian/*.pdf) directly from source text, the same "every fixture
is regenerated from code, never a hand-edited opaque binary" principle
packages/ingest/src/groundwork_ingest/fixtures.py already applies to the
synthetic test PDFs.

PyMuPDF's own HTML flowing engine (fitz.Story) was tried first and
rejected: its text shaper applies standard typographic ligature
substitution ("fl" becomes a single glyph U+FB02), which round-trips back
out of get_text() as a different character than what was typed, silently
corrupting anything downstream that matches on the literal word ("workflow"
extracts as "workﬂow"). Every real font tested reproduced this; only a
monospace font avoided it, which was not an acceptable look for content a
hiring manager will actually read. Page.insert_textbox with PyMuPDF's
Base14 fonts uses a simpler glyph path with no shaping step, so this
builder does its own layout instead: one textbox per block, and a manual
page break the moment a block does not fit, rather than the library
choosing where a page breaks for it.

Header font sizes are chosen against groundwork_chunk.segments.py's real
thresholds (HEADER_SIZE_RATIO=1.25 against the document's median size, or
HEADER_BOLD_SIZE_RATIO=1.05 for a bold span), not picked for looks, so the
documents this builds give structure aware chunking genuine headers to
split on rather than a cosmetic approximation of one.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz  # PyMuPDF; "fitz" is the deprecated import alias

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
MARGIN = 72.0

BODY_SIZE = 11.0
H1_SIZE = 17.0
H2_SIZE = 13.0
ROW_HEIGHT = 20.0
COLUMN_GAP = 20.0


def _wrapped_line_count(text: str, width: float, fontsize: float, fontname: str) -> int:
    """How many lines `text` wraps to at `fontsize` within `width`, using
    PyMuPDF's own glyph width table rather than a characters-per-line
    guess, which is exactly wrong across a font's actual mix of narrow and
    wide characters.

    insert_textbox wraps at word boundaries only, never mid-word, so a
    single word wider than `width` cannot be rendered at any row height;
    that is a content or column-width problem, not a height problem, and
    is raised here immediately with the word named, rather than left to
    surface several calls later as a table() cell that mysteriously does
    not fit its own measured height.
    """
    words = text.split()
    if not words:
        return 1
    for word in words:
        if fitz.get_text_length(word, fontname=fontname, fontsize=fontsize) > width:
            raise ValueError(
                f"word {word!r} alone is wider than the {width}pt column at fontsize "
                f"{fontsize}; widen the column or shorten the text"
            )
    lines = 1
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if fitz.get_text_length(candidate, fontname=fontname, fontsize=fontsize) <= width:
            current = candidate
        else:
            lines += 1
            current = word
    return lines


class DocBuilder:
    """Lays out h1/h2/paragraph/table blocks top to bottom on Letter sized
    pages, starting a new page whenever the next block will not fit in the
    space remaining on the current one. insert_textbox renders nothing at
    all when a block does not fit its rect (a negative return, not a
    partial one), which is what makes "retry on a fresh page" a safe,
    complete strategy rather than one that risks a silently truncated
    paragraph.
    """

    def __init__(self) -> None:
        self.doc = fitz.open()
        self.page: fitz.Page = self.doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.cursor_y = MARGIN

    def _remaining_rect(self) -> fitz.Rect:
        return fitz.Rect(MARGIN, self.cursor_y, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)

    def _new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.cursor_y = MARGIN

    def _place(self, text: str, *, fontsize: float, fontname: str, gap_after: float) -> None:
        rect = self._remaining_rect()
        # insert_textbox raises instead of returning "doesn't fit" when the
        # rect itself has no usable height left, so that case is checked
        # before ever calling it, not caught from its return value.
        if rect.height < fontsize * 2:
            self._new_page()
            rect = self._remaining_rect()
        spare = self.page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname)
        if spare < 0:
            self._new_page()
            rect = self._remaining_rect()
            spare = self.page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname)
            if spare < 0:
                raise ValueError(f"block does not fit even on a fresh page: {text[:60]!r}")
        self.cursor_y += (rect.height - spare) + gap_after

    def h1(self, text: str) -> None:
        self._place(text, fontsize=H1_SIZE, fontname="hebo", gap_after=14.0)

    def h2(self, text: str) -> None:
        self._place(text, fontsize=H2_SIZE, fontname="hebo", gap_after=10.0)

    def paragraph(self, text: str) -> None:
        self._place(text, fontsize=BODY_SIZE, fontname="helv", gap_after=12.0)

    def table(self, rows: list[list[str]], *, col_widths: list[float], header: bool = True) -> None:
        """Draws a bordered grid, one call per row, with the row height
        measured from each cell's actual wrapped line count rather than a
        fixed guess. A fixed row height silently dropped any cell whose
        text needed more than one line, since insert_textbox renders
        nothing at all for a cell it cannot fit (see DocBuilder's own
        docstring): real glyph width measurement, not characters-per-line
        arithmetic, is what catches that before it ships as a blank cell
        in committed demo content.
        """
        fontsize = 9.5
        row_heights = [
            self._table_row_height(row, col_widths, is_header=header and r == 0, fontsize=fontsize)
            for r, row in enumerate(rows)
        ]

        needed = sum(row_heights)
        if self.cursor_y + needed > PAGE_HEIGHT - MARGIN and self.cursor_y > MARGIN + 1:
            self._new_page()
        for r, row in enumerate(rows):
            row_height = row_heights[r]
            if self.cursor_y + row_height > PAGE_HEIGHT - MARGIN:
                self._new_page()
            y0 = self.cursor_y
            y1 = y0 + row_height
            x = MARGIN
            is_header_row = header and r == 0
            fontname = "hebo" if is_header_row else "helv"
            for c, cell in enumerate(row):
                w = col_widths[c]
                rect = fitz.Rect(x, y0, x + w, y1)
                self.page.draw_rect(rect, color=(0, 0, 0), width=0.75)
                spare = self.page.insert_textbox(
                    fitz.Rect(x + 4, y0 + 2, x + w - 2, y1 - 2),
                    cell,
                    fontsize=fontsize,
                    fontname=fontname,
                )
                if spare < 0:
                    raise ValueError(
                        f"table cell does not fit its measured row height: {cell!r} in column "
                        f"width {w}"
                    )
                x += w
            self.cursor_y = y1
        self.cursor_y += 12.0

    def _table_row_height(
        self, row: list[str], col_widths: list[float], *, is_header: bool, fontsize: float
    ) -> float:
        fontname = "hebo" if is_header else "helv"
        # insert_textbox needs roughly 1.68x the font size in vertical
        # space per line (measured empirically, not documented); 1.8x
        # leaves a safety margin rather than sitting right at that edge.
        line_height = fontsize * 1.8
        max_lines = 1
        for cell, w in zip(row, col_widths, strict=True):
            usable_width = w - 6.0
            max_lines = max(max_lines, _wrapped_line_count(cell, usable_width, fontsize, fontname))
        return max_lines * line_height + 6.0

    def two_column_terms(self, entries: list[tuple[str, str]]) -> None:
        """Lays (term, definition) pairs into two newspaper style columns,
        filling the left column before the right, and starting a fresh
        page whenever both columns on the current one are full. This is
        the document's one genuinely multi column section: real column
        geometry for extraction to encounter, not narrower paragraphs
        dressed up to look like columns.
        """
        col_width = (PAGE_WIDTH - 2 * MARGIN - COLUMN_GAP) / 2
        # Starting at the top of a fresh page keeps the column bookkeeping
        # below from having to reason about a partially used first page.
        if self.cursor_y > MARGIN + 1:
            self._new_page()
        columns_x = [MARGIN, MARGIN + col_width + COLUMN_GAP]
        col_index = 0
        y = MARGIN

        def advance_column() -> tuple[int, float]:
            if col_index == 0:
                return 1, MARGIN
            self._new_page()
            return 0, MARGIN

        for term, definition in entries:
            text = f"{term}. {definition}"
            rect = fitz.Rect(
                columns_x[col_index], y, columns_x[col_index] + col_width, PAGE_HEIGHT - MARGIN
            )
            if rect.height < BODY_SIZE * 2:
                col_index, y = advance_column()
                rect = fitz.Rect(
                    columns_x[col_index], y, columns_x[col_index] + col_width, PAGE_HEIGHT - MARGIN
                )
            spare = self.page.insert_textbox(rect, text, fontsize=BODY_SIZE, fontname="helv")
            if spare < 0:
                col_index, y = advance_column()
                rect = fitz.Rect(
                    columns_x[col_index], y, columns_x[col_index] + col_width, PAGE_HEIGHT - MARGIN
                )
                spare = self.page.insert_textbox(rect, text, fontsize=BODY_SIZE, fontname="helv")
                if spare < 0:
                    raise ValueError(f"glossary entry does not fit even at a column top: {term!r}")
            y = rect.y0 + (rect.height - spare) + 10.0

        self.cursor_y = PAGE_HEIGHT  # force whatever comes next onto a fresh page

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        self.doc.close()
