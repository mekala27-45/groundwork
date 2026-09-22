"""The OCR fallback path: pdf2image renders a page to a raster image,
pytesseract reads text off the raster. Used only for pages whose native
text layer is too sparse to trust (see extract.py's density heuristic),
which in practice means a scanned page with no text layer at all.

ocr_page_image and ocr_image are split out as two separate, separately
mockable functions specifically so the OCR trigger test in
test_ocr_fallback.py can assert the OCR path actually ran (the "did a step
run" pattern) independent of asserting what text it returned.
"""

from __future__ import annotations

from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL.Image import Image


def is_tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def render_page_to_image(pdf_path: Path, page_number: int, dpi: int = 300) -> Image:
    """page_number is 1-indexed, matching the rest of this package."""
    images = convert_from_path(
        str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number
    )
    image: Image = images[0]
    return image


def ocr_image(image: Image) -> str:
    text: str = pytesseract.image_to_string(image)
    return text


def ocr_page(pdf_path: Path, page_number: int, dpi: int = 300) -> str:
    image = render_page_to_image(pdf_path, page_number, dpi=dpi)
    return ocr_image(image)
