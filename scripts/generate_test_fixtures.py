"""Regenerates the fixture PDFs evalset/questions.yaml's boundary spanning
and injection categories point at, from the same builder functions
packages/ingest/tests already exercises directly.

fixtures.py's own docstring names this script as the place those builders
get turned into committed files under evalset/, for the same reason every
other document in evalset/ is generated from source rather than hand
edited once: a reader (or CI) can regenerate every fixture from code and
confirm the committed binary matches, rather than trusting an opaque PDF
no one can diff.

evalset/page_boundary_test.pdf is a second copy of the same document
packages/ingest/tests/test_page_boundary.py builds inline for its own
assertions; committing one here as well is what lets the eval set's
boundary spanning questions and the web app's demo material reference a
stable, named file path instead of a pytest tmp_path that stops existing
the moment the test process exits.

evalset/injection_test.pdf backs the eval set's injection category and
the packages/verify injection defense test still to come; both assert
against groundwork_ingest.fixtures.INJECTION_TEST_MARKER rather than a
literal typed twice.

Run with: uv run python scripts/generate_test_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from groundwork_ingest.fixtures import (
    INJECTION_TEST_MARKER,
    build_injection_test_pdf,
    build_page_boundary_test_pdf,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "evalset"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_page_boundary_test_pdf(OUT_DIR / "page_boundary_test.pdf")
    build_injection_test_pdf(OUT_DIR / "injection_test.pdf", marker=INJECTION_TEST_MARKER)
    for pdf in sorted(OUT_DIR.glob("*_test.pdf")):
        print(pdf.relative_to(OUT_DIR.parent))


if __name__ == "__main__":
    main()
