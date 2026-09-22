#!/usr/bin/env python3
"""The claim gate, ported as the day 3 renderer rather than the day 1 parser.

README.md and RESULTS.md are not written by hand and then checked for
numbers that look plausible. They are rendered from templates in
docs/templates/ against a manifest built by querying stored eval and test
records (packages/api/src/groundwork_api/claims.py: build_manifest), and
this script re-renders both files from the current manifest and diffs the
result byte for byte against what is committed.

A number cannot be edited into README.md or RESULTS.md by hand: the moment
it stops matching the template's re-render, this gate fails the build. This
is strictly stronger than scanning prose for digits near a keyword, because
it produces an exact diff instead of a heuristic guess, and it makes "every
figure came from a real query today" enforceable rather than aspirational.

Usage:
    python scripts/check_published_numbers.py            # check, exit 1 on drift
    python scripts/check_published_numbers.py --write     # render and write in place
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "api" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "core" / "src"))

TEMPLATES = {
    "README.md": REPO_ROOT / "docs" / "templates" / "README.md.tmpl",
    "RESULTS.md": REPO_ROOT / "docs" / "templates" / "RESULTS.md.tmpl",
}


def render_all() -> dict[str, str]:
    from groundwork_api.claims import build_manifest, render_template

    manifest = build_manifest()
    rendered = {}
    for name, template_path in TEMPLATES.items():
        if not template_path.exists():
            print(f"check_published_numbers: missing template {template_path}", file=sys.stderr)
            raise SystemExit(1)
        rendered[name] = render_template(template_path.read_text(encoding="utf-8"), manifest)
    return rendered


def main(argv: list[str]) -> int:
    write = "--write" in argv
    rendered = render_all()

    if not rendered:
        print(
            "check_published_numbers: rendered nothing, refusing to report a pass", file=sys.stderr
        )
        return 1

    if write:
        for name, content in rendered.items():
            (REPO_ROOT / name).write_text(content, encoding="utf-8")
            print(f"check_published_numbers: wrote {name}")
        return 0

    drift = []
    for name, content in rendered.items():
        target = REPO_ROOT / name
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != content:
            drift.append(name)

    if drift:
        print(
            f"check_published_numbers: {', '.join(drift)} do not match a fresh render "
            "of the stored records. Run with --write to regenerate, then review the diff.",
            file=sys.stderr,
        )
        return 1

    print(f"check_published_numbers: {len(rendered)} document(s) match the stored records exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
