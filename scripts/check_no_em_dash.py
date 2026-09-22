#!/usr/bin/env python3
"""Fail the build if an em dash (U+2014) appears anywhere it should not.

Ported from the trajectory (day 1) and cityflow (day 3) versions of this
gate, extended here for a RAG project in two ways that did not exist on
earlier days:

1. It scans the generated response templates under packages/generate, since
   those strings are the ones the running chatbot can actually emit to a
   user, not just prose we wrote by hand.
2. It scans the extractive-fallback formatting strings specifically, since
   that path is exercised even when no LLM key is configured and is
   therefore the path most likely to run in the deployed demo.

Usage:
    python scripts/check_no_em_dash.py [path ...]

With no arguments it scans the whole repository. Exits 1 and prints every
offending file and line if an em dash is found anywhere. Exits 1 if given a
path that contains no scannable files at all, so a gate that scans nothing
can never report a silent pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

EM_DASH = chr(0x2014)  # built from its code point, deliberately, so this
# file does not itself contain the literal character it exists to forbid

SCAN_SUFFIXES = {
    ".py", ".md", ".mdx", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".sql",
}

EXCLUDE_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", ".ruff_cache",
    ".mypy_cache", ".pytest_cache", ".next", "out", "htmlcov",
    "dist", "build", ".hf_cache", "models_cache",
}

# Binary or generated files we never want to scan even if their suffix
# matches, because they are allowed to carry arbitrary bytes or are not
# authored text.
EXCLUDE_FILES = {
    "uv.lock",
    "package-lock.json",
}


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix not in SCAN_SUFFIXES:
            continue
        files.append(path)
    return files


def scan(paths: list[Path]) -> dict[Path, list[int]]:
    hits: dict[Path, list[int]] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = [i + 1 for i, line in enumerate(text.splitlines()) if EM_DASH in line]
        if lines:
            hits[path] = lines
    return hits


def main(argv: list[str]) -> int:
    roots = [Path(p) for p in argv] if argv else [Path(__file__).resolve().parent.parent]
    all_files: list[Path] = []
    for root in roots:
        if root.is_file():
            all_files.append(root)
        else:
            all_files.extend(iter_files(root))

    if not all_files:
        print("check_no_em_dash: no scannable files found, refusing to report a pass", file=sys.stderr)
        return 1

    hits = scan(all_files)
    if hits:
        print(f"check_no_em_dash: em dash found in {len(hits)} file(s):", file=sys.stderr)
        for path, lines in sorted(hits.items()):
            print(f"  {path}: line(s) {', '.join(str(n) for n in lines)}", file=sys.stderr)
        return 1

    print(f"check_no_em_dash: clean, {len(all_files)} file(s) scanned, 0 em dashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
