"""The published number gate gets the same three tests every gate in this
project gets, per the rule carried forward from days 1 through 4: a clean
pass, a deliberate violation, and a refusal to report a pass when handed
nothing to render from.

Unlike the em dash gate, this one always needs a real Postgres (every
render calls build_manifest(), which queries it), so every test here is
requires_postgres. --output-dir lets the clean pass and deliberate
violation tests write into a throwaway tmp_path instead of the real
committed README.md and RESULTS.md; see check_published_numbers.py's own
module docstring for why that flag exists.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_published_numbers.py"

pytestmark = pytest.mark.requires_postgres


def run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )


def test_clean_render_then_check_passes(tmp_path: Path) -> None:
    write_result = run_gate("--write", "--output-dir", str(tmp_path))
    assert write_result.returncode == 0, write_result.stderr
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "RESULTS.md").exists()

    check_result = run_gate("--output-dir", str(tmp_path))
    assert check_result.returncode == 0, check_result.stderr
    assert "match the stored records exactly" in check_result.stdout


def test_deliberate_violation_is_caught(tmp_path: Path) -> None:
    write_result = run_gate("--write", "--output-dir", str(tmp_path))
    assert write_result.returncode == 0, write_result.stderr

    results_path = tmp_path / "RESULTS.md"
    results_path.write_text(
        results_path.read_text(encoding="utf-8") + "\nThis line was edited in by hand.\n",
        encoding="utf-8",
    )

    check_result = run_gate("--output-dir", str(tmp_path))
    assert check_result.returncode == 1
    assert "RESULTS.md" in check_result.stderr
    assert "do not match a fresh render" in check_result.stderr


def test_missing_template_refuses_to_pass() -> None:
    """render_all()'s own defense against silently rendering nothing: a
    template path that does not exist fails loudly rather than producing
    an empty or partial document. Exercised in process, not via
    subprocess, since it needs to hand render_all() a templates mapping
    the real CLI never takes as an argument.
    """
    sys.path.insert(0, str(REPO_ROOT / "packages" / "api" / "src"))
    sys.path.insert(0, str(REPO_ROOT / "packages" / "core" / "src"))
    sys.path.insert(0, str(SCRIPT.parent))
    import asyncio

    import check_published_numbers

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(
            check_published_numbers.render_all(
                templates={"README.md": Path("/nonexistent/does_not_exist.tmpl")}
            )
        )
    assert excinfo.value.code == 1


def test_gate_matches_the_real_committed_files() -> None:
    """The end to end check: the actual committed README.md and
    RESULTS.md, re-rendered from the current database right now, so a
    hand edited number anywhere in either file cannot slip past CI.
    """
    result = run_gate()
    assert result.returncode == 0, result.stderr
