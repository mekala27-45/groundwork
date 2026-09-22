"""The em dash gate gets three tests, per the rule carried forward from
days 1 through 4: a clean pass, a deliberate violation, and a refusal to
report a pass when handed nothing to scan. The third one is the one Day 3's
note says everybody forgets, and it is the one that would have caught a
gate that silently scanned zero files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_no_em_dash.py"
EM_DASH = chr(0x2014)


def run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_tree_passes(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text("x = 1\n# a normal comment, no violation\n")
    result = run_gate(str(tmp_path))
    assert result.returncode == 0
    assert "0 em dashes" in result.stdout


def test_deliberate_violation_is_caught(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.md"
    bad_file.write_text(f"a sentence with a stray dash{EM_DASH}right here\n")
    result = run_gate(str(tmp_path))
    assert result.returncode == 1
    assert "bad.md" in result.stderr


def test_empty_scan_refuses_to_pass(tmp_path: Path) -> None:
    empty_dir = tmp_path / "nothing_here"
    empty_dir.mkdir()
    result = run_gate(str(empty_dir))
    assert result.returncode == 1
    assert "refusing to report a pass" in result.stderr


def test_gate_scans_the_real_repository_clean() -> None:
    """The end to end check: run the gate against this actual repository,
    not a synthetic tmp_path tree, so a violation anywhere in the real
    source cannot slip past CI.
    """
    result = run_gate()
    assert result.returncode == 0, result.stderr
