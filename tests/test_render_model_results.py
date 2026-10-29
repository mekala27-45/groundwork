"""The real model page gets the three tests every gate in this project gets:
a clean pass, a deliberate violation, and a refusal to report a pass when
there is nothing behind the page. Two more pin the cases between: records
with no page fail, and no records with no page says plainly that nothing
was checked. A last test checks the committed page against the committed
records, which is the check CI runs.

No database is involved: the page is rendered from committed JSON records,
which is the point, since CI holds no model key and cannot rerun them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "render_model_results.py"


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": 1,
        "recorded_at": "2026-10-29T03:00:00+00:00",
        "commit": "0123456789abcdef",
        "workflow_run": "12345",
        "configuration": {
            "model": "nvidia_nim/example/answering-model",
            "judge_model": "nvidia_nim/other/judge-model",
            "backends": {
                "generation": "litellm",
                "quality_judge": "litellm_judge",
                "embedding": "local_model",
                "stored_vectors": ["local_model"],
                "rerank": "local_model",
                "faithfulness": "local_model",
                "tokenizer": "tiktoken",
            },
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "nli_model": "cross-encoder/nli-deberta-v3-base",
            "relevance_threshold": 0.15,
            "spend_ceiling_usd": 0.5,
        },
        "turns": {
            "total": 29,
            "written_by_model": 27,
            "extractive": 2,
            "judged": 27,
            "median_model_latency_ms": 1834.2,
            "cost_usd": 0.0,
        },
        "manifest": {
            "document_count": 6,
            "naive_chunk_count": 53,
            "structure_chunk_count": 73,
            "eval_question_count": 45,
            "eval_category_counts": "12 direct, 8 boundary",
            "turn_count": 29,
            "faithfulness_claim_count": 40,
            "faithfulness_entailed_count": 33,
            "faithfulness_contradicted_count": 2,
            "faithfulness_unsupported_count": 5,
            "out_of_scope_pass_count": 14,
            "out_of_scope_total_count": 16,
            "injection_pass_count": 10,
            "injection_total_count": 10,
            "workspace_isolation_pass_count": 3,
            "workspace_isolation_total_count": 3,
            "retrieval_metrics_table": "| config | recall@5 |\n| --- | --- |\n| naive | 0.704 |",
            "boundary_comparison_table": "| config | recall@5 |\n| --- | --- |\n| structure | 1.000 |",
            "out_of_scope_failures_table": "| case | answer |\n| --- | --- |\n| q1 | an answer |",
        },
    }
    record.update(overrides)
    return record


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def _write_records(directory: Path, *records: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        (directory / f"run-{i}.json").write_text(json.dumps(record), encoding="utf-8")


def test_clean_render_then_check_passes(tmp_path: Path) -> None:
    records, page = tmp_path / "records", tmp_path / "MODEL-RESULTS.md"
    _write_records(records, _record())

    written = _run("--write", "--records-dir", str(records), "--output", str(page))
    checked = _run("--records-dir", str(records), "--output", str(page))

    assert written.returncode == 0, written.stdout + written.stderr
    assert checked.returncode == 0, checked.stdout + checked.stderr
    text = page.read_text(encoding="utf-8")
    assert "`nvidia_nim/example/answering-model`" in text
    assert "| 14 of 16 |" in text
    assert "27 of 29" in text


def test_deliberate_violation_is_caught(tmp_path: Path) -> None:
    records, page = tmp_path / "records", tmp_path / "MODEL-RESULTS.md"
    _write_records(records, _record())
    assert _run("--write", "--records-dir", str(records), "--output", str(page)).returncode == 0

    page.write_text(
        page.read_text(encoding="utf-8").replace("14 of 16", "16 of 16"), encoding="utf-8"
    )
    checked = _run("--records-dir", str(records), "--output", str(page))

    assert checked.returncode == 1
    assert "differs from a fresh render" in checked.stdout


def test_a_page_with_no_record_behind_it_refuses_to_pass(tmp_path: Path) -> None:
    page = tmp_path / "MODEL-RESULTS.md"
    page.write_text("# Real model runs\n", encoding="utf-8")

    checked = _run("--records-dir", str(tmp_path / "records"), "--output", str(page))

    assert checked.returncode == 1
    assert "no record is committed" in checked.stdout


def test_records_with_no_page_fail(tmp_path: Path) -> None:
    records = tmp_path / "records"
    _write_records(records, _record())

    checked = _run("--records-dir", str(records), "--output", str(tmp_path / "MODEL-RESULTS.md"))

    assert checked.returncode == 1
    assert "is missing" in checked.stdout


def test_nothing_recorded_and_nothing_published_says_nothing_was_checked(tmp_path: Path) -> None:
    checked = _run(
        "--records-dir", str(tmp_path / "records"), "--output", str(tmp_path / "MODEL-RESULTS.md")
    )

    assert checked.returncode == 0
    assert "nothing to check" in checked.stdout


def test_two_runs_render_newest_first_in_their_sections(tmp_path: Path) -> None:
    records, page = tmp_path / "records", tmp_path / "MODEL-RESULTS.md"
    later = _record(recorded_at="2026-11-02T03:00:00+00:00", workflow_run="67890")
    _write_records(records, _record(), later)

    assert _run("--write", "--records-dir", str(records), "--output", str(page)).returncode == 0
    text = page.read_text(encoding="utf-8")

    assert text.index("## 2026-11-02") < text.index("## 2026-10-29")


def test_the_committed_page_matches_the_committed_records() -> None:
    """What ci.yml's claims job runs: with no record committed yet this
    reports that nothing was checked, and from the first merged run on it
    holds MODEL-RESULTS.md to the records."""
    checked = _run()

    assert checked.returncode == 0, checked.stdout + checked.stderr
