"""scripts/record_model_run.py's own decisions, without a database: how it
counts a run's turns, that it refuses a run no model wrote any turn of, and
how it names the file. Collecting a real run needs the seeded database and
a key, which is what .github/workflows/real-model.yml provides.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "record_model_run", REPO_ROOT / "scripts" / "record_model_run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _turn(
    *, extractive: bool, judged: bool = False, latency: float = 0.0, cost: float = 0.0
) -> dict[str, object]:
    return {
        "extractive_fallback": extractive,
        "judge_scores": {"clarity": 4} if judged else None,
        "latency_ms": latency,
        "cost_usd": cost,
    }


def test_turns_are_counted_by_who_wrote_them() -> None:
    m = _module()

    summary = m.summarize_turns(
        [
            _turn(extractive=False, judged=True, latency=1000.0),
            _turn(extractive=False, judged=True, latency=3000.0),
            _turn(extractive=True),
        ]
    )

    assert summary == {
        "total": 3,
        "written_by_model": 2,
        "extractive": 1,
        "judged": 2,
        "median_model_latency_ms": 2000.0,
        "cost_usd": 0.0,
    }


def test_a_run_no_model_wrote_is_refused() -> None:
    m = _module()
    turns = m.summarize_turns([_turn(extractive=True), _turn(extractive=True)])

    with pytest.raises(m.NoModelTurnsError, match="none of the 2 turns"):
        m.build_record(manifest={}, turns=turns, configuration={"model": "x/y"}, provenance={})


def test_a_run_with_model_turns_is_recorded_with_its_schema() -> None:
    m = _module()
    turns = m.summarize_turns([_turn(extractive=False, latency=10.0)])

    record = m.build_record(
        manifest={"turn_count": 1},
        turns=turns,
        configuration={"model": "nvidia_nim/a/b"},
        provenance={
            "recorded_at": "2026-10-29T00:00:00+00:00",
            "commit": None,
            "workflow_run": None,
        },
    )

    assert record["schema"] == m.SCHEMA_VERSION
    assert record["manifest"] == {"turn_count": 1}
    assert record["turns"]["written_by_model"] == 1


def test_the_file_is_named_by_date_model_and_run() -> None:
    m = _module()

    name = m.record_name(
        "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b", "2026-10-29T03:00:00+00:00", "987"
    )

    assert name == "2026-10-29-nemotron-3-ultra-550b-a55b-987.json"
    assert (
        m.record_name("gpt-4o-mini", "2026-10-29T03:00:00+00:00", None)
        == "2026-10-29-gpt-4o-mini-local.json"
    )
