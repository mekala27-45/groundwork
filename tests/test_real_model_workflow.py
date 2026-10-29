"""Structural tests for .github/workflows/real-model.yml, the same kind of
check tests/test_ci_workflow.py runs for ci.yml: GitHub Actions cannot be
run from pytest, so this parses the committed YAML and asserts the
properties the run depends on. It is the only workflow that holds a model
key, which is why most of what is checked here is what it must not do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "real-model.yml"


def _load() -> dict[str, Any]:
    return dict(yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8")))


def _on(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML reads the bare key `on` as the boolean True.
    return dict(workflow.get("on") or workflow[True])


def _steps() -> list[dict[str, Any]]:
    return list(_load()["jobs"]["run"]["steps"])


def test_it_runs_only_when_started_by_hand() -> None:
    assert list(_on(_load())) == ["workflow_dispatch"]


def test_the_key_comes_from_the_secret_and_nowhere_else() -> None:
    env = _load()["jobs"]["run"]["env"]
    assert env["GROUNDWORK_LLM_API_KEY"] == "${{ secrets.NVIDIA_API_KEY }}"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert text.count("secrets.") == 1


def test_it_stops_first_when_the_key_is_missing() -> None:
    first = _steps()[0]
    assert "GROUNDWORK_LLM_API_KEY" in first["run"] and "exit 1" in first["run"]


def test_the_judge_defaults_to_another_family_than_the_answering_model() -> None:
    inputs = _on(_load())["workflow_dispatch"]["inputs"]
    answering = inputs["model"]["default"].split("/")[1]
    judging = inputs["judge_model"]["default"].split("/")[1]
    assert answering != judging


def test_no_backend_is_pinned_so_the_real_models_load() -> None:
    env = _load()["jobs"]["run"]["env"]
    for name in (
        "GROUNDWORK_EMBEDDING_BACKEND",
        "GROUNDWORK_RERANK_BACKEND",
        "GROUNDWORK_FAITHFULNESS_BACKEND",
        "GROUNDWORK_TOKENIZER_BACKEND",
    ):
        assert name not in env


def test_it_seeds_builds_evaluates_records_and_renders_in_that_order() -> None:
    runs = [step.get("run", "") for step in _steps()]

    def index(fragment: str) -> int:
        return next(i for i, run in enumerate(runs) if fragment in run)

    order = [
        index("alembic upgrade head"),
        index("scripts/seed_demo_workspaces.py"),
        index("scripts/build_eval_questions.py"),
        index("scripts/run_eval.py"),
        index("scripts/record_model_run.py"),
        index("scripts/render_model_results.py --write"),
        index("git push"),
    ]
    assert order == sorted(order)


def test_it_pushes_a_review_branch_and_never_the_default_branch() -> None:
    push = next(step["run"] for step in _steps() if "git push" in step.get("run", ""))
    assert 'git push origin "HEAD:refs/heads/$branch"' in push
    assert 'branch="real-model-$RUN_ID"' in push
    assert "master" not in push and "--force" not in push


def test_the_model_name_reaches_the_shell_through_env_not_expression() -> None:
    push = next(step for step in _steps() if "git push" in step.get("run", ""))
    assert "${{" not in push["run"]
    assert push["env"]["MODEL"] == "${{ inputs.model }}"


def test_it_runs_against_a_real_pgvector_postgres() -> None:
    image = _load()["jobs"]["run"]["services"]["postgres"]["image"]
    assert "pgvector" in image
