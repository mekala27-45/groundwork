"""The CI workflow itself gets a test, the same "failure mode is silence"
reasoning the em dash gate and the claim gate already get their own tests
for. conftest.py has stated since it was first written that
".github/workflows/ci.yml starts a real Postgres service container", and
nothing anywhere checked that the file existed at all until this test was
written: the gap sat there, correct sounding prose over nothing, for every
build order step from 15 through 17. This does not re-run the workflow
(pytest cannot do that), it parses the committed YAML and asserts the
structural properties this repository actually depends on, so a future
edit that silently drops the Postgres service, or a job, or reintroduces a
missing file, fails locally at `make check` rather than only being
discovered by reading the file by eye.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"

EXPECTED_JOBS = {"lint", "typecheck", "em-dash", "test", "claims", "web"}


def _load_workflow() -> dict[str, Any]:
    return dict(yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8")))


def _resolve(value: str, workflow_env: dict[str, str]) -> str:
    """Resolves a bare `${{ env.NAME }}` reference against this workflow's
    own top level env block. Real GitHub Actions expression evaluation is
    far more general than this; this only ever needs to follow the one
    shape this file actually uses, a whole-string reference to a workflow
    level env var, not interpolate an expression embedded in a larger
    string.
    """
    prefix, suffix = "${{ env.", " }}"
    if value.startswith(prefix) and value.endswith(suffix):
        name = value[len(prefix) : -len(suffix)]
        return workflow_env[name]
    return value


def test_ci_workflow_file_exists() -> None:
    assert WORKFLOW_PATH.is_file(), (
        "conftest.py promises .github/workflows/ci.yml; it must actually exist"
    )


def test_ci_workflow_is_valid_yaml_with_the_six_expected_jobs() -> None:
    workflow = _load_workflow()
    assert set(workflow["jobs"].keys()) == EXPECTED_JOBS


def test_test_and_claims_jobs_run_against_a_real_pgvector_postgres() -> None:
    """conftest.py's own claim, checked directly: the two jobs that run
    anything marked requires_postgres or that touches groundwork_dev must
    both start a Postgres service, and that service must be a pgvector
    image, not bare postgres, since the schema's own first migration
    depends on CREATE EXTENSION vector succeeding.
    """
    workflow = _load_workflow()
    workflow_env = workflow.get("env", {})
    for job_name in ("test", "claims"):
        job = workflow["jobs"][job_name]
        postgres_service = job["services"]["postgres"]
        image = _resolve(postgres_service["image"], workflow_env)
        assert "pgvector" in image


def test_claims_job_pins_deterministic_backends_for_reproducibility() -> None:
    """The claims job reproduces committed README.md and RESULTS.md
    figures, computed in this build's own sandbox under the deterministic
    fallback backends. Leaving these on "auto" would make the gate's
    result depend on a CI runner's network reaching huggingface.co on any
    given day, exactly the flakiness a claim gate exists to rule out."""
    workflow = _load_workflow()
    claims_env = workflow["jobs"]["claims"]["env"]
    assert claims_env["GROUNDWORK_TOKENIZER_BACKEND"] == "approximate"
    assert claims_env["GROUNDWORK_EMBEDDING_BACKEND"] == "tfidf"
    assert claims_env["GROUNDWORK_RERANK_BACKEND"] == "lexical"
    assert claims_env["GROUNDWORK_FAITHFULNESS_BACKEND"] == "lexical"


def test_no_job_requires_an_llm_api_key_to_go_green() -> None:
    """This build's own hard constraint, checked structurally: design for
    a missing LLM key from the first commit. A GROUNDWORK_LLM_API_KEY
    anywhere in this workflow's env blocks would mean CI can only pass
    with a secret this portfolio repository is not meant to carry."""
    workflow = _load_workflow()
    for job in workflow["jobs"].values():
        assert "GROUNDWORK_LLM_API_KEY" not in job.get("env", {})
        for step in job.get("steps", []):
            assert "GROUNDWORK_LLM_API_KEY" not in step.get("env", {})


def test_web_job_does_not_fail_before_the_web_app_exists() -> None:
    """Build order steps 22 to 25 have not run yet, so web/package.json
    does not exist in this commit. Every step in the web job that assumes
    it does must be conditional, or CI goes red the moment this file is
    committed rather than the moment it is supposed to."""
    workflow = _load_workflow()
    web_steps = workflow["jobs"]["web"]["steps"]
    for step in web_steps:
        touches_web_app = step.get("working-directory") == "web" or "npm" in step.get("run", "")
        if touches_web_app:
            assert step.get("if"), (
                f"step {step.get('name', step)!r} touches the web app but has no "
                "condition guarding it"
            )
