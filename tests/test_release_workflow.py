"""release.yml gets the same structural test treatment ci.yml and
pages.yml already get, the same "failure mode is silence" reasoning
test_ci_workflow.py's own module docstring explains: this does not run
the release for real (this sandbox has no GitHub remote and no tag has
ever actually been pushed), it parses the committed YAML and asserts the
properties this project's own release story actually depends on, so a
future edit that silently drops the verify gate, or points the release
job at a commit nobody checked, fails locally at `make check` instead of
only being discovered by a tag that produced a broken or absent release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"


def _load_workflow() -> dict[str, Any]:
    return dict(yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8")))


def _resolve(value: str, workflow_env: dict[str, str]) -> str:
    """Resolves a bare `${{ env.NAME }}` reference against this workflow's
    own top level env block, the identical helper test_ci_workflow.py
    already defines for the same reason: real GitHub Actions expression
    evaluation is far more general than this, but this only ever needs to
    follow the one shape both files actually use, a whole-string reference
    to a workflow level env var.
    """
    prefix, suffix = "${{ env.", " }}"
    if value.startswith(prefix) and value.endswith(suffix):
        name = value[len(prefix) : -len(suffix)]
        return workflow_env[name]
    return value


def test_release_workflow_file_exists() -> None:
    assert WORKFLOW_PATH.is_file(), (
        "CONTRIBUTING.md and the build order both promise a release workflow; "
        "it must actually exist"
    )


def test_release_workflow_is_valid_yaml_with_a_verify_and_release_job() -> None:
    workflow = _load_workflow()
    assert set(workflow["jobs"].keys()) == {"verify", "release"}


def test_workflow_only_triggers_on_a_version_tag() -> None:
    workflow = _load_workflow()
    # PyYAML parses the bare `on:` key as the boolean True, the same
    # documented quirk test_pages_workflow.py's own equivalent test relies
    # on, not a "on" string key.
    push_trigger = workflow[True]["push"]
    assert push_trigger["tags"] == ["v*.*.*"]
    assert "branches" not in push_trigger, (
        "a release workflow that also triggers on an ordinary branch push "
        "would try to cut a release with no version tag to name it after"
    )


def test_release_job_depends_on_verify() -> None:
    """Publishing a release built from a commit that was never actually
    linted, type checked, em dash checked, tested, or claim gated would
    defeat the entire point of running those checks anywhere at all."""
    workflow = _load_workflow()
    release_needs = workflow["jobs"]["release"]["needs"]
    assert release_needs == "verify" or "verify" in release_needs


def test_verify_job_runs_against_a_real_pgvector_postgres() -> None:
    workflow = _load_workflow()
    workflow_env = workflow.get("env", {})
    postgres_service = workflow["jobs"]["verify"]["services"]["postgres"]
    image = _resolve(postgres_service["image"], workflow_env)
    assert "pgvector" in image


def test_verify_job_runs_every_make_check_gate() -> None:
    """Mirrors CONTRIBUTING.md's own description of `make check`: ruff,
    ruff format, mypy strict, the em dash gate, the test suite, and the
    claim gate, in that order, not a subset that happens to be faster."""
    workflow = _load_workflow()
    run_commands = [step["run"] for step in workflow["jobs"]["verify"]["steps"] if "run" in step]
    joined = "\n".join(run_commands)
    assert "ruff check ." in joined
    assert "ruff format --check ." in joined
    assert "mypy packages" in joined
    assert "check_no_em_dash.py" in joined
    assert "pytest --cov" in joined
    assert "check_published_numbers.py" in joined


def test_claims_step_pins_deterministic_backends_for_reproducibility() -> None:
    """The same reasoning test_ci_workflow.py's own equivalent test
    applies to ci.yml's claims job: a release must reproduce the
    committed README.md and RESULTS.md figures under the exact
    deterministic fallbacks they were actually rendered with, not
    whatever a runner's network happens to reach that day."""
    workflow = _load_workflow()
    claims_step = next(
        step
        for step in workflow["jobs"]["verify"]["steps"]
        if "check_published_numbers.py" in step.get("run", "")
    )
    claims_env = claims_step["env"]
    assert claims_env["GROUNDWORK_TOKENIZER_BACKEND"] == "approximate"
    assert claims_env["GROUNDWORK_EMBEDDING_BACKEND"] == "tfidf"
    assert claims_env["GROUNDWORK_RERANK_BACKEND"] == "lexical"
    assert claims_env["GROUNDWORK_FAITHFULNESS_BACKEND"] == "lexical"


def test_release_job_creates_a_real_github_release_from_the_pushed_tag() -> None:
    workflow = _load_workflow()
    release_steps = workflow["jobs"]["release"]["steps"]
    release_step = next(
        step for step in release_steps if "gh release create" in step.get("run", "")
    )
    assert "github.ref_name" in release_step["run"]
    assert "--generate-notes" in release_step["run"]


def test_release_job_needs_no_third_party_action_or_extra_secret() -> None:
    """No docker/login-action, no softprops/action-gh-release, no PAT: gh
    release create is preinstalled and GITHUB_TOKEN is enough, the same
    "no secret required for any job to go green" rule ci.yml's own header
    states. Checked directly rather than only claimed in a comment, since
    a comment saying this is exactly the kind of thing this project's own
    claim gate exists to stop anyone from trusting on faith."""
    workflow = _load_workflow()
    release_steps = workflow["jobs"]["release"]["steps"]
    for step in release_steps:
        uses = step.get("uses", "")
        assert "docker/" not in uses
        assert "action-gh-release" not in uses
    release_env = next(step["env"] for step in release_steps if "env" in step)
    assert release_env["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"


def test_no_step_requires_an_llm_api_key_to_go_green() -> None:
    workflow = _load_workflow()
    for job in workflow["jobs"].values():
        assert "GROUNDWORK_LLM_API_KEY" not in job.get("env", {})
        for step in job.get("steps", []):
            assert "GROUNDWORK_LLM_API_KEY" not in step.get("env", {})


def test_release_workflow_does_not_publish_a_container_image() -> None:
    """A deliberate scope decision, not an oversight: see this file's own
    header comment. Checked structurally so a future edit cannot silently
    add back an unverifiable GHCR push without at least breaking this
    test first."""
    workflow = _load_workflow()
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            run = step.get("run", "")
            assert "ghcr.io" not in uses
            assert "ghcr.io" not in run
            assert "docker build" not in run
            assert "docker push" not in run
