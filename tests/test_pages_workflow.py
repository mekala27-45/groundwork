"""pages.yml gets the same structural test ci.yml already gets, the same
"failure mode is silence" reasoning test_ci_workflow.py's own module
docstring explains: this does not run the deploy for real (this sandbox
has no GitHub remote and no Pages site to publish to), it parses the
committed YAML and asserts the properties this project's own deploy
story actually depends on, so a future edit that silently drops the
NEXT_BASE_PATH, or points the build at production Pages permissions it
does not have, fails locally at `make check` instead of only being
discovered by a broken live site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pages.yml"


def _load_workflow() -> dict[str, Any]:
    return dict(yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8")))


def test_pages_workflow_file_exists() -> None:
    assert WORKFLOW_PATH.is_file(), (
        "next.config.ts's own comment promises .github/workflows/pages.yml sets "
        "NEXT_BASE_PATH for the real GitHub Pages build; it must actually exist"
    )


def test_pages_workflow_is_valid_yaml_with_a_build_and_deploy_job() -> None:
    workflow = _load_workflow()
    assert set(workflow["jobs"].keys()) == {"build", "deploy"}


def test_deploy_job_depends_on_build() -> None:
    """Publishing an artifact that was never actually built, type checked,
    or linted would defeat the point of having those steps at all."""
    workflow = _load_workflow()
    deploy_needs = workflow["jobs"]["deploy"]["needs"]
    assert deploy_needs == "build" or "build" in deploy_needs


def test_workflow_only_triggers_on_commits_that_touch_the_web_app() -> None:
    """ci.yml's own web job already proves every push still builds; this
    workflow spends actual GitHub Pages deploy minutes, so it should not
    fire on a commit that never touched web/ at all."""
    workflow = _load_workflow()
    # PyYAML parses the bare `on:` key as the boolean True, not the string
    # "on", since YAML 1.1 treats unquoted on/off as booleans; this workflow
    # file relies on that exact quirk being present, not a "on" string key.
    push_trigger = workflow[True]["push"]
    assert any(path.startswith("web/") for path in push_trigger["paths"])


def test_workflow_declares_the_permissions_a_real_pages_deploy_needs() -> None:
    workflow = _load_workflow()
    permissions = workflow["permissions"]
    assert permissions["pages"] == "write"
    assert permissions["id-token"] == "write"


def test_build_job_sets_the_project_page_base_path() -> None:
    """A GitHub Pages project page (mekala27-45.github.io/groundwork/) is
    served from a subpath, not the domain root; next.config.ts's own
    basePath only becomes /groundwork when this exact variable is set."""
    workflow = _load_workflow()
    build_steps = workflow["jobs"]["build"]["steps"]
    build_step = next(step for step in build_steps if step.get("name") == "Build the static export")
    assert build_step["env"]["NEXT_BASE_PATH"] == "/groundwork"


def test_no_step_requires_an_llm_api_key_to_go_green() -> None:
    """Same hard constraint test_ci_workflow.py's own equivalent test
    checks for ci.yml, applied here: this workflow needs no secret this
    portfolio repository is not meant to carry."""
    workflow = _load_workflow()
    for job in workflow["jobs"].values():
        assert "GROUNDWORK_LLM_API_KEY" not in job.get("env", {})
        for step in job.get("steps", []):
            assert "GROUNDWORK_LLM_API_KEY" not in step.get("env", {})
