# Contributing to groundwork

This started as a one day solo build and the workflow below is the one
that built it, kept here because it is the workflow that keeps the
project's central claim true: every number in README.md and RESULTS.md
was computed by a query against stored records, not typed by hand.

## Setup

```
uv sync --all-packages --dev
cd web && npm install && cd ..
make db-up
cd packages/api && uv run alembic upgrade head && cd ../..
```

Postgres needs the pgvector extension once per database:

```
sudo -u postgres psql -d groundwork_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Before every commit

```
make check
```

This runs, in order: ruff, mypy strict, the em dash gate, the full test
suite with the coverage floor enforced, and the claim gate. All five have
to pass. The claim gate specifically re-renders README.md and RESULTS.md
from `docs/templates/*.tmpl` against a manifest built by querying stored
eval and test records, then diffs the result against what is committed. If
you changed a number by hand instead of by running the pipeline that
produced it, this is the step that catches it, and it is supposed to.

## Adding a retrieval or faithfulness backend

Every model-backed component (embeddings, reranking, faithfulness scoring)
follows the same shape: a small `Protocol` in `packages/retrieve` or
`packages/verify`, a `local_model` implementation that calls the real
model, a deterministic zero-dependency fallback, and a `resolve_backend()`
function that probes once and picks one. Look at
`packages/retrieve/src/groundwork_retrieve/embeddings.py` before adding a
new one; the shape is meant to be copied, not reinvented per component.

## A real model run

README.md and RESULTS.md publish the offline fallback run, the one CI
reproduces with no key. A run with real models is published separately,
in MODEL-RESULTS.md, and never by hand:

1. Add the key once: repository settings, Secrets and variables, Actions,
   a new repository secret named `NVIDIA_API_KEY` (a key for NVIDIA's free
   build tier at build.nvidia.com; no card).
2. Actions, `real model`, Run workflow. The defaults are a free answering
   model and a judge from another family; either can be changed there.
3. The workflow seeds a fresh database, runs the usual three steps with
   every backend on `auto` (the runner reaches huggingface.co, so the real
   embedding, reranking and faithfulness models load), writes the run to
   `results/real-model/` with `scripts/record_model_run.py`, renders
   MODEL-RESULTS.md with `scripts/render_model_results.py --write`, and
   pushes both to a branch named `real-model-<run id>`.
4. Read that branch and merge it. From then on the claims job holds
   MODEL-RESULTS.md to the committed records on every push.

`scripts/record_model_run.py` refuses a run in which no turn was written by
the model, so a key that was set but never used cannot publish extractive
numbers under a model's name.

## Tests

- A new gate (anything that can pass or fail a build) needs three tests:
  a clean pass, a deliberate violation, and a refusal to report a pass on
  empty input. See `tests/test_check_no_em_dash.py` for the shape.
- A step whose failure mode is silence (something either ran or it did
  not, independent of whether its output looks right) needs a test that
  asks whether it ran, not just whether the result looks correct.
- Anything that shells out to tesseract, or that needs a locally
  downloaded model, gets an availability probe and a named
  `pytest.mark.skip` reason, never a bare skip. CI asserts the dependency
  is present rather than silently accepting the skip.

## Commit style

Conventional commits (`feat(scope): ...`, `fix(scope): ...`, `test(scope):
...`, `docs: ...`, `ci: ...`, `chore: ...`). One logical change per commit.

## Code of conduct

Be direct and be kind. Disagree on the technical merits, assume good
faith, and leave the codebase clearer than you found it.
