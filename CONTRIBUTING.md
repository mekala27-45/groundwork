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
