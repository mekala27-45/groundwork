# Runbook: deploying and operating groundwork for real

This repository was built entirely in a cloud sandbox with no GitHub
remote, no Neon project, and no Fly.io account reachable from it (see
docs/security.md's own Deployment section for exactly what that means and
does not mean). Every command below is the real sequence to take this
repository from a local clone to a live deployment; none of it has been
run end to end in the environment that built this project, stated plainly
rather than presented as already proven. Read docs/security.md's
Deployment section before running any of this against anything that
matters.

## Prerequisites

A GitHub account, a Neon account (free, no card required as of this
build), a Fly.io account (a card on file is required as of this build;
see RESULTS.md's Limitations section for the finding that Fly's own
always-free compute tier no longer exists), and locally: `uv`, `npm`,
the `fly` CLI, and `git`.

## 1. Push this repository to GitHub

```
git remote add origin git@github.com:<your-account>/groundwork.git
git push -u origin master
```

`.github/workflows/ci.yml` starts running on the first push; expect it
green, since every job here was verified locally under the identical
Postgres image and deterministic backend pins CI itself uses.

## 2. Create the database on Neon

Create a new Neon project, then enable pgvector once per database (the
extension installs per database, not per project, and needs no add-on or
paid tier on any Neon plan):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Copy the connection string Neon gives you; it is `GROUNDWORK_DATABASE_URL`
below. Use the `postgresql+psycopg://` scheme groundwork's own SQLAlchemy
setup expects, not the bare `postgresql://` Neon's dashboard may show by
default.

## 3. Migrate and seed the real database

From a machine with real network access (unlike this sandbox):

```
GROUNDWORK_DATABASE_URL="<neon connection string>" \
  uv run --directory packages/api alembic upgrade head
GROUNDWORK_DATABASE_URL="<neon connection string>" \
  uv run python scripts/seed_demo_workspaces.py
GROUNDWORK_DATABASE_URL="<neon connection string>" \
  uv run python scripts/build_eval_questions.py
GROUNDWORK_DATABASE_URL="<neon connection string>" \
  uv run python scripts/run_eval.py
```

This is the point where a real embedding model and, if
`GROUNDWORK_LLM_API_KEY` is set, a real generation model actually run for
the first time: `huggingface.co` is reachable from a normal machine, so
the "auto" backend resolves to the local model path everywhere this
sandbox's own numbers used the deterministic fallback instead. Re-run
`scripts/check_published_numbers.py --write` afterward and commit the
result if you want README.md and RESULTS.md to reflect the real-model
numbers rather than this sandbox's fallback-backend ones; both are
honest, since both are computed the same way from real stored rows, just
under different reachable backends.

## 4. Deploy the API to Fly

```
fly launch --dockerfile Dockerfile --config fly.toml --no-deploy
```

`--no-deploy` first, deliberately: `fly launch` on a fresh account
creates the app and assigns it its real, globally unique hostname, which
is the value `fly.toml`'s `app` field has stood in as a placeholder for
throughout this build. Set the real secrets before the first real deploy:

```
fly secrets set \
  GROUNDWORK_DATABASE_URL="<neon connection string>" \
  GROUNDWORK_LLM_API_KEY="<optional>" \
  GROUNDWORK_JUDGE_MODEL="<optional>" \
  GROUNDWORK_ADMIN_TOKEN="<a real random secret>"
```

Never set these in `fly.toml`'s own `[env]` block, which this repository
commits; `.env.example` documents every one of them and none belongs in
a committed file. Then deploy for real:

```
fly deploy
```

`fly.toml`'s `release_command` runs `alembic upgrade head` once, in its
own machine, before traffic reaches the new release; a failed migration
fails the deploy rather than leaving a half-migrated database serving
requests. Confirm the API is actually up:

```
curl https://<your-app>.fly.dev/healthz
```

## 5. Point the web app at the real API, and deploy it

If the app name Fly assigned is not `groundwork-api` (the placeholder
`fly.toml` ships with), set this repository's own Actions variable so
`pages.yml` builds against the real host instead of the placeholder's
expected one: repository Settings, Secrets and variables, Actions,
Variables tab, add `NEXT_PUBLIC_API_BASE_URL` as
`https://<your-app>.fly.dev`.

Update `fly.toml`'s own `GROUNDWORK_CORS_ORIGINS` if your GitHub Pages
origin differs from `https://mekala27-45.github.io` (a fork, a custom
domain, a different account), and redeploy the API so the new value takes
effect; CORS is enforced server side, so the web app cannot work around a
mismatched origin from the browser.

Enable Pages itself, a one-time repository setting `pages.yml` does not
turn on by itself: repository Settings, Pages, under "Build and
deployment" set Source to "GitHub Actions." Then push any change under
`web/` (or run `pages.yml` manually from the Actions tab,
`workflow_dispatch`) to trigger the first real deploy. The site is served
at `https://<your-account>.github.io/groundwork/`.

## 6. Verify the real deployment

Open the deployed site's `/chat` page and ask a real question against
one of the seeded workspaces; open the browser console and confirm no
CORS error. Open `/eval` and confirm the dashboard renders real numbers,
not an empty state. Open `/trace` for a turn id from step 3's real eval
run and confirm every section renders. None of this is optional
sign-off theater: it is the actual first time this specific deployment
path will have been exercised end to end, per docs/security.md's own
disclosure.

## Ongoing operations

**Logs**: `fly logs` for the API; `groundwork_api.logging`'s structured
logger redacts workspace names to a content hash (docs/security.md), so a
log line is safe to paste into an issue without disclosing tenant data.

**Redeploying after a code change**: `fly deploy` for the API (the
release command re-runs migrations, a no-op if none are pending); a push
touching `web/` for the site, automatic via `pages.yml`.

**Adding a migration**: `cd packages/api && uv run alembic revision
--autogenerate -m "..."`, review the generated file by hand (autogenerate
misses some pgvector-specific DDL), then commit it. It runs automatically
on the next `fly deploy`.

**Re-seeding demo content or re-running the eval harness against
production**: the same three commands from step 3, run again. Both are
idempotent by design:
`scripts/seed_demo_workspaces.py`'s reset helper truncates its own three
named workspaces first, and `scripts/build_eval_questions.py` re-resolves
every question's expected chunk ids fresh rather than trusting stale
ones. Follow with `check_published_numbers.py --write` and commit if the
published figures should move.

**Rolling back a bad deploy**: Fly has no dedicated rollback command by
design; `fly releases --image` lists prior releases with their image
references, and `fly deploy --image registry.fly.io/<app>:<tag>` redeploys
one directly, the same deploy mechanism as any other release. A rolled
back release does not automatically roll back a migration that already
ran; check whether the bad release shipped a migration before assuming a
straight rollback is safe.

**Fly billing**: as of this build, Fly.io's compute has no always-free
tier, only a trial period followed by usage-based billing (RESULTS.md's
Limitations section). `auto_stop_machines` and `min_machines_running = 0`
in `fly.toml` scale the API to zero between requests, the closest this
configuration gets to the portfolio series' original free-tier-only
constraint; monitor actual usage against Fly's current pricing before
assuming the cost stays negligible.

## Before you deploy this with anything that matters

Read docs/security.md's Deployment section in full first. In short: pin
the Docker base image to a resolved digest once you have real registry
access to verify one against, run `docker compose up` at least once
locally to confirm the compose file actually works rather than trusting
it built correctly, and treat the admin token as the only access control
this system has, since it is.
