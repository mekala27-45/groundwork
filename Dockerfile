# syntax=docker/dockerfile:1
#
# Multi stage build for groundwork_api. One WORKDIR, /app, reused
# identically in the builder stage and the runtime stage, on purpose:
# `uv sync` installs this workspace's own packages editable, which bakes
# the absolute source path from sync time into .venv's .pth files. A
# prior project in this series lost an entire deploy to exactly this,
# discovered only once it actually built somewhere else: the builder used
# one working directory, the runtime stage another, the recorded path did
# not exist in the final image, and every container exited on
# ModuleNotFoundError before uvicorn was ever reached. Reusing the same
# path in both stages, and copying /app to /app rather than to a
# differently named directory, keeps the recorded path valid. The build
# time import checks below exist specifically to catch a regression of
# that exact failure with a readable build error, not a silent, healthy
# looking image that cannot serve a single request.
#
# Base image pinned to an exact patch tag rather than a floating
# "3.12-slim", the same two tier pinning reasoning this repo's own CI
# workflow documents for GitHub Actions. Pinning to a resolved digest
# would be stronger still, and is the right next step at actual deploy
# time: this sandbox's network policy returns 403 for every container
# registry it was tried against (Docker Hub and GHCR both), so a digest
# noted here could not be verified against a real pull and would be a
# guess dressed up as a measurement. A tag this sandbox cannot silently
# get wrong is the honest choice until a real `docker pull` can confirm
# one.

FROM python:3.12.7-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the exact version this repo's own uv.lock was generated with
# in this sandbox (`uv --version`), not a guess.
RUN pip install --no-cache-dir --root-user-action=ignore "uv==0.8.17"

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages packages

# --locked: fail the build rather than silently drift from the committed
# lockfile, the same discipline ci.yml enforces for every job. No --dev:
# ruff, mypy, pytest and the rest of the dev dependency group have no
# reason to exist in a production image, and leaving them out is the
# default here, not a flag added to strip them back out.
RUN uv sync --locked --all-packages

RUN /app/.venv/bin/python -c "\
import groundwork_api, groundwork_core, groundwork_ingest, groundwork_chunk, \
    groundwork_retrieve, groundwork_generate, groundwork_verify" \
    && echo "builder stage import check passed"

FROM python:3.12.7-slim-bookworm AS runtime

# tesseract-ocr backs pytesseract, poppler-utils backs pdf2image (it
# shells out to pdftoppm): both are the OCR fallback path's real runtime
# dependencies, not just its test-time ones. Every test that needs either
# already has its own availability probe and named skip marker for a
# machine that lacks them, exactly per this repo's own testing
# convention, but a deployed demo answering a real scanned PDF upload
# needs the real binaries present, not a skip mark.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# --system: a service account, not a human login, deliberately given no
# explicit --uid. A prior project in this series hardcoded uid 1000 in
# every Dockerfile and two of its images failed to build outright,
# because uid 1000 is also the uid the official node base images already
# assign their own predefined user. Leaving uid selection to the system
# sidesteps that whole class of collision rather than trusting this one
# base image not to repeat it.
RUN useradd --system --create-home --shell /usr/sbin/nologin groundwork

WORKDIR /app

COPY --from=builder --chown=groundwork:groundwork /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Repeated in the runtime stage, not merely trusted from the builder: this
# is the image that actually ships, built from a fresh COPY at a path
# that must independently resolve correctly on its own.
RUN python -c "import groundwork_api" && echo "runtime stage import check passed"

USER groundwork

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

# Migrations run as fly.toml's own release_command, once per deploy,
# never as part of this CMD: a crash looping web process and a failed
# migration are different failures an operator needs to tell apart, and
# folding both into one process makes every restart re-attempt a
# migration that already succeeded or already failed for a real reason.
CMD ["uvicorn", "groundwork_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
