#!/usr/bin/env python3
"""Record a real model evaluation run, for review before it is published.

Run it after the usual three steps (scripts/seed_demo_workspaces.py, then
scripts/build_eval_questions.py, then scripts/run_eval.py) in an environment
with a model key: .github/workflows/real-model.yml does exactly that. It
builds the same manifest the claim gate renders README.md and RESULTS.md
from (groundwork_api.claims.build_manifest), adds what produced it (the
answering model, the judge, the backend every component resolved to, which
embedding backend wrote the stored vectors, and how many turns a model
actually wrote), and writes all of it to results/real-model/ as one JSON
record. scripts/render_model_results.py renders MODEL-RESULTS.md from the
committed records; nothing in that page is typed by hand.

It refuses to write a record in which no turn came from the model. A key
that was configured but never used, because the spend ceiling was spent or
the provider refused every call, falls back to the extractive generator
without an error, and those extractive numbers must never be published as a
model's.

Usage:
    python scripts/record_model_run.py                 # writes results/real-model/<name>.json
    python scripts/record_model_run.py --output PATH   # writes PATH instead
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = REPO_ROOT / "results" / "real-model"
SCHEMA_VERSION = 1


class NoModelTurnsError(RuntimeError):
    """The run wrote no turn with a model: there is nothing to record as a model run."""


def summarize_turns(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts over the run's turns. Each turn is a dict with the Turn columns
    this needs: extractive_fallback, judge_scores, latency_ms, cost_usd."""
    by_model = [t for t in turns if not t["extractive_fallback"]]
    latencies = [float(t["latency_ms"]) for t in by_model]
    return {
        "total": len(turns),
        "written_by_model": len(by_model),
        "extractive": len(turns) - len(by_model),
        "judged": sum(1 for t in turns if t["judge_scores"] is not None),
        "median_model_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "cost_usd": round(sum(float(t["cost_usd"]) for t in turns), 6),
    }


def build_record(
    *,
    manifest: dict[str, Any],
    turns: dict[str, Any],
    configuration: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """The record as written. Raises NoModelTurnsError when no turn came from
    the model, whatever else the run produced."""
    if turns["written_by_model"] == 0:
        raise NoModelTurnsError(
            f"none of the {turns['total']} turns was written by {configuration['model']}: "
            "every answer fell back to the extractive generator (no key reached the run, the "
            "spend ceiling was spent, or the provider refused every call), so there is no "
            "model run to record"
        )
    return {
        "schema": SCHEMA_VERSION,
        **provenance,
        "configuration": configuration,
        "turns": turns,
        "manifest": manifest,
    }


def record_name(model: str, recorded_at: str, run_id: str | None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower().split("/")[-1]).strip("-")
    return f"{recorded_at[:10]}-{slug}-{run_id or 'local'}.json"


def _commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def collect() -> dict[str, Any]:
    from sqlalchemy import select
    from sqlmodel import col

    from groundwork_api.claims import build_manifest
    from groundwork_api.db import dispose_engine, session_scope
    from groundwork_api.models import Chunk, Turn
    from groundwork_chunk.tokenize import get_tokenizer_backend
    from groundwork_core.config import get_settings
    from groundwork_generate.generate import resolve_generation_backend
    from groundwork_retrieve.embeddings import resolve_embedding_backend
    from groundwork_retrieve.rerank import resolve_rerank_backend
    from groundwork_verify.faithfulness import resolve_faithfulness_backend
    from groundwork_verify.quality import resolve_quality_backend

    settings = get_settings()
    try:
        async with session_scope() as session:
            manifest = await build_manifest(session)
            rows = (await session.execute(select(Turn))).scalars().all()
            vectors = (
                (await session.execute(select(col(Chunk.embedding_backend)).distinct()))
                .scalars()
                .all()
            )
    finally:
        await dispose_engine()

    turns = summarize_turns(
        [
            {
                "extractive_fallback": t.extractive_fallback,
                "judge_scores": t.judge_scores,
                "latency_ms": t.latency_ms,
                "cost_usd": t.cost_usd,
            }
            for t in rows
        ]
    )
    configuration = {
        "model": settings.llm_model,
        "judge_model": settings.judge_model or settings.llm_model,
        "backends": {
            "generation": resolve_generation_backend(),
            "quality_judge": resolve_quality_backend(),
            "embedding": resolve_embedding_backend(),
            "stored_vectors": sorted(str(v) for v in vectors),
            "rerank": resolve_rerank_backend(),
            "faithfulness": resolve_faithfulness_backend(),
            "tokenizer": get_tokenizer_backend(),
        },
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
        "nli_model": settings.nli_model_name,
        "relevance_threshold": settings.relevance_threshold,
        "spend_ceiling_usd": settings.spend_ceiling_usd,
    }
    recorded_at = datetime.now(UTC).isoformat(timespec="seconds")
    provenance = {
        "recorded_at": recorded_at,
        "commit": _commit(),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
    }
    return build_record(
        manifest=manifest, turns=turns, configuration=configuration, provenance=provenance
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a real model evaluation run.")
    parser.add_argument("--output", type=Path, default=None, help="write the record here")
    args = parser.parse_args(argv)
    try:
        record = asyncio.run(collect())
    except NoModelTurnsError as exc:
        print(f"not recorded: {exc}", file=sys.stderr)
        return 1
    path = args.output or RECORDS_DIR / record_name(
        record["configuration"]["model"], record["recorded_at"], record["workflow_run"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    turns = record["turns"]
    print(
        f"recorded {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}: "
        f"{turns['written_by_model']} of {turns['total']} turns written by "
        f"{record['configuration']['model']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
