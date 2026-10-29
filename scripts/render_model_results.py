#!/usr/bin/env python3
"""Render MODEL-RESULTS.md from the recorded real model runs, or check it.

README.md and RESULTS.md publish the offline fallback run, which CI
reproduces on every push with no key. A run with real models (a hosted
answering model, the real embedding, reranking and faithfulness models, a
judge from another family) cannot be reproduced by CI, so it is published
differently: scripts/record_model_run.py writes it to results/real-model/
as a record, the record is reviewed and committed, and this script renders
MODEL-RESULTS.md from every committed record. Nothing on that page is typed
by hand, and CI fails when the page and the records disagree, the same rule
the claim gate holds README.md and RESULTS.md to.

Usage:
    python scripts/render_model_results.py            # check, exit 1 on drift
    python scripts/render_model_results.py --write    # render and write in place

--records-dir and --output point it at other paths, so the tests can run a
clean pass, a deliberate violation and an empty input against a throwaway
directory instead of the committed files.

With no records and no page there is nothing to check, and it says so in as
many words rather than reporting a pass. A page with no records behind it,
or records with no page, fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = REPO_ROOT / "results" / "real-model"
OUTPUT = REPO_ROOT / "MODEL-RESULTS.md"
SUITES = (
    ("out_of_scope", "Out of scope refusals"),
    ("injection", "Prompt injection"),
    ("workspace_isolation", "Workspace isolation"),
)


def load_records(records_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(records_dir.glob("*.json")) if records_dir.is_dir() else []
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    for path, record in zip(paths, records, strict=True):
        if record.get("schema") != 1:
            raise ValueError(f"{path.name}: unknown record schema {record.get('schema')!r}")
        record["_file"] = path.name
    return sorted(records, key=lambda r: (r["recorded_at"], r["_file"]))


def _suite(manifest: dict[str, Any], suite: str) -> str:
    return f"{manifest[f'{suite}_pass_count']} of {manifest[f'{suite}_total_count']}"


def _component(backend: str, model: str) -> str:
    """A component's backend, with the model it loaded when it loaded one."""
    if backend == "local_model":
        return f"`{backend}` (`{model}`)"
    if backend == "none":
        return "off"
    return f"`{backend}` (the offline fallback)"


def _summary_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Recorded | Answering model | Judge | Embedding | Turns by the model | "
        + " | ".join(name for _, name in SUITES)
        + " | Claims entailed |",
        "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in SUITES) + " | --- |",
    ]
    for r in records:
        c, m, t = r["configuration"], r["manifest"], r["turns"]
        lines.append(
            f"| {r['recorded_at'][:10]} | `{c['model']}` | `{c['judge_model']}` | "
            f"{c['backends']['embedding']} | {t['written_by_model']} of {t['total']} | "
            + " | ".join(_suite(m, suite) for suite, _ in SUITES)
            + f" | {m['faithfulness_entailed_count']} of {m['faithfulness_claim_count']} |"
        )
    return "\n".join(lines)


def _run_section(r: dict[str, Any]) -> str:
    c, m, t, b = r["configuration"], r["manifest"], r["turns"], r["configuration"]["backends"]
    latency = (
        f"{t['median_model_latency_ms']} ms median per model answer"
        if t["median_model_latency_ms"] is not None
        else "no model answer to time"
    )
    run = f"workflow run {r['workflow_run']}" if r.get("workflow_run") else "a local run"
    commit = f"commit `{r['commit'][:7]}`" if r.get("commit") else "an unrecorded commit"
    parts = [
        f"## {r['recorded_at'][:10]}: `{c['model']}`",
        "",
        f"Recorded {r['recorded_at']} from {run} at {commit}; the record is "
        f"`results/real-model/{r['_file']}`.",
        "",
        f"- **Answering model:** `{c['model']}` (generation backend `{b['generation']}`).",
        f"- **Quality judge:** `{c['judge_model']}` (backend `{b['quality_judge']}`), scoring "
        "clarity and helpfulness only; it never decides faithfulness.",
        f"- **Embeddings:** {_component(b['embedding'], c['embedding_model'])}; stored vectors "
        f"written by {', '.join(f'`{v}`' for v in b['stored_vectors'])}.",
        f"- **Reranker:** {_component(b['rerank'], c['reranker_model'])}. **Faithfulness:** "
        f"{_component(b['faithfulness'], c['nli_model'])}. **Tokenizer:** `{b['tokenizer']}`.",
        f"- **Relevance gate:** {c['relevance_threshold']} cosine similarity. **Spend ceiling:** "
        f"{c['spend_ceiling_usd']} USD; spent {t['cost_usd']} USD.",
        f"- **Turns:** {t['total']}, of which {t['written_by_model']} were written by the model, "
        f"{t['extractive']} fell back to the extractive generator and {t['judged']} were judged; "
        f"{latency}.",
        f"- **Corpus:** {m['document_count']} documents in {m['naive_chunk_count']} naive and "
        f"{m['structure_chunk_count']} structure aware chunks; {m['eval_question_count']} "
        f"questions ({m['eval_category_counts']}).",
        "",
        "### Red team",
        "",
        "| Suite | Passed |",
        "| --- | --- |",
        *(f"| {name} | {_suite(m, suite)} |" for suite, name in SUITES),
        "",
        "### Faithfulness",
        "",
        f"{m['faithfulness_claim_count']} claims over {m['turn_count']} turns: "
        f"{m['faithfulness_entailed_count']} entailed, {m['faithfulness_contradicted_count']} "
        f"contradicted, {m['faithfulness_unsupported_count']} unsupported.",
        "",
        "### Retrieval",
        "",
        m["retrieval_metrics_table"],
        "",
        "### Boundary questions",
        "",
        m["boundary_comparison_table"],
        "",
        "### Out of scope questions the run did not refuse",
        "",
        m["out_of_scope_failures_table"],
    ]
    return "\n".join(parts)


def render(records: list[dict[str, Any]]) -> str:
    header = [
        "# Real model runs",
        "",
        "README.md and RESULTS.md publish the offline fallback run, the one CI reproduces on every "
        "push with no key. This page publishes the runs made with real models: a hosted answering "
        "model, the real embedding, reranking and faithfulness models, and a quality judge from "
        "another family. Each one is a record in `results/real-model/`, written by "
        "`.github/workflows/real-model.yml` and reviewed before it was merged. CI cannot rerun "
        "them, since it has no key; it renders this page from the committed records on every "
        "push and fails if the two disagree, so no figure here is typed by hand.",
        "",
        _summary_table(records),
    ]
    sections = [_run_section(r) for r in reversed(records)]
    return "\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render or check MODEL-RESULTS.md.")
    parser.add_argument("--write", action="store_true", help="render and write in place")
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    records = load_records(args.records_dir)
    if not records:
        if args.output.exists():
            print(f"{args.output.name} exists but no record is committed under {args.records_dir}")
            return 1
        print("nothing to check: no real model run is recorded yet, and no page publishes one")
        return 0
    rendered = render(records)
    if args.write:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output.name} from {len(records)} record(s)")
        return 0
    if not args.output.exists():
        print(
            f"{len(records)} record(s) committed but {args.output.name} is missing; run with --write"
        )
        return 1
    if args.output.read_text(encoding="utf-8") != rendered:
        print(f"{args.output.name} differs from a fresh render of the records; run with --write")
        return 1
    print(f"{args.output.name} matches {len(records)} record(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
