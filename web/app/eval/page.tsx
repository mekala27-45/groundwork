"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  getFaithfulnessScorecard,
  listEvalRuns,
  listRedTeamResults,
  listWorkspaces,
} from "@/lib/api";
import type {
  EvalRunOut,
  FaithfulnessScorecardOut,
  RedTeamResultOut,
  WorkspaceOut,
} from "@/lib/types";
import { CategoryBarChart, GroupedBarChart, type BarSeries } from "@/components/BarChart";
import { StatusPill } from "@/components/StatusPill";

// Fixed order, never reassigned per view, series-1 through series-N in
// the order each dimension is naturally listed elsewhere in this project
// (claims.py's own RED_TEAM_SUITES constant for the suites, section 8's
// four configurations for retrieval). A category keeps the same color
// everywhere it appears on this page.
const CONFIG_SERIES: BarSeries[] = [
  { key: "naive+no_rerank", label: "naive, no rerank", color: "var(--color-series-1)" },
  { key: "naive+rerank", label: "naive, rerank", color: "var(--color-series-2)" },
  { key: "structure+no_rerank", label: "structure, no rerank", color: "var(--color-series-3)" },
  { key: "structure+rerank", label: "structure, rerank", color: "var(--color-series-4)" },
];

const METRICS: { key: "recall_at_3" | "recall_at_5" | "precision_at_5" | "mrr"; label: string }[] = [
  { key: "recall_at_3", label: "Recall@3" },
  { key: "recall_at_5", label: "Recall@5" },
  { key: "precision_at_5", label: "Precision@5" },
  { key: "mrr", label: "MRR" },
];

const RED_TEAM_ORDER = ["injection", "out_of_scope", "workspace_isolation"] as const;
type RedTeamSuite = (typeof RED_TEAM_ORDER)[number];
const RED_TEAM_LABELS: Record<RedTeamSuite, string> = {
  injection: "injection",
  out_of_scope: "out of scope",
  workspace_isolation: "workspace isolation",
};
const RED_TEAM_COLORS: Record<RedTeamSuite, string> = {
  injection: "var(--color-series-1)",
  out_of_scope: "var(--color-series-2)",
  workspace_isolation: "var(--color-series-3)",
};

const FAITHFULNESS_ORDER = ["entailed", "contradicted", "unsupported"] as const;
type FaithfulnessLabel = (typeof FAITHFULNESS_ORDER)[number];
const FAITHFULNESS_COLORS: Record<FaithfulnessLabel, string> = {
  entailed: "var(--color-series-1)",
  contradicted: "var(--color-series-2)",
  unsupported: "var(--color-series-3)",
};

function percent(passed: number, total: number): string {
  return total === 0 ? "n/a" : `${Math.round((passed / total) * 100)}%`;
}

function emptyRedTeamCounts(): Record<RedTeamSuite, { passed: number; total: number }> {
  return {
    injection: { passed: 0, total: 0 },
    out_of_scope: { passed: 0, total: 0 },
    workspace_isolation: { passed: 0, total: 0 },
  };
}

export default function EvalPage(): React.JSX.Element {
  const [workspaces, setWorkspaces] = useState<WorkspaceOut[] | null>(null);
  const [evalRuns, setEvalRuns] = useState<EvalRunOut[] | null>(null);
  const [redTeamResults, setRedTeamResults] = useState<RedTeamResultOut[] | null>(null);
  const [faithfulness, setFaithfulness] = useState<FaithfulnessScorecardOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([listWorkspaces(), listEvalRuns(), listRedTeamResults(), getFaithfulnessScorecard()])
      .then(([workspacesData, evalRunsData, redTeamData, faithfulnessData]) => {
        if (cancelled) return;
        setWorkspaces(workspacesData);
        setEvalRuns(evalRunsData);
        setRedTeamResults(redTeamData);
        setFaithfulness(faithfulnessData);
        const allCategoryRuns = evalRunsData.filter((run) => run.category === "all");
        const mostQuestions = allCategoryRuns.reduce<EvalRunOut | null>((best, run) => {
          if (!best || run.n_questions > best.n_questions) return run;
          return best;
        }, null);
        setSelectedWorkspaceId(mostQuestions?.workspace_id ?? null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "could not reach the API");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const workspaceName = useMemo(() => {
    const map = new Map((workspaces ?? []).map((w) => [w.id, w.name]));
    return (id: string) => map.get(id) ?? id;
  }, [workspaces]);

  const workspaceOptions = useMemo(() => {
    if (!evalRuns) return [];
    const ids = new Set(
      evalRuns.filter((run) => run.category === "all").map((run) => run.workspace_id)
    );
    return Array.from(ids);
  }, [evalRuns]);

  const selectedRuns = useMemo(() => {
    if (!evalRuns || !selectedWorkspaceId) return [];
    return evalRuns.filter(
      (run) => run.category === "all" && run.workspace_id === selectedWorkspaceId
    );
  }, [evalRuns, selectedWorkspaceId]);

  const retrievalChartValues = METRICS.map((metric) =>
    CONFIG_SERIES.map((series) => {
      const run = selectedRuns.find((r) => r.config_label === series.key);
      return run ? run[metric.key] : 0;
    })
  );

  const redTeamCounts = useMemo(() => {
    const counts = emptyRedTeamCounts();
    for (const row of redTeamResults ?? []) {
      const bucket = counts[row.suite];
      bucket.total += 1;
      if (row.passed) bucket.passed += 1;
    }
    return counts;
  }, [redTeamResults]);

  const faithfulnessCounts: Record<FaithfulnessLabel, number> = {
    entailed: faithfulness?.entailed_count ?? 0,
    contradicted: faithfulness?.contradicted_count ?? 0,
    unsupported: faithfulness?.unsupported_count ?? 0,
  };
  const faithfulnessTotal = faithfulness?.total_count ?? 0;

  const injection = redTeamCounts.injection;

  if (error) {
    return (
      <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] p-6">
        <p className="status-fail font-medium">Could not reach the groundwork API.</p>
        <p className="mt-2 text-sm text-[var(--text-secondary)]">{error}</p>
      </div>
    );
  }

  if (loading) {
    return <p className="text-sm text-[var(--text-muted)]">Loading eval results...</p>;
  }

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Eval</h1>
        <p className="mt-1 max-w-2xl text-sm text-[var(--text-secondary)]">
          Every number below comes from the same stored rows RESULTS.md is rendered from, read
          live from the API rather than pasted in. A table sits alongside every chart with the
          exact values the bars round for display.
        </p>
      </div>

      <section className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] p-5">
        <p className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
          Indirect prompt injection defense
        </p>
        <p className="mt-2 text-sm text-[var(--text-secondary)]">
          The single number a careful reader looks for first: whether retrieved document
          content was ever treated as an instruction rather than data.
        </p>
        <div className="mt-3 flex items-center gap-3">
          <span className="mono text-3xl font-semibold text-[var(--text-primary)]">
            {injection.passed}/{injection.total}
          </span>
          <StatusPill
            status={injection.total > 0 && injection.passed === injection.total ? "pass" : "fail"}
          >
            {percent(injection.passed, injection.total)} passed
          </StatusPill>
        </div>
      </section>

      <section>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-medium">Retrieval metrics</h2>
            <p className="text-sm text-[var(--text-secondary)]">
              Naive versus structure aware chunking, with and without reranking, section
              8&apos;s full comparison.
            </p>
          </div>
          {workspaceOptions.length > 1 && (
            <div className="flex flex-wrap gap-1.5">
              {workspaceOptions.map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setSelectedWorkspaceId(id)}
                  className={`rounded-md border px-2.5 py-1 text-xs ${
                    id === selectedWorkspaceId
                      ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--text-primary)]"
                      : "border-[var(--border-subtle)] text-[var(--text-secondary)] hover:border-[var(--border-strong)]"
                  }`}
                >
                  {workspaceName(id)}
                </button>
              ))}
            </div>
          )}
        </div>
        {selectedRuns.length === 0 ? (
          <p className="mt-4 text-sm text-[var(--text-muted)]">
            No EvalRun rows yet. Run <code className="mono">scripts/run_eval.py</code>.
          </p>
        ) : (
          <div className="mt-4 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] p-4">
            <GroupedBarChart
              groups={METRICS.map((m) => m.label)}
              series={CONFIG_SERIES}
              values={retrievalChartValues}
              maxValue={1}
              formatValue={(v) => v.toFixed(2)}
              ariaLabel="Retrieval metrics by chunking strategy and reranking"
            />
            <details className="mt-3 text-sm" open>
              <summary className="cursor-pointer text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
                Table view
              </summary>
              <div className="mt-2 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="text-[var(--text-muted)]">
                      <th className="py-1 pr-3 font-medium">Config</th>
                      {METRICS.map((m) => (
                        <th key={m.key} className="py-1 pr-3 font-medium">
                          {m.label}
                        </th>
                      ))}
                      <th className="py-1 font-medium">N</th>
                    </tr>
                  </thead>
                  <tbody className="mono">
                    {CONFIG_SERIES.map((series) => {
                      const run = selectedRuns.find((r) => r.config_label === series.key);
                      return (
                        <tr key={series.key} className="border-t border-[var(--border-subtle)]">
                          <td className="py-1 pr-3 text-[var(--text-secondary)]">{series.label}</td>
                          {METRICS.map((m) => (
                            <td key={m.key} className="py-1 pr-3 text-[var(--text-primary)]">
                              {run ? run[m.key].toFixed(3) : "n/a"}
                            </td>
                          ))}
                          <td className="py-1 text-[var(--text-primary)]">
                            {run ? run.n_questions : "n/a"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-medium">Faithfulness scorecard</h2>
        <p className="text-sm text-[var(--text-secondary)]">
          Every generated claim, independently checked against its cited passage: entailed,
          contradicted, or unsupported. Contradicted is counted separately from unsupported,
          the two mean very different things to a reader deciding how much to trust an answer.
        </p>
        <div className="mt-4 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] p-4">
          {faithfulnessTotal === 0 ? (
            <p className="text-sm text-[var(--text-muted)]">No claims recorded yet.</p>
          ) : (
            <>
              <CategoryBarChart
                bars={FAITHFULNESS_ORDER.map((label) => ({
                  label,
                  value: faithfulnessCounts[label],
                  color: FAITHFULNESS_COLORS[label],
                }))}
                maxValue={Math.max(1, faithfulnessTotal)}
                formatValue={(v) => `${v}`}
                ariaLabel="Faithfulness claim counts by NLI label"
              />
              <details className="mt-3 text-sm" open>
                <summary className="cursor-pointer text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
                  Table view
                </summary>
                <table className="mono mt-2 w-full text-left text-xs">
                  <tbody>
                    {FAITHFULNESS_ORDER.map((label) => (
                      <tr key={label} className="border-t border-[var(--border-subtle)]">
                        <td className="py-1 pr-3 text-[var(--text-secondary)]">{label}</td>
                        <td className="py-1 pr-3 text-[var(--text-primary)]">
                          {faithfulnessCounts[label]}
                        </td>
                        <td className="py-1 text-[var(--text-primary)]">
                          {((faithfulnessCounts[label] / faithfulnessTotal) * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            </>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-lg font-medium">Red team category pass rates</h2>
        <p className="text-sm text-[var(--text-secondary)]">
          Section 11&apos;s three probes: a document boundary that should never be crossed, an
          instruction that should never be obeyed, and a question that should be refused rather
          than guessed at.
        </p>
        <div className="mt-4 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] p-4">
          <CategoryBarChart
            bars={RED_TEAM_ORDER.map((suite) => {
              const bucket = redTeamCounts[suite];
              return {
                label: RED_TEAM_LABELS[suite],
                value: bucket.total ? (bucket.passed / bucket.total) * 100 : 0,
                color: RED_TEAM_COLORS[suite],
              };
            })}
            maxValue={100}
            formatValue={(v) => `${Math.round(v)}%`}
            ariaLabel="Red team pass rate by category"
          />
          <details className="mt-3 text-sm" open>
            <summary className="cursor-pointer text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
              Table view
            </summary>
            <table className="mono mt-2 w-full text-left text-xs">
              <tbody>
                {RED_TEAM_ORDER.map((suite) => {
                  const bucket = redTeamCounts[suite];
                  return (
                    <tr key={suite} className="border-t border-[var(--border-subtle)]">
                      <td className="py-1 pr-3 text-[var(--text-secondary)]">
                        {RED_TEAM_LABELS[suite]}
                      </td>
                      <td className="py-1 pr-3 text-[var(--text-primary)]">
                        {bucket.passed}/{bucket.total}
                      </td>
                      <td className="py-1 text-[var(--text-primary)]">
                        {percent(bucket.passed, bucket.total)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </details>
          {redTeamCounts.out_of_scope.total > 0 && (
            <p className="mt-3 text-xs text-[var(--text-muted)]">
              Out of scope failures are published in full, with each one linked to its trace, in{" "}
              <a
                href="https://github.com/mekala27-45/groundwork/blob/main/RESULTS.md"
                target="_blank"
                rel="noreferrer"
                className="text-[var(--color-accent)] hover:underline"
              >
                RESULTS.md
              </a>
              .
            </p>
          )}
        </div>
      </section>
    </div>
  );
}
