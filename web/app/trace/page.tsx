"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ApiError, getChunk, getTurn } from "@/lib/api";
import type { ChunkOut, TurnOut } from "@/lib/types";
import { StatusPill, type Status } from "@/components/StatusPill";

function nliStatus(label: string): Status {
  if (label === "entailed") return "pass";
  if (label === "contradicted") return "fail";
  return "warn"; // "unsupported"
}

// Turn.retrieved_chunk_ids is an ordered array of bare chunk ids: nothing
// in this codebase persists a per-chunk retrieval score or a rerank
// before/after position (packages/retrieve/src/groundwork_retrieve/
// search.py computes cosine distance only to order the SQL query, never
// selects it back out; the rerank cross encoder's own scores are used the
// same way in packages/api/src/groundwork_api/chat.py's _retrieve). This
// page shows the real final order instead of inventing numbers section 12
// asks for but this build never measured: rank position and the turn
// level reranked flag are true, a fabricated per-chunk score would not
// be.
const SCORE_DISCLOSURE =
  "This build orders chunks by retrieval rank, the real order the answer " +
  "was generated from. It does not persist a per-chunk similarity score " +
  "or a rerank before/after position, so neither is shown here rather " +
  "than invented.";

function TracePageInner(): React.JSX.Element {
  const router = useRouter();
  const searchParams = useSearchParams();
  const turnId = searchParams.get("turn");

  const [turnIdInput, setTurnIdInput] = useState(turnId ?? "");
  const [turn, setTurn] = useState<TurnOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [turnError, setTurnError] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Record<string, ChunkOut | "error">>({});

  useEffect(() => {
    setTurnIdInput(turnId ?? "");
    setChunks({});
    if (!turnId) {
      setTurn(null);
      setTurnError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setTurn(null);
    setTurnError(null);
    getTurn(turnId)
      .then((data) => {
        if (!cancelled) setTurn(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setTurnError(err instanceof ApiError ? err.message : "could not reach the API");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [turnId]);

  useEffect(() => {
    if (!turn || turn.retrieved_chunk_ids.length === 0) return;
    let cancelled = false;
    void Promise.all(
      turn.retrieved_chunk_ids.map(async (id) => {
        try {
          return [id, await getChunk(id)] as const;
        } catch {
          return [id, "error" as const] as const;
        }
      })
    ).then((entries) => {
      if (!cancelled) setChunks(Object.fromEntries(entries));
    });
    return () => {
      cancelled = true;
    };
  }, [turn]);

  function handleJump(event: React.FormEvent): void {
    event.preventDefault();
    const trimmed = turnIdInput.trim();
    if (!trimmed) return;
    router.push(`/trace?turn=${trimmed}`);
  }

  const citationByChunkId = new Map(
    (turn?.citation_verifications ?? []).map((citation) => [citation.chunk_id, citation])
  );

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Trace</h1>
          <p className="mt-1 max-w-2xl text-sm text-[var(--text-secondary)]">
            Every stage one answer actually went through: what was retrieved, what was
            answered, whether each claim held up against its cited passage, and whether
            each citation is real. Open one from a chat answer or the eval dashboard, or
            paste a turn id below.
          </p>
        </div>
        <form onSubmit={handleJump} className="flex max-w-md items-center gap-2">
          <input
            type="text"
            value={turnIdInput}
            onChange={(event) => setTurnIdInput(event.target.value)}
            placeholder="turn id"
            className="mono flex-1 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-3 py-1.5 text-xs outline-none focus:border-[var(--color-accent)]"
          />
          <button
            type="submit"
            className="rounded-md border border-[var(--border-subtle)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] hover:border-[var(--color-accent)] hover:text-[var(--text-primary)]"
          >
            Open
          </button>
        </form>
      </div>

      {!turnId && (
        <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--bg-raised)] p-6 text-sm text-[var(--text-muted)]">
          No turn selected. Ask a question in{" "}
          <Link href="/chat" className="text-[var(--color-accent)] hover:underline">
            chat
          </Link>{" "}
          and follow its &ldquo;View full trace&rdquo; link, or open a failing case from{" "}
          <Link href="/eval" className="text-[var(--color-accent)] hover:underline">
            eval
          </Link>
          .
        </div>
      )}

      {loading && <p className="text-sm text-[var(--text-muted)]">Loading trace...</p>}

      {turnError && (
        <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] p-6">
          <p className="status-fail font-medium">Could not load this trace.</p>
          <p className="mt-2 text-sm text-[var(--text-secondary)]">{turnError}</p>
        </div>
      )}

      {turn && (
        <div className="space-y-6">
          <section>
            <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
              1. Question
            </p>
            <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-4 py-3 text-sm">
              {turn.question}
            </div>
          </section>

          <section>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <p className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
                2. Retrieved chunks ({turn.retrieved_chunk_ids.length})
              </p>
              <StatusPill status="neutral">{turn.reranked ? "reranked" : "no rerank"}</StatusPill>
            </div>
            <p className="mb-2 text-xs text-[var(--text-muted)]">{SCORE_DISCLOSURE}</p>
            {turn.retrieved_chunk_ids.length === 0 ? (
              <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-4 py-3 text-sm text-[var(--text-muted)]">
                Nothing was retrieved, or the top match fell below the relevance threshold
                before generation ran.
              </div>
            ) : (
              <ol className="space-y-2">
                {turn.retrieved_chunk_ids.map((chunkId, index) => {
                  const chunk = chunks[chunkId];
                  const citation = citationByChunkId.get(chunkId);
                  return (
                    <li
                      key={`${chunkId}-${index}`}
                      id={`chunk-${chunkId}`}
                      className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-4 py-3 text-sm"
                    >
                      <div className="mb-1.5 flex flex-wrap items-center gap-2">
                        <span className="mono text-xs text-[var(--text-muted)]">
                          rank {index + 1}
                        </span>
                        {citation ? (
                          <StatusPill status={citation.verified ? "pass" : "fail"}>
                            cited, {citation.verified ? "verified" : "unverified"}
                          </StatusPill>
                        ) : (
                          <StatusPill status="neutral">not cited</StatusPill>
                        )}
                      </div>
                      {chunk === undefined && (
                        <p className="text-[var(--text-muted)]">Loading passage...</p>
                      )}
                      {chunk === "error" && (
                        <p className="status-fail">Could not load this passage.</p>
                      )}
                      {chunk && chunk !== "error" && (
                        <div className="space-y-1">
                          <p className="text-xs text-[var(--text-muted)]">
                            {chunk.section_title ? `${chunk.section_title} ` : ""}
                            {chunk.page_start === chunk.page_end
                              ? `page ${chunk.page_start}`
                              : `pages ${chunk.page_start} to ${chunk.page_end}`}{" "}
                            &middot; {chunk.strategy} chunking
                          </p>
                          <p className="text-[var(--text-primary)]">{chunk.text}</p>
                          {chunk.injection_flag && (
                            <p className="status-warn text-xs">
                              Flagged at ingestion: {chunk.injection_flag}
                            </p>
                          )}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ol>
            )}
          </section>

          <section>
            <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
              3. Answer
            </p>
            <div className="space-y-2 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-4 py-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <StatusPill status={turn.extractive_fallback ? "warn" : "neutral"}>
                  {turn.extractive_fallback ? "extractive answer" : "generated answer"}
                </StatusPill>
                <StatusPill status="neutral">
                  {turn.chunking_strategy ?? "no retrieval"}
                </StatusPill>
              </div>
              <p className="whitespace-pre-wrap text-[var(--text-primary)]">{turn.answer}</p>
            </div>
          </section>

          <section>
            <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
              4. Claims ({turn.claims.length})
            </p>
            {turn.claims.length === 0 ? (
              <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-4 py-3 text-sm text-[var(--text-muted)]">
                This answer made no separately checked claims, the expected shape for a
                stated refusal.
              </div>
            ) : (
              <ul className="space-y-2">
                {turn.claims.map((claim, index) => (
                  <li
                    key={`claim-${index}`}
                    className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-4 py-3 text-sm"
                  >
                    <div className="mb-1.5 flex flex-wrap items-center gap-2">
                      <StatusPill status={nliStatus(claim.nli_label)}>
                        {claim.nli_label}
                      </StatusPill>
                      <span className="mono text-xs text-[var(--text-muted)]">
                        confidence {claim.score.toFixed(2)}
                      </span>
                      {claim.cited_chunk_id && (
                        <a
                          href={`#chunk-${claim.cited_chunk_id}`}
                          className="text-xs text-[var(--color-accent)] hover:underline"
                        >
                          cited passage &darr;
                        </a>
                      )}
                    </div>
                    <p className="text-[var(--text-primary)]">{claim.text}</p>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section>
            <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
              5. Citation verification ({turn.citation_verifications.length})
            </p>
            {turn.citation_verifications.length === 0 ? (
              <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-4 py-3 text-sm text-[var(--text-muted)]">
                No citation was made. Verified only means the quoted text really appears in
                the cited chunk, not that the chunk actually answers the question, which is
                why a wrong-but-quoted-verbatim answer can still show every citation
                verified: see the out of scope failures in RESULTS.md for real examples.
              </div>
            ) : (
              <ul className="space-y-2">
                {turn.citation_verifications.map((citation, index) => (
                  <li
                    key={`${citation.chunk_id}-${index}`}
                    className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] px-4 py-3 text-sm"
                  >
                    <StatusPill status={citation.verified ? "pass" : "fail"}>
                      {citation.verified ? "verified" : "unverified"}
                    </StatusPill>
                    <a
                      href={`#chunk-${citation.chunk_id}`}
                      className="mono text-xs text-[var(--color-accent)] hover:underline"
                    >
                      {citation.chunk_id}
                    </a>
                    {citation.reason && (
                      <span className="text-xs text-[var(--text-muted)]">
                        {citation.reason}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

export default function TracePage(): React.JSX.Element {
  return (
    <Suspense fallback={<p className="text-sm text-[var(--text-muted)]">Loading...</p>}>
      <TracePageInner />
    </Suspense>
  );
}
