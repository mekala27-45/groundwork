"use client";

import { useState } from "react";
import { ApiError, getChunk } from "@/lib/api";
import type { ChunkOut, CitationVerificationOut } from "@/lib/types";
import { StatusPill } from "./StatusPill";

// Section 12's own requirement: "Citations are inline, numbered, and
// expandable to the exact highlighted source passage on click, with its
// page number." Turn.citation_verifications carries no citation number of
// its own (see lib/types.ts's own comment on why this file mirrors the
// API schemas exactly), so this list numbers each entry by its position
// in the array, the same order the answer's own citations were verified
// in, and fetches the real chunk text from GET /chunks/{id} only once a
// reader actually expands one, rather than eagerly loading every source
// for every turn whether or not anyone looks at it.
export function CitationList({
  citations,
}: {
  citations: CitationVerificationOut[];
}): React.JSX.Element | null {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [chunks, setChunks] = useState<Record<string, ChunkOut | "error">>({});

  if (citations.length === 0) {
    return null;
  }

  async function toggle(chunkId: string): Promise<void> {
    if (expandedId === chunkId) {
      setExpandedId(null);
      return;
    }
    setExpandedId(chunkId);
    if (!(chunkId in chunks)) {
      try {
        const chunk = await getChunk(chunkId);
        setChunks((prev) => ({ ...prev, [chunkId]: chunk }));
      } catch (err) {
        setChunks((prev) => ({ ...prev, [chunkId]: "error" }));
        if (!(err instanceof ApiError)) {
          throw err;
        }
      }
    }
  }

  return (
    <div className="mt-3 space-y-1.5">
      <p className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
        Sources
      </p>
      <ul className="space-y-1.5">
        {citations.map((citation, index) => {
          const isOpen = expandedId === citation.chunk_id;
          const chunk = chunks[citation.chunk_id];
          return (
            <li
              key={`${citation.chunk_id}-${index}`}
              className="rounded-md border border-[var(--border-subtle)] bg-[var(--bg-raised)]"
            >
              <button
                type="button"
                onClick={() => void toggle(citation.chunk_id)}
                className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm"
                aria-expanded={isOpen}
              >
                <span className="flex items-center gap-2">
                  <span className="mono text-xs text-[var(--text-muted)]">
                    [{index + 1}]
                  </span>
                  <StatusPill status={citation.verified ? "pass" : "fail"}>
                    {citation.verified ? "verified" : "unverified"}
                  </StatusPill>
                  {!citation.verified && citation.reason && (
                    <span className="text-xs text-[var(--text-muted)]">{citation.reason}</span>
                  )}
                </span>
                <span className="text-xs text-[var(--text-muted)]">
                  {isOpen ? "Hide passage" : "Show passage"}
                </span>
              </button>
              {isOpen && (
                <div className="border-t border-[var(--border-subtle)] px-3 py-2 text-sm">
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
                          : `pages ${chunk.page_start} to ${chunk.page_end}`}
                        {" "}&middot; {chunk.strategy} chunking
                      </p>
                      <p className="border-l-2 border-[var(--color-accent)] pl-3 text-[var(--text-primary)]">
                        {chunk.text}
                      </p>
                      {chunk.injection_flag && (
                        <p className="status-warn text-xs">
                          Flagged at ingestion: {chunk.injection_flag}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
