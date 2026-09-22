"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ApiError,
  askQuestion,
  createConversation,
  createWorkspace,
  listWorkspaces,
} from "@/lib/api";
import type { ChunkStrategy, TurnOut, WorkspaceOut } from "@/lib/types";
import { CitationList } from "@/components/CitationList";
import { StatusPill } from "@/components/StatusPill";

function ChatPageInner(): React.JSX.Element {
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedWorkspaceId = searchParams.get("workspace");

  const [workspaces, setWorkspaces] = useState<WorkspaceOut[] | null>(null);
  const [workspacesError, setWorkspacesError] = useState<string | null>(null);
  const [workspaceId, setWorkspaceId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<TurnOut[]>([]);
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState<ChunkStrategy>("structure");
  const [useReranking, setUseReranking] = useState(true);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listWorkspaces()
      .then((data) => {
        if (cancelled) return;
        setWorkspaces(data);
        const [first] = data;
        const initial =
          data.find((w) => w.id === requestedWorkspaceId)?.id ?? first?.id ?? null;
        setWorkspaceId(initial);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setWorkspacesError(err instanceof ApiError ? err.message : "could not reach the API");
      });
    return () => {
      cancelled = true;
    };
    // requestedWorkspaceId is read once, on the initial load, matching the
    // "open a link straight into a workspace" use case; a workspace picked
    // afterward through the UI updates workspaceId directly instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!workspaceId) return;
    let cancelled = false;
    setConversationId(null);
    setTurns([]);
    setAskError(null);
    createConversation(workspaceId)
      .then((conversation) => {
        if (!cancelled) setConversationId(conversation.id);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setAskError(err instanceof ApiError ? err.message : "could not start a conversation");
        }
      });
    router.replace(`/chat?workspace=${workspaceId}`, { scroll: false });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceId]);

  async function handleAsk(): Promise<void> {
    if (!conversationId || !question.trim() || asking) return;
    setAsking(true);
    setAskError(null);
    const asked = question;
    setQuestion("");
    try {
      const turn = await askQuestion(conversationId, {
        question: asked,
        strategy,
        use_reranking: useReranking,
      });
      setTurns((prev) => [...prev, turn]);
    } catch (err) {
      setQuestion(asked);
      setAskError(err instanceof ApiError ? err.message : "the request failed");
    } finally {
      setAsking(false);
    }
  }

  async function handleUpload(file: File): Promise<void> {
    setUploading(true);
    setUploadError(null);
    try {
      const created = await createWorkspace(file, file.name.replace(/\.pdf$/i, ""));
      setWorkspaces((prev) => (prev ? [...prev, created] : [created]));
      setWorkspaceId(created.id);
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.message : "upload failed");
    } finally {
      setUploading(false);
    }
  }

  if (workspacesError) {
    return (
      <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] p-6">
        <p className="status-fail font-medium">Could not reach the groundwork API.</p>
        <p className="mt-2 text-sm text-[var(--text-secondary)]">{workspacesError}</p>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          If you are running this app locally, confirm the API is running and
          NEXT_PUBLIC_API_BASE_URL points at it.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
      <aside className="space-y-4">
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
            Workspace
          </p>
          <div className="space-y-1.5">
            {workspaces === null && (
              <p className="text-sm text-[var(--text-muted)]">Loading...</p>
            )}
            {workspaces?.map((workspace) => (
              <button
                key={workspace.id}
                type="button"
                onClick={() => setWorkspaceId(workspace.id)}
                className={`w-full rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                  workspace.id === workspaceId
                    ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--text-primary)]"
                    : "border-[var(--border-subtle)] bg-[var(--bg-raised)] text-[var(--text-secondary)] hover:border-[var(--border-strong)]"
                }`}
              >
                {workspace.name}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="block w-full cursor-pointer rounded-md border border-dashed border-[var(--border-strong)] px-3 py-2 text-center text-sm text-[var(--text-secondary)] transition-colors hover:border-[var(--color-accent)] hover:text-[var(--text-primary)]">
            {uploading ? "Uploading..." : "Upload your own PDF"}
            <input
              type="file"
              accept="application/pdf"
              className="hidden"
              disabled={uploading}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleUpload(file);
                event.target.value = "";
              }}
            />
          </label>
          {uploadError && <p className="status-fail mt-1 text-xs">{uploadError}</p>}
        </div>
        <div className="space-y-2 border-t border-[var(--border-subtle)] pt-4">
          <p className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
            Retrieval
          </p>
          <div className="flex flex-col gap-1.5 text-sm">
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="strategy"
                checked={strategy === "naive"}
                onChange={() => setStrategy("naive")}
              />
              Naive chunking
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="strategy"
                checked={strategy === "structure"}
                onChange={() => setStrategy("structure")}
              />
              Structure aware chunking
            </label>
            <label className="mt-1 flex items-center gap-2">
              <input
                type="checkbox"
                checked={useReranking}
                onChange={(event) => setUseReranking(event.target.checked)}
              />
              Rerank results
            </label>
          </div>
          <p className="text-xs text-[var(--text-muted)]">
            Both strategies are indexed for every document; switching here changes which
            index a new question is answered against, so the same question can be asked
            twice to compare retrieval directly. See /eval for the full measured comparison.
          </p>
        </div>
      </aside>

      <section className="flex min-h-[60vh] flex-col rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)]">
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {turns.length === 0 && (
            <p className="text-sm text-[var(--text-muted)]">
              Ask this workspace anything covered by its documents. A question outside their
              scope gets a stated refusal, not a guess.
            </p>
          )}
          {turns.map((turn) => (
            <div key={turn.id} className="space-y-2">
              <div className="ml-auto max-w-[85%] rounded-lg bg-[var(--color-accent)]/15 px-3 py-2 text-sm">
                {turn.question}
              </div>
              <div className="max-w-[90%] rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-sunken)] px-3 py-2 text-sm">
                <p className="whitespace-pre-wrap text-[var(--text-primary)]">{turn.answer}</p>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <StatusPill status={turn.extractive_fallback ? "warn" : "neutral"}>
                    {turn.extractive_fallback ? "extractive answer" : "generated answer"}
                  </StatusPill>
                  <StatusPill status="neutral">
                    {turn.chunking_strategy ?? "no retrieval"}
                    {turn.reranked ? " + rerank" : ""}
                  </StatusPill>
                  <Link
                    href={`/trace?turn=${turn.id}`}
                    className="text-xs text-[var(--color-accent)] hover:underline"
                  >
                    View full trace &rarr;
                  </Link>
                </div>
                <CitationList citations={turn.citation_verifications} />
              </div>
            </div>
          ))}
        </div>
        <form
          className="flex items-center gap-2 border-t border-[var(--border-subtle)] p-3"
          onSubmit={(event) => {
            event.preventDefault();
            void handleAsk();
          }}
        >
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={conversationId ? "Ask a question..." : "Starting conversation..."}
            disabled={!conversationId || asking}
            className="flex-1 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-base)] px-3 py-2 text-sm outline-none focus:border-[var(--color-accent)]"
          />
          <button
            type="submit"
            disabled={!conversationId || asking || !question.trim()}
            className="rounded-md bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {asking ? "Asking..." : "Ask"}
          </button>
        </form>
        {askError && <p className="status-fail px-3 pb-3 text-xs">{askError}</p>}
      </section>
    </div>
  );
}

export default function ChatPage(): React.JSX.Element {
  return (
    <Suspense fallback={<p className="text-sm text-[var(--text-muted)]">Loading...</p>}>
      <ChatPageInner />
    </Suspense>
  );
}
