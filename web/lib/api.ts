import type {
  AskRequest,
  ChunkOut,
  ConversationOut,
  ErrorOut,
  EvalRunOut,
  RedTeamResultOut,
  TurnOut,
  WorkspaceDetailOut,
  WorkspaceOut,
} from "./types";

// The static export has no server of its own (next.config.ts's own
// comment explains why), so every call here runs in the visitor's
// browser against whichever Fly hosted API NEXT_PUBLIC_API_BASE_URL
// pointed at when this build was produced. Falling back to localhost
// keeps `npm run dev` usable against a locally running
// `uvicorn groundwork_api.app:app` with no environment file required for
// the common case of iterating on this app against a real backend on the
// same machine.
const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (
      body !== null &&
      typeof body === "object" &&
      "detail" in body &&
      typeof (body as ErrorOut).detail === "string"
    ) {
      return (body as ErrorOut).detail;
    }
  } catch {
    // response body was not JSON, fall through to the status text below
  }
  return response.statusText || `request failed with status ${response.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response));
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function listWorkspaces(): Promise<WorkspaceOut[]> {
  return request<WorkspaceOut[]>("/workspaces");
}

export function getWorkspace(workspaceId: string): Promise<WorkspaceDetailOut> {
  return request<WorkspaceDetailOut>(`/workspaces/${workspaceId}`);
}

export function createWorkspace(file: File, name?: string): Promise<WorkspaceDetailOut> {
  const form = new FormData();
  form.append("file", file);
  if (name) {
    form.append("name", name);
  }
  return request<WorkspaceDetailOut>("/workspaces", { method: "POST", body: form });
}

export function addDocument(workspaceId: string, file: File): Promise<WorkspaceDetailOut> {
  const form = new FormData();
  form.append("file", file);
  return request<WorkspaceDetailOut>(`/workspaces/${workspaceId}/documents`, {
    method: "POST",
    body: form,
  });
}

export function createConversation(workspaceId: string): Promise<ConversationOut> {
  return request<ConversationOut>(`/workspaces/${workspaceId}/conversations`, {
    method: "POST",
  });
}

export function listTurns(conversationId: string): Promise<TurnOut[]> {
  return request<TurnOut[]>(`/conversations/${conversationId}/turns`);
}

export function askQuestion(conversationId: string, body: AskRequest): Promise<TurnOut> {
  return request<TurnOut>(`/conversations/${conversationId}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getTurn(turnId: string): Promise<TurnOut> {
  return request<TurnOut>(`/turns/${turnId}`);
}

export function getChunk(chunkId: string): Promise<ChunkOut> {
  return request<ChunkOut>(`/chunks/${chunkId}`);
}

export function listEvalRuns(): Promise<EvalRunOut[]> {
  return request<EvalRunOut[]>("/eval/runs");
}

export function listRedTeamResults(): Promise<RedTeamResultOut[]> {
  return request<RedTeamResultOut[]>("/eval/red-team-results");
}
