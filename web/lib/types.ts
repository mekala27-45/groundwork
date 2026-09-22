// Mirrors packages/api/src/groundwork_api/schemas.py exactly, field for
// field. There is no code generation step between the two: the API has
// no OpenAPI client generator wired in (a static export has no build
// time access to a running API to generate against), so this file is
// hand kept in sync the same way the rest of this project treats a
// contract, deliberately, not accidentally. A field added there and
// missed here fails at the point this file's types are actually used,
// not silently.

export type ChunkStrategy = "naive" | "structure";
export type ExtractionMethod = "text" | "ocr" | "mixed";
export type NliLabel = "entailed" | "contradicted" | "unsupported";

export interface DocumentOut {
  id: string;
  filename: string;
  page_count: number;
  extraction_method: ExtractionMethod;
  uploaded_at: string;
}

export interface ChunkOut {
  id: string;
  document_id: string;
  workspace_id: string;
  strategy: ChunkStrategy;
  text: string;
  page_start: number;
  page_end: number;
  section_title: string | null;
  injection_flag: string | null;
}

export interface WorkspaceOut {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
}

export interface WorkspaceDetailOut extends WorkspaceOut {
  documents: DocumentOut[];
  naive_chunk_count: number;
  structure_chunk_count: number;
}

export interface ConversationOut {
  id: string;
  workspace_id: string;
  created_at: string;
}

export interface AskRequest {
  question: string;
  strategy: ChunkStrategy;
  use_reranking: boolean;
}

export interface ClaimOut {
  text: string;
  cited_chunk_id: string | null;
  nli_label: string;
  score: number;
}

export interface CitationVerificationOut {
  chunk_id: string;
  verified: boolean;
  reason: string | null;
}

export interface TurnOut {
  id: string;
  conversation_id: string;
  workspace_id: string;
  question: string;
  retrieved_chunk_ids: string[];
  reranked: boolean;
  chunking_strategy: ChunkStrategy | null;
  answer: string;
  claims: ClaimOut[];
  citation_verifications: CitationVerificationOut[];
  extractive_fallback: boolean;
  judge_scores: Record<string, unknown> | null;
  latency_ms: number;
  cost_usd: number;
  created_at: string;
}

export interface EvalRunOut {
  id: string;
  run_id: string;
  run_at: string;
  embedding_backend: string;
  rerank_backend: string;
  config_label: string;
  workspace_id: string;
  category: string;
  recall_at_3: number;
  recall_at_5: number;
  precision_at_5: number;
  mrr: number;
  n_questions: number;
}

export interface RedTeamResultOut {
  id: string;
  run_id: string;
  run_at: string;
  suite: "injection" | "out_of_scope" | "workspace_isolation";
  case_id: string;
  passed: boolean;
  detail: string | null;
  turn_id: string | null;
}

export interface ErrorOut {
  detail: string;
}
