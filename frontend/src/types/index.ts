export interface EvidenceItem {
  passage_id: number;
  text: string;
  score: number;
  signal: string;
  source_url?: string;
}
export interface SearchResponse {
  original_query: string;
  expansions: string[];
  results: EvidenceItem[];
  latency_ms: number;
  config: Record<string, unknown>;
}
export interface AnswerResponse {
  answer: string;
  citations: number[];
  insufficient_evidence: boolean;
  evidence: EvidenceItem[];
  expansions: string[];
  latency_ms: number;
}
export interface GenerateResponse {
  answer: string;
  citations: number[];
  insufficient_evidence: boolean;
  latency_ms: number;
}
