import axios from 'axios';
import type { AnswerResponse, EvidenceItem, GenerateResponse, SearchResponse } from '../types';

function apiBase(): string {
  if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL as string;
  const host = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
  return `http://${host}:5268`;
}

export const api = axios.create({ baseURL: `${apiBase()}/api` });

export const search = (
  query: string,
  mode = 'hybrid',
  fusion = 'weighted',
  top_k = 20,
  use_expansion = false,
) =>
  api.post<SearchResponse>('/search', { query, mode, fusion, top_k, use_expansion }).then(r => r.data);

export const generate = (query: string, evidence: EvidenceItem[]) =>
  api.post<GenerateResponse>('/generate', { query, evidence }).then(r => r.data);

/** One-shot search + answer (slower; prefer search → generate for progressive UI). */
export const answer = (
  query: string,
  mode = 'hybrid',
  fusion = 'weighted',
  top_k = 20,
  use_expansion = false,
) =>
  api.post<AnswerResponse>('/answer', { query, mode, fusion, top_k, use_expansion }).then(r => r.data);
