import { useState } from 'react';
import Header from './components/Header';
import SearchInput from './components/SearchInput';
import QueryExpansion from './components/QueryExpansion';
import EvidencePanel from './components/EvidencePanel';
import AnswerPanel from './components/AnswerPanel';
import ConfigPanel, { type SearchConfig } from './components/ConfigPanel';
import { search, generate } from './services/api';
import type { AnswerResponse } from './types';

function LoadingState({ label }: { label: string }) {
  return (
    <div className="space-y-4 fade-in" aria-label={label}>
      <div className="rounded-xl2 border border-slate-200 bg-white p-5 shadow-card">
        <div className="skeleton-line h-3 w-24 mb-4" />
        <div className="skeleton-line h-4 w-full mb-2" />
        <div className="skeleton-line h-4 w-11/12 mb-2" />
        <div className="skeleton-line h-4 w-3/4" />
      </div>
      <div>
        <div className="skeleton-line h-3 w-20 mb-3" />
        <div className="space-y-2.5">
          {[0, 1, 2].map(i => (
            <div key={i} className="rounded-xl border border-slate-200 bg-white p-4 shadow-card">
              <div className="skeleton-line h-3 w-24 mb-3" />
              <div className="skeleton-line h-3.5 w-full mb-1.5" />
              <div className="skeleton-line h-3.5 w-4/5" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  // Expansion off by default — LLM expansion dominates latency (~20s local)
  const [cfg, setCfg] = useState<SearchConfig>({
    mode: 'hybrid', fusion: 'weighted', top_k: 20, use_expansion: false,
  });
  const [res, setRes] = useState<AnswerResponse | null>(null);
  const [lastQuery, setLastQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [error, setError] = useState('');

  const run = async (q: string) => {
    setSearching(true);
    setAnswering(false);
    setError('');
    setLastQuery(q);
    setRes(null);
    try {
      // Phase 1: retrieval only — evidence appears immediately
      const sr = await search(q, cfg.mode, cfg.fusion, cfg.top_k, cfg.use_expansion);
      setRes({
        answer: '',
        citations: [],
        insufficient_evidence: false,
        evidence: sr.results,
        expansions: sr.expansions,
        latency_ms: sr.latency_ms,
      });
      setSearching(false);
      setAnswering(true);
      // Phase 2: grounded answer from retrieved passages (no re-search)
      const gr = await generate(q, sr.results);
      setRes(prev => prev && {
        ...prev,
        answer: gr.answer,
        citations: gr.citations,
        insufficient_evidence: gr.insufficient_evidence,
        latency_ms: prev.latency_ms + gr.latency_ms,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setRes(null);
    } finally {
      setSearching(false);
      setAnswering(false);
    }
  };

  const loading = searching && !res;

  return (
    <div className="min-h-screen">
      <Header />
      <main className="max-w-4xl mx-auto px-4 py-8 space-y-5">
        <div className="space-y-4">
          <SearchInput onSearch={run} loading={searching || answering} />
          <ConfigPanel cfg={cfg} setCfg={setCfg} />
        </div>

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 fade-in">
            <span className="font-semibold">Request failed. </span>
            {error}
          </div>
        )}

        {loading && <LoadingState label="Searching" />}

        {!loading && !res && !error && (
          <div className="text-center text-sm text-ink-faint pt-10 space-y-1 fade-in">
            <p className="text-lg text-ink-soft font-medium">Search 40,221 BioASQ passages</p>
            <p>BM25 · dense vectors · hybrid fusion · grounded, cited answers</p>
          </div>
        )}

        {!loading && res && (
          <div className="space-y-6">
            {answering ? (
              <div className="rounded-xl2 border border-slate-200 bg-white p-5 shadow-card fade-in">
                <div className="skeleton-line h-3 w-28 mb-4" />
                <div className="skeleton-line h-4 w-full mb-2" />
                <div className="skeleton-line h-4 w-10/12" />
                <p className="mt-3 text-xs text-ink-faint">Writing grounded answer…</p>
              </div>
            ) : (
              <AnswerPanel answer={res.answer} insufficient={res.insufficient_evidence} citations={res.citations} />
            )}

            <QueryExpansion original={lastQuery} expansions={res.expansions} />
            <p className="text-xs text-ink-faint -mt-2 text-right">
              {res.latency_ms.toFixed(0)} ms · mode {cfg.mode}
              {cfg.mode === 'hybrid' ? ` / ${cfg.fusion}` : ''} · top-{cfg.top_k}
              {cfg.use_expansion ? ' · expansion' : ''}
            </p>

            <EvidencePanel items={res.evidence} />
          </div>
        )}
      </main>
    </div>
  );
}
