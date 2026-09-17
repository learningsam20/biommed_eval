import { useEffect, useState } from 'react';
import { api } from '../services/api';

export default function Header() {
  const [vectorDb, setVectorDb] = useState('vector DB');
  useEffect(() => {
    api.get('/config').then(r => {
      if (r.data?.vector_db) setVectorDb(String(r.data.vector_db));
    }).catch(() => {});
  }, []);

  return (
    <header className="border-b border-slate-200/80 bg-white/80 backdrop-blur sticky top-0 z-40">
      <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-clinical-500 to-clinical-700 flex items-center justify-center shadow-card">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 3v18M3 12h18" />
              <path d="M7 7l10 10M17 7L7 17" />
            </svg>
          </div>
          <div>
            <h1 className="text-base font-bold text-ink leading-tight">BioMed Search</h1>
            <p className="text-[11px] text-ink-faint leading-tight">RAG answers with cited evidence · 40,221 BioASQ passages</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="chip border-clinical-200 bg-clinical-50 text-clinical-700">
            <span className="w-1.5 h-1.5 rounded-full bg-clinical-500 animate-pulse" />
            {vectorDb}
          </span>
          <span className="chip hidden sm:inline-flex border-slate-200 bg-white text-ink-soft">
            BM25 + Dense
          </span>
        </div>
      </div>
    </header>
  );
}
