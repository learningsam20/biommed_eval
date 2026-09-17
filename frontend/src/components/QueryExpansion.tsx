import { useState } from 'react';

export default function QueryExpansion({ original, expansions }: { original: string; expansions: string[] }) {
  const [open, setOpen] = useState(true);
  const n = expansions.length + 1;

  return (
    <div className="rounded-xl border border-accent-200 bg-accent-50/70 p-3">
      <button onClick={() => setOpen(v => !v)} className="flex w-full items-center justify-between text-left">
        <span className="flex items-center gap-2 text-xs font-semibold text-accent-700">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
          {expansions.length
            ? `Query expanded into ${n} searches`
            : 'Original query'}
        </span>
        <span className={`text-accent-600 transition-transform ${open ? 'rotate-180' : ''}`}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <path d="m6 9 6 6 6-6" />
          </svg>
        </span>
      </button>
      {open && (
        <div className="mt-3 space-y-1.5 text-[13px]">
          <div className="flex items-start gap-2">
            <span className="chip shrink-0 border-accent-300 bg-white text-accent-700">original</span>
            <span className="text-ink-soft leading-relaxed">{original}</span>
          </div>
          {expansions.map((e, i) => (
            <div key={i} className="flex items-start gap-2">
              <span className="chip shrink-0 border-accent-200 bg-accent-100 text-accent-700">alt {i + 1}</span>
              <span className="text-ink-soft leading-relaxed">{e}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
