import { useState } from 'react';
import type { EvidenceItem } from '../types';

const SIGNAL_STYLES: Record<string, string> = {
  lexical: 'border-slate-200 bg-slate-50 text-ink-soft',
  dense: 'border-accent-200 bg-accent-50 text-accent-700',
  hybrid: 'border-clinical-200 bg-clinical-50 text-clinical-700',
};

function EvidenceCard({ item }: { item: EvidenceItem }) {
  const [expanded, setExpanded] = useState(false);
  const text = item.text;
  const clipped = text.length > 480;
  const shown = expanded || !clipped ? text : text.slice(0, 480);

  return (
    <li id={`ev-${item.passage_id}`} className="scroll-mt-24 rounded-xl border border-slate-200 bg-white p-4 shadow-card transition-shadow hover:shadow-lift">
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <a
          href={`#ev-${item.passage_id}`}
          className="inline-flex items-center gap-1.5 rounded-md bg-slate-100 px-2 py-0.5 font-mono text-xs font-semibold text-ink-soft transition-colors hover:bg-clinical-100 hover:text-clinical-700"
        >
          #{item.passage_id}
        </a>
        <span className={`chip ${SIGNAL_STYLES[item.signal] ?? SIGNAL_STYLES.hybrid}`}>
          {item.signal}
        </span>
        <span className="text-xs text-ink-faint font-mono">{item.score.toFixed(4)}</span>
        {item.source_url && (
          <a href={item.source_url} target="_blank" rel="noreferrer" className="ml-auto text-xs text-accent-600 hover:text-accent-700 hover:underline">
            source ↗
          </a>
        )}
      </div>
      <p className="text-[13.5px] leading-relaxed text-ink-soft whitespace-pre-wrap">{shown}</p>
      {clipped && (
        <button
          onClick={() => setExpanded(v => !v)}
          className="mt-2 text-xs font-semibold text-clinical-600 hover:text-clinical-700"
        >
          {expanded ? 'Show less ↑' : `Show full passage (${(text.length / 1000).toFixed(1)}k chars) ↓`}
        </button>
      )}
    </li>
  );
}

export default function EvidencePanel({ items }: { items: EvidenceItem[] }) {
  const [allOpen, setAllOpen] = useState(false);
  const shown = allOpen ? items : items.slice(0, 6);

  if (!items.length) {
    return <p className="text-sm text-ink-faint">No passages retrieved.</p>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-ink">Evidence</span>
          <span className="chip border-slate-200 bg-white text-ink-soft">{items.length} passages</span>
          <span className="text-xs text-ink-faint">{allOpen ? 'all' : 'first 6'} · ranked</span>
        </div>
        {items.length > 6 && (
          <button onClick={() => setAllOpen(v => !v)} className="text-xs font-semibold text-clinical-600 hover:text-clinical-700">
            {allOpen ? 'Collapse' : `Expand all`}
          </button>
        )}
      </div>
      <ol className="space-y-3">
        {shown.map(item => <EvidenceCard key={item.passage_id} item={item} />)}
      </ol>
    </div>
  );
}