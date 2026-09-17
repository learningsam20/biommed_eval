import { useRef } from 'react';

export interface SearchConfig {
  mode: string;
  fusion: string;
  top_k: number;
  use_expansion: boolean;
}

const MODES = [
  { value: 'hybrid', label: 'Hybrid' },
  { value: 'lexical', label: 'Lexical (BM25)' },
  { value: 'dense', label: 'Dense' },
];
const FUSIONS = [
  { value: 'weighted', label: 'Weighted' },
  { value: 'rrf', label: 'RRF' },
];

export default function ConfigPanel({ cfg, setCfg }: { cfg: SearchConfig; setCfg: (c: SearchConfig) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const onChange = (patch: Partial<SearchConfig>) => setCfg({ ...cfg, ...patch });

  const baseSelect = 'rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-sm text-ink-soft outline-none transition-colors focus:border-clinical-400 focus:ring-2 focus:ring-clinical-100 hover:border-slate-300';

  return (
    <div ref={ref} className="flex items-center gap-2 flex-wrap text-sm">
      <span className="text-xs font-semibold uppercase tracking-wider text-ink-faint mr-1">Retrieval</span>

      <label className="flex items-center gap-1.5 text-ink-soft">
        <span className="text-xs font-medium">Mode</span>
        <select value={cfg.mode} onChange={e => onChange({ mode: e.target.value })} className={baseSelect}>
          {MODES.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
      </label>

      {cfg.mode === 'hybrid' && (
        <label className="flex items-center gap-1.5 text-ink-soft">
          <span className="text-xs font-medium">Fusion</span>
          <select value={cfg.fusion} onChange={e => onChange({ fusion: e.target.value })} className={baseSelect}>
            {FUSIONS.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}
          </select>
        </label>
      )}

      <label className="flex items-center gap-1.5 text-ink-soft">
        <span className="text-xs font-medium">Top-K</span>
        <input
          type="number" value={cfg.top_k} min={5} max={50}
          onChange={e => onChange({ top_k: Number(e.target.value) })}
          className={`${baseSelect} w-20`}
        />
      </label>

      <label className="ml-1 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 cursor-pointer select-none transition-colors hover:border-clinical-300">
        <input
          type="checkbox"
          checked={cfg.use_expansion}
          onChange={e => onChange({ use_expansion: e.target.checked })}
          className="h-3.5 w-3.5 rounded accent-clinical-600"
        />
        <span className="text-xs font-medium text-ink-soft">Query expansion</span>
        <span className="text-[10px] text-ink-faint">(slower)</span>
      </label>
    </div>
  );
}