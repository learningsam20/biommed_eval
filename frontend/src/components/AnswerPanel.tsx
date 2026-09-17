import { Fragment } from 'react';

function renderAnswer(text: string, citations: number[]) {
  const tokens: React.ReactNode[] = [];
  const regex = /(\*\*[^*]+\*\*|\[(\d+)\])/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) tokens.push(text.slice(last, m.index));
    if (m[2]) {
      const pid = Number(m[2]);
      const idx = citations.indexOf(pid);
      tokens.push(
        <a
          key={`c${key++}`}
          href={idx > -1 ? `#ev-${pid}` : undefined}
          onClick={idx > -1 ? e => { e.preventDefault(); document.getElementById(`ev-${pid}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); } : undefined}
          className="group relative mx-0.5 inline-flex items-center translate-y-[-2px] rounded-md border border-clinical-200 bg-clinical-50 px-1.5 py-0.5 text-[10.5px] font-semibold text-clinical-700 transition-colors hover:bg-clinical-100 hover:border-clinical-300 cursor-pointer"
          title={`Passage ${pid}`}
        >
          {pid}
        </a>
      );
    } else {
      const inner = m[1].slice(2, -2);
      tokens.push(<strong key={`b${key++}`} className="font-semibold text-ink">{inner}</strong>);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) tokens.push(text.slice(last));
  return tokens;
}

export default function AnswerPanel({ answer, insufficient, citations }: { answer: string; insufficient: boolean; citations: number[] }) {
  if (insufficient) {
    return (
      <div className="flex items-start gap-3 rounded-xl2 border border-amber-300 bg-amber-50 p-4">
        <div className="shrink-0 mt-0.5 text-amber-500">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <path d="M12 9v4M12 17h.01" />
          </svg>
        </div>
        <div>
          <p className="font-semibold text-amber-800">Insufficient evidence</p>
          <p className="text-sm text-amber-700 mt-0.5">The retrieved passages do not support an answer to this question. Try a different mode or enable query expansion.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fade-in">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="chip border-clinical-300 bg-white text-clinical-700 font-semibold">Answer</span>
          {citations.length > 0 && (
            <span className="text-xs text-ink-faint">cites {citations.length} passage{citations.length > 1 ? 's' : ''} — click a number to jump</span>
          )}
        </div>
      </div>
      <div className="rounded-xl2 border border-slate-200 bg-white p-5 shadow-card">
        <p className="text-[15px] leading-[1.75] text-ink whitespace-pre-wrap">{renderAnswer(answer, citations)}</p>
      </div>
    </div>
  );
}