import { useState } from 'react';

export default function SearchInput({ onSearch, loading }: { onSearch: (q: string) => void; loading: boolean }) {
  const [q, setQ] = useState('');
  const submit = () => { if (q.trim() && !loading) onSearch(q); };

  return (
    <form
      onSubmit={e => { e.preventDefault(); submit(); }}
      className="relative"
    >
      <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="11" cy="11" r="7" />
          <path d="m21 21-4.3-4.3" />
        </svg>
      </div>
      <label htmlFor="biomed-q" className="sr-only">Biomedical question</label>
      <input
        id="biomed-q"
        type="text"
        value={q}
        onChange={e => setQ(e.target.value)}
        placeholder="Ask a biomedical question — e.g. How many times is CLAST faster than BLAST?"
        className="search-bar pr-28"
        autoFocus
      />
      <button
        type="submit"
        disabled={loading || !q.trim()}
        className="btn-primary absolute right-2 top-1/2 -translate-y-1/2 !py-2 !px-4 !rounded-lg"
      >
        {loading ? (
          <>
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
            Searching
          </>
        ) : (
          'Search'
        )}
      </button>
    </form>
  );
}