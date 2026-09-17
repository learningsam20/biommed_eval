# BioMed Hybrid Search — Benchmark & Evaluation Project Specification

## 0. Purpose

Build a reproducible biomedical retrieval + grounded question-answering benchmark on top of
`rag-datasets/rag-mini-bioasq` (Hugging Face).

The system must:

1. Retrieve useful passages with three retrieval signals: lexical (BM25), dense (embeddings + vector DB), hybrid (fusion).
2. Expand every user query into a small set of alternate queries before retrieval, retaining the original.
3. Show evidence to the user with passage IDs, scores, and source links where available.
4. Generate a final LLM answer grounded strictly in retrieved passages with inline citations.
5. Evaluate all configurations offline on a fixed 100-query set with traditional retrieval metrics and LLM-judge metrics.
6. Select a best hybrid configuration backed by experiment results, with latency / cost trade-offs documented.

This spec defines scope, architecture, interfaces, rules, and ways of working. It is written as a
benchmarking spec — no external context is referenced.

---

## 1. Dataset Contract

Source: `https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq`

Two configs:

### 1.1 `text-corpus` / split `passages` — 40,221 rows
| field | type | notes |
|---|---|---|
| `id` | int | passage identifier, citation unit |
| `passage` | str | passage text |

### 1.2 `question-answer-passages` / split `test` — 4,719 rows
| field | type | notes |
|---|---|---|
| `id` | int | query identifier |
| `question` | str | biomedical question |
| `answer` | str | reference answer |
| `relevant_passage_ids` | list[int] | gold passage IDs for recall-based metrics |

Constraints:
- Same indexed passage set must be used for all retrieval modes.
- Gold `relevant_passage_ids` are used only for scoring, never for retrieval or answer construction.
- Fixed 100-query eval subset is sampled once (seeded) and frozen as JSON.

---

## 2. Tech Stack

Providers are selected in `.env`. Defaults below; swap without changing retrieval/generation code.

- Backend: Python 3.11+, FastAPI, Uvicorn, Pydantic v2 + pydantic-settings
- Retrieval: `rank-bm25` (lexical), `sentence-transformers` (embeddings), vector DB abstraction
- Vector DB: `VECTOR_DB=pinecone|qdrant`. Default Pinecone serverless (managed 40k-vector index, free-tier friendly). Qdrant for fully local runs.
- Embeddings: `EMBEDDING_MODEL` (default `BAAI/bge-base-en-v1.5`, 768-d). Strong English retrieval at modest encode cost; re-index after a swap.
- LLM: `LLM_PROVIDER=openrouter|ollama`, same client for expansion, answers, and judge. OpenRouter for hosted models; Ollama for local/$0.
- Fusion: weighted (default; won nDCG here) or RRF.
- Eval: custom retrieval metrics + RAGAS / DeepEval-compatible LLM-judge interface
- Frontend: React 18 + Vite + TypeScript + Tailwind CSS + TanStack Query + axios
- Infra: `docker-compose.yml` (optional local Qdrant), `Makefile`, `.env` for all config

New vector DB or LLM backends need a factory implementation plus this spec update.

---

## 3. System Architecture

### 3.1 Offline data path
```
HF dataset load -> passage normalisation -> cache (data/passages.jsonl, qa_pairs.jsonl)
  -> BM25 index build (persisted pickle)
  -> embedding encode (batched) -> vector upsert (Pinecone auto-create index if absent / Qdrant collection)
```

### 3.2 Online query path
```
user query
 -> query expansion (LLM, N alternates + original retained)
 -> parallel retrieval: BM25 top LEXICAL_TOP_K + dense top DENSE_TOP_K
 -> hybrid fusion: RRF (k=RRF_K) AND weighted (WEIGHT_BM25/WEIGHT_DENSE) — both implemented
 -> top HYBRID_TOP_K context selection
 -> grounded answer generation (LLM, context-only, inline [id] citations)
 -> citation validation (regex extract, subset check vs retrieved IDs)
 -> insufficient-evidence branch if no usable context
```

### 3.3 User-facing path
```
SearchInput -> POST /api/search -> EvidencePanel (ranked id/score/text/link)
           -> POST /api/answer  -> AnswerPanel (cited answer or insufficient-evidence state)
           -> ConfigPanel (mode, fusion, top_k) + QueryExpansion view
```

### 3.4 Experiment path
```
create_eval_set (seeded 100) -> run_evaluation per config -> per-query JSONL + aggregate JSON
 -> generate_report (comparison table + trade-offs + failure cases)
 -> docs/architecture.md + docs/evaluation_report.md
```

See `docs/architecture.md` for the Mermaid diagram (must show all four paths).

---

## 4. Backend Specification

### 4.1 Module layout
```
backend/app/
  main.py                 FastAPI factory, CORS, lifespan logging
  config.py               Settings (pydantic-settings, .env only)
  models/schemas.py       SearchRequest/Response, AnswerRequest/Response, Eval schemas
  models/vector_store.py  VectorStore protocol + PineconeVectorStore + QdrantVectorStore + factory
  retrieval/bm25.py       BM25LexicalIndex (build/query/persist)
  retrieval/dense.py      DenseRetriever (encode + vector-store search)
  retrieval/hybrid.py     reciprocal_rank_fusion() + weighted_fusion()
  retrieval/expansion.py  expand_query() via LLM
  generation/llm.py       get_llm_client() — OpenAI-compatible (OpenRouter) / Ollama
  generation/answer.py    build_grounded_answer() + validate_citations()
  evaluation/metrics.py   recall@k, mrr@k, ndcg@k, latency aggregation
  evaluation/judge.py     LLM-judge interface (correctness, groundedness, context relevance)
  evaluation/runner.py    run_config() orchestration
  api/search.py           POST /search, /expand, /answer
  api/health.py           GET /health
backend/scripts/
  build_index.py          offline index builder
  create_eval_set.py      frozen 100-query sampler
  run_evaluation.py       multi-config eval runner
  generate_report.py      markdown report generator
```

### 4.2 Vector store abstraction (must-have)
```python
class VectorStore(Protocol):
    def create_index_if_not_exists(self) -> None: ...
    def upsert(self, ids, vectors, metadatas) -> None: ...
    def search(self, vector, top_k) -> list[(str|int, float)]: ...
    def health_check(self) -> bool: ...
```

- `PineconeVectorStore`:
  - Uses `pinecone-client` v3 (`from pinecone import Pinecone, ServerlessSpec`).
  - Serverless: `cloud=PINECONE_CLOUD` (default `aws`), `region=PINECONE_REGION` (default `us-east-1`).
  - `create_index_if_not_exists()`: if `PINECONE_INDEX_NAME` absent, create with `dimension=EMBEDDING_DIM`, `metric=PINECONE_METRIC` (default `cosine`), serverless spec.
  - On startup, if index exists but dimension != `EMBEDDING_DIM`, fail fast with clear error.
  - Batch upserts (100 vectors/batch).
  - Namespace: `PINECONE_NAMESPACE` (default empty / `bioasq`).
- `QdrantVectorStore`: local fallback (`QDRANT_URL`, `QDRANT_COLLECTION`, cosine/HNSW).
- Factory `get_vector_store(settings)` switches on `VECTOR_DB`. No business logic may import a concrete store directly.

### 4.3 LLM abstraction (must-have)
- `LLM_PROVIDER=openrouter`: OpenAI-compatible client with `base_url=OPENROUTER_BASE_URL`, `api_key=OPENROUTER_API_KEY`, `model=OPENROUTER_MODEL`.
- `LLM_PROVIDER=ollama`: Ollama chat endpoint at `OLLAMA_BASE_URL`, `model=OLLAMA_MODEL` (e.g. `llama3.1:8b`).
- Same interface for expansion and answering. Temperature fixed per use-case and recorded in eval logs.
- Query-expansion prompt must request N biomedical alternates as JSON list; always retain original.
- Answer prompt must instruct: answer ONLY from provided passages, cite every factual claim as `[<passage_id>]`, output `INSUFFICIENT_EVIDENCE` sentinel when context is inadequate.

### 4.4 API contract
- `POST /api/search { query, mode: lexical|dense|hybrid, fusion: rrf|weighted, top_k, use_expansion }`
  -> `{ original_query, expansions[], results[{passage_id, text, score, source}], latency_ms, config }`
- `POST /api/expand { query, count }` -> `{ original_query, expansions[] }`
- `POST /api/answer { query, passage_ids?, top_k, mode, fusion }`
  -> `{ answer, citations[], insufficient_evidence: bool, latency_ms }`
- `GET /api/health` -> `{ status }`
- `GET /api/config` -> non-secret current config (never leak API keys)
- Every search must log retrieved IDs + scores for reproducibility.

### 4.5 Citation validation (must-have)
- Extract `\[(\d+)\]` from answer text.
- Assert cited set ⊆ retrieved set; strip or flag hallucinations, never silently pass.
- `insufficient_evidence=true` when: zero results, fusion scores below threshold, or LLM emits sentinel.

---

## 5. Frontend Specification

Stack: React 18 + Vite + TypeScript + Tailwind + TanStack Query. Must run locally (`npm run dev`) and build for static hosting (`npm run build` → Netlify/Vercel).

Required components:
1. `SearchInput.tsx` — biomedical question textarea + mode/fusion/top_k selectors + search action.
2. `QueryExpansion.tsx` — original query + expansion chips (collapsible).
3. `EvidencePanel.tsx` — ranked list: passage ID badge, score badge (with signal label), truncated text with expand, PubMed/source link where derivable (`https://pubmed.ncbi.nlm.nih.gov/<id>/` when ID looks like PMID, else dataset link).
4. `AnswerPanel.tsx` — rendered answer with clickable inline citations (scroll to evidence item), plus explicit `Insufficient evidence — retrieved passages do not support an answer.` empty state.
5. `ConfigPanel.tsx` — retrieval mode, fusion method, top-K, expansion toggle; shows active `.env`-derived backend config (non-secret).
6. `services/api.ts` — typed axios client; `VITE_API_URL` env.
7. States: loading skeletons, error toasts, empty results, citation-hover highlight.

Accessibility: labels on all inputs, keyboard-submit (Cmd/Ctrl+Enter), sufficient contrast.

---

## 6. Evaluation Specification

### 6.1 Frozen 100-query set
- Script: `backend/scripts/create_eval_set.py --output data/eval_queries_100.json --seed 42 --size 100`
- Stratified across surface types: yes/no, definition, list, treatment, mechanism, effect (≈16–17 each, best-effort keyword heuristic + manual spot-check).
- Schema per item: `{ query_id, question, answer, relevant_passage_ids, qtype }`
- Committed to repo; never regenerated during comparison runs.

### 6.2 Configurations under test (all required)
| key | retrieval | expansion |
|---|---|---|
| `lexical` | BM25 only | off |
| `dense` | dense only | off |
| `hybrid_rrf` | hybrid RRF | off |
| `hybrid_weighted` | hybrid weighted | off |
| `hybrid_expansion` | hybrid (configured fusion) | on |
| `best` | winner from above (or tuned weights) | as configured |

Same 100 query IDs for every run. Same judge model/prompt/settings across runs.

### 6.3 Metrics
Traditional (using `relevant_passage_ids`):
- Recall@5, Recall@10, MRR@10, nDCG@10, p50/p95 latency_ms, estimated cost (tokens × price or local-seconds).

LLM-judge (RAGAS/DeepEval-compatible):
- answer correctness (vs reference answer), groundedness (claims ⊆ context), context relevance (retrieved ⊆ question intent), citation validity (code-checked, not LLM-graded).
- Judge prompt, model, temperature frozen and stored alongside results.

Manual review:
- ≥3 success + ≥3 failure cases per best/worst config; disagreement analysis where judge and retrieval metrics diverge.

### 6.4 Scripts
- `python -m scripts.run_evaluation --config all --eval-set data/eval_queries_100.json --output results/`
  Writes local per-query JSON under `results/` (gitignored). Same retrieve path as `POST /api/search`.
- `python -m scripts.generate_report --results-dir results/ --output docs/evaluation_report.md`
  Writes `docs/evaluation_report.md` (comparison table, winner, examples) and `docs/best_hybrid_config.json`.
- All scripts: `--help` documented, exit non-zero on error, deterministic given same `.env` + eval set.

---

## 7. Configuration — `.env` Is the Single Source of Truth

No hardcoded model names, keys, URLs, paths, or hyperparameters in code. Every tunable lives in `.env` (see `.env.example`):

```
LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL,
OLLAMA_BASE_URL, OLLAMA_MODEL,
EMBEDDING_MODEL, EMBEDDING_DIM,
VECTOR_DB,
HF_TOKEN,
PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_REGION, PINECONE_INDEX_NAME, PINECONE_METRIC, PINECONE_NAMESPACE,
QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
BM25_K1, BM25_B, LEXICAL_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
FUSION_METHOD, RRF_K, WEIGHT_BM25, WEIGHT_DENSE,
QUERY_EXPANSION_ENABLED, EXPANSION_COUNT,
EVAL_QUERY_SET, EVAL_SEED,
BACKEND_HOST, BACKEND_PORT, FRONTEND_URL, LOG_LEVEL
```

Rules:
- `.env` is git-ignored; `.env.example` holds placeholders only.
- `GET /api/config` must never return secrets.
- Changing embedding model requires re-indexing; changing `EMBEDDING_DIM` without re-creating the Pinecone index must fail fast.

---

## 8. Must-Haves (Acceptance Checklist)

- [ ] BM25, dense, hybrid-RRF, hybrid-weighted, hybrid+expansion all implemented and callable via API + eval runner
- [ ] Query expansion retains original + N alternates, displayed in UI
- [ ] UI shows: input, expansions, ranked evidence (id/score/source), cited answer, insufficient-evidence state
- [ ] Final answer cites only retrieved IDs; validator enforced in code path (not just prompt)
- [ ] Frozen 100-query eval set committed; single-command eval + report generation
- [ ] Recall@5/10, MRR@10, nDCG@10, latency, cost reported per config
- [ ] LLM-judge (correctness, groundedness, context relevance) with frozen settings + citation-validity code check
- [ ] Comparison table + winner + trade-offs + success/failure cases in `docs/evaluation_report.md`
- [ ] Mermaid system diagram in `docs/architecture.md` + design-choice rationale in README
- [ ] All config via `.env`; `requirements.txt` pinned; Docker Compose for optional local Qdrant
- [ ] Pinecone serverless auto-create with dimension check; Qdrant fallback intact

---

## 9. Dos

- Do use type hints, Pydantic schemas, and dependency injection throughout backend.
- Do make retrieval deterministic given same `.env` + corpus (seeded eval, sorted tie-breaks).
- Do batch embedding encodes and vector upserts; cache embeddings to avoid recompute.
- Do use `async` for I/O (LLM, vector DB); sync only for CPU-bound BM25 encode where justified.
- Do log query → expansions → IDs/scores → fusion weights → answer → citations per request (JSON lines).
- Do write unit tests for `hybrid.py` (RRF/weighted), `metrics.py`, `answer.py` citation validator.
- Do document every public endpoint in OpenAPI (FastAPI automatic) + example curl in README.
- Do keep frontend and backend strictly decoupled via `VITE_API_URL`.
- Do pin all dependencies and record judge model/prompt/temperature in eval outputs.

## 10. Don'ts

- Don't hardcode keys, model names, URLs, file paths, or hyperparameters.
- Don't answer from parametric knowledge — final LLM call receives only retrieved passages.
- Don't emit citations that were not retrieved; don't suppress the insufficient-evidence state.
- Don't vary judge settings between configs; don't grade citation validity with the LLM (code-check it).
- Don't leak secrets in logs, API responses, or committed files.
- Don't use global mutable config; always inject `Settings`.
- Don't regenerate the 100-query set between runs; don't mix gold IDs into retrieval.
- Don't commit `.env`, `data/*.jsonl` caches over 50 MB, or `results/` large artifacts without LFS consideration.
- Don't introduce a new vector DB / LLM provider without updating spec §4.2–4.3 and `.env.example`.

---

## 11. Way of Working

1. Phase 1 — Data & indexes: HF load, cache, BM25 pickle, Pinecone auto-create + upsert, `build_index.py` green.
2. Phase 2 — Retrieval core: `bm25.py`, `dense.py`, `hybrid.py` (both fusions) + unit tests.
3. Phase 3 — Generation: `llm.py` (OpenRouter + Ollama), `expansion.py`, `answer.py` + citation validator.
4. Phase 4 — Eval pipeline: `create_eval_set.py`, `metrics.py`, `judge.py`, `runner.py`, `run_evaluation.py`, `generate_report.py`.
5. Phase 5 — Frontend: Vite scaffold, five components, typed API client, local + build verified.
6. Phase 6 — Integration: end-to-end search → evidence → cited answer; full 100-query runs; report + failure analysis.
7. Phase 7 — Docs & release: architecture diagram, README rationale, pinned requirements, Compose cleanup.

Branch convention: `feat/<module>`, `fix/<issue>`; commit after each phase with eval artifacts hashes logged.

---

## 12. Dependencies

Backend (`backend/requirements.txt`, Python 3.11+):
```
fastapi, uvicorn[standard], pydantic, pydantic-settings, python-dotenv,
httpx, rank-bm25, sentence-transformers, torch (via st),
pinecone-client>=3.0.0, qdrant-client,
openai (OpenRouter-compatible), ollama (optional),
ragas, deepeval, datasets, huggingface-hub,
numpy, pandas, tqdm, pyyaml, orjson, tenacity, pytest
```
Frontend (`frontend/package.json`, Node 20+): `react, react-dom, axios, @tanstack/react-query` + dev `vite, typescript, tailwindcss, postcss, autoprefixer, eslint, @vitejs/plugin-react`.

---

## 13. Commands

```bash
cp .env.example .env            # fill secrets
make backend-install            # pip install -r backend/requirements.txt
make frontend-install           # npm install
make qdrant-up                  # only needed if VECTOR_DB=qdrant
python -m scripts.build_index   # build BM25 + vector index
make frontend-dev / make backend-dev
python -m scripts.create_eval_set --output data/eval_queries_100.json --seed 42 --size 100
python -m scripts.run_evaluation --config all --eval-set data/eval_queries_100.json --output results/
python -m scripts.generate_report --results-dir results/ --output docs/evaluation_report.md
```

---

## 14. Success Criteria

| criterion | target |
|---|---|
| 6 configs evaluated on identical 100 queries | yes, `results/summary.json` |
| reproducibility | one command per stage, seeded |
| UI local | `npm run dev` + `uvicorn` end-to-end |
| UI deployable | `npm run build` artifact on Netlify/Vercel |
| metrics | 4 retrieval + latency + cost + 3 judge + citation-validity |
| report | comparison table + winner + trade-offs + examples |
| docs | Mermaid diagram + README rationale |

---

## 15. Risks & Mitigations

- Pinecone free-tier limits (records/RPS) → batch upserts, backoff via tenacity, document index size (40k × dim).
- Embedding dim mismatch → fail-fast check at startup vs live index.
- LLM cost during 100-query × 6-config judge → cache answers, judge only `best` vs baselines if budget-constrained (must document).
- Biomedical jargon recall → query expansion + hybrid fusion; failure cases retained in report.
