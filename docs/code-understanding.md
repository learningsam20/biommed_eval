# How the code works

This is a walkthrough of the BioMed Hybrid Search codebase: request flow, retrieval, grounded answers, and how evaluation metrics are calculated. Architecture (four paths) lives in [architecture.md](architecture.md). Numbers from the 100-query run live in [evaluation_report.md](evaluation_report.md).

Gold `relevant_passage_ids` are **never** used at query time. They appear only in `backend/app/evaluation/metrics.py` after retrieval.

## Map

| Path | What it is |
|---|---|
| `frontend/src/` | Search UI (question, evidence, answer / insufficient) |
| `backend/app/main.py` | FastAPI app; loads BM25, vector store, LLM into memory |
| `backend/app/api/search.py` | `/api/search`, `/api/answer`, `/api/generate`, `/api/expand` |
| `backend/app/retrieval/` | BM25, dense, fusion, query expansion |
| `backend/app/generation/` | LLM client + citation-validated answers |
| `backend/app/evaluation/` | Recall/MRR/nDCG + custom / RAGAS / DeepEval judges |
| `backend/scripts/build_index.py` | Offline: pickle BM25, embed corpus, upsert vectors |
| `backend/scripts/run_evaluation.py` | Same retrieve path over frozen 100 queries |
| `backend/scripts/generate_report.py` | Comparison table + winner recipe |

`.env` (`backend/app/config.py`) is the only place to swap vector DB, embedder, LLM, fusion weights, and `JUDGE_FRAMEWORK`.

## What happens on a UI search

```
UI  →  POST /api/search   (evidence only)
    →  POST /api/generate (answer from those passages)
```

`POST /api/answer` does both in one call (used by eval and scripts).

1. **`_retrieve`** (`api/search.py`)
   - If expansion is on: LLM produces alternate queries; original is always kept (`expansion.expand_query`).
   - BM25 and dense search run **in parallel** (thread pool).
   - Hits from several expanded queries are **deduped** by best score (`hybrid.dedupe_by_best_score`).
   - Mode:
     - `lexical` → sort BM25 scores
     - `dense` → sort cosine scores
     - `hybrid` + `weighted` → min-max each list, then `0.5 * bm25 + 0.5 * dense`
     - `hybrid` + `rrf` → `1 / (k + rank)` votes from each list (`k=60`)
2. Passage texts are fetched from Pinecone/Qdrant (`fetch_texts`) and returned as evidence (ID, score, PubMed URL).
3. **`generate_grounded_answer`** (`generation/answer.py`) asks the LLM to answer **only** from those passages with `[passage_id]` cites. Then **code** (not the LLM) checks:
   - empty evidence, empty text, or the string `INSUFFICIENT_EVIDENCE` → insufficient
   - any cited ID not in the retrieved set → insufficient (hallucinated cite)
   - no valid cites at all → insufficient
   - otherwise the raw answer plus the valid ID list

Interactive UI defaults: **weighted hybrid**, expansion **off** (~80 ms). Expansion is an extra LLM round-trip (~27 s in the 100-query expansion run).

## Offline index (once)

`make index` → `scripts/build_index.py`:

1. Load Hugging Face `rag-mini-bioasq` `text-corpus` / `passages` (~40k).
2. Fit Okapi BM25 (`k1=1.2`, `b=0.75`), pickle to `data/bm25_index.pkl`.
3. Cache `data/passages.jsonl`.
4. Embed with `BAAI/bge-base-en-v1.5` (L2-normalized) and upsert into Pinecone or Qdrant.

The API lifespan loads that pickle and the same embedder + store. Missing indexes → empty search results, not a crash.

## Evaluation runner

`make eval-run` uses **the same retrieve functions** as the API, not a separate ranker.

For each config (`lexical`, `dense`, `hybrid_rrf`, `hybrid_weighted`, `hybrid_expansion`, `best`) and each frozen query:

1. `make_retrieve_fn` returns `[(passage_id, score), ...]`.
2. Wall-clock of that call is `latency_ms`.
3. `score_query(retrieved_ids, gold_ids, latency)` writes Recall / MRR / nDCG.
4. `aggregate` macro-averages those rows and adds p50/p95 latency.

`best` is an alias of **weighted hybrid + expansion** (same recipe as `hybrid_expansion`).

`make eval-judge --limit 20` then, for each judged query:

1. Fetch texts for the top passages (capped at 5 × 4000 chars for the judge).
2. Generate a grounded answer (same citation rules as the UI).
3. Score it with every framework in `JUDGE_FRAMEWORK` (`custom`, `ragas`, `deepeval`, or `all`).
4. Merge judge fields into `results/judge/` **without** overwriting the 100-query rank JSON.

Per-query judge budget: 120 s; leftover frameworks are skipped if fewer than 8 s remain.

`make eval-report` reads `results/summary.json` and writes the markdown table plus `docs/best_hybrid_config.json`.

## Traditional metrics

Implemented in `backend/app/evaluation/metrics.py`. Relevance is **binary**: a retrieved ID is either in the gold set or not. There is no graded gain.

Let \(R\) be the gold ID set, \(L_k\) the first \(k\) retrieved IDs (order preserved).

### Recall@k

\[
\mathrm{Recall@}k = \frac{|L_k \cap R|}{|R|}
\]

Zero if \(R\) is empty. Measures coverage of gold passages, not rank quality beyond “in the window.” We report **@5** and **@10**.

### MRR@10

\[
\mathrm{MRR@}10 =
\begin{cases}
1 / r & \text{first gold hit is at rank } r \le 10 \\
0 & \text{no gold in top-10}
\end{cases}
\]

High when a relevant passage is near the top.

### nDCG@10

Gain is 1 for a gold ID, 0 otherwise. Discount is \(\log_2(\mathrm{rank}+1)\).

\[
\mathrm{DCG@}k = \sum_{r=1}^{k} \frac{\mathbf{1}[L[r] \in R]}{\log_2(r+1)}
\]

\[
\mathrm{IDCG@}k = \sum_{r=1}^{\min(|R|,k)} \frac{1}{\log_2(r+1)}
\]

\[
\mathrm{nDCG@}k = \mathrm{DCG@}k / \mathrm{IDCG@}k
\]

(0 if IDCG is 0.) This is the **primary** rank metric. The selected config is **max nDCG@10, then Recall@10**.

### Latency

Retrieve-path wall time in milliseconds (expansion + BM25 + dense + fusion). Answer generation is **not** included in the 100-query table latency.

Percentiles: sorted latencies, p50 = middle value, p95 = value at index \(\lfloor 0.95 n \rfloor\).

### Cost

`generation/llm.py` `UsageAccumulator`:

- **Ollama:** token counts recorded, `cost_usd = 0`.
- **OpenRouter:** `(prompt_tokens * $0.50 + completion_tokens * $1.50) / 1e6` as a ballpark when the API does not return a bill.

Eval stores `cost_usd_per_query` and `tokens_per_query` on each config.

## LLM-judge (20 queries)

Same three 0–1 numbers on every framework, **fixed** prompt / model / temperature across configs (`JUDGE_TEMPERATURE=0.0`). Mapping:

| Our key | Custom | RAGAS 0.4.3 | DeepEval 4.2.3 |
|---|---|---|---|
| correctness | JSON `correctness` | `FactualCorrectness(mode=precision)` | `AnswerRelevancyMetric` |
| groundedness | JSON `groundedness` | `Faithfulness` | `FaithfulnessMetric` |
| context_relevance | JSON `context_relevance` | `ContextRelevance` | `ContextualRelevancyMetric` |

**Custom** (`judge.run_custom_judge`): one JSON completion. Parse failure → 0, 0, 0.

**RAGAS** (`ragas_judge.score_ragas_one`):

- **Precision** FactualCorrectness = fraction of **answer claims** that NLI supports against the **gold answer** (not F1). Default RAGAS F1 is 0 when a short RAG answer misses extra gold claims.
- Faithfulness = fraction of answer claims supported by **retrieved context**.
- ContextRelevance = whether retrieved passages are relevant to the question.
- `INSUFFICIENT_EVIDENCE` vs a real gold → precision 0. Empty claims / NLI failure → 0. Errors are caught and recorded as 0.
- Uses a **reused asyncio loop + cached AsyncOpenAI** so `asyncio.run` does not close httpx.

**DeepEval**: AnswerRelevancy (answer vs question), Faithfulness (answer vs context), ContextualRelevancy (context vs question). Small local models often score the insufficient sentinel as 1.0 on relevancy.

`primary_scores` prefers the **custom** triad for the `judge_correctness` columns so the report stays stable when several frameworks run.

Macro-average per framework is `aggregate_frameworks`.

The 20-query judge **does not pick the winner**. It is for disagreement review (see the report).

## Fusion formulas (hybrid)

**Weighted** (won nDCG on this set):

1. Min-max each of BM25 and dense lists to \([0,1]\).
2. \(s(id) = w_{\mathrm{bm25}} s_{\mathrm{lex}}(id) + w_{\mathrm{dense}} s_{\mathrm{dense}}(id)\) with defaults \(0.5, 0.5\).
3. Missing from one list contributes 0 for that signal.

**RRF**:

\[
s(id) = \sum_{\text{lists}} \frac{1}{k + \mathrm{rank}_{\text{list}}(id)}
\quad (k=60)
\]

Each list votes once per ID.

## Query expansion

`expand_query` prepends the original question. Failures return `[original]` so retrieval never dies. Only `hybrid_expansion` / `best` (eval) or `use_expansion=true` (API) issue that LLM call; BM25 and dense then search **all** queries and merge.

## File pointers for a first read

1. `backend/app/api/search.py` — `_retrieve`, then `search` / `answer`
2. `backend/app/retrieval/hybrid.py` — fusion
3. `backend/app/generation/answer.py` — citation gate
4. `backend/app/evaluation/metrics.py` — rank metrics
5. `backend/scripts/run_evaluation.py` — `make_retrieve_fn` + judge merge
6. `backend/app/evaluation/judge.py` — framework dispatch
