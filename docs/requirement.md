# BioMed Hybrid Search — Requirements

A biomedical search and question-answering system over [`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq). The system retrieves useful passages, shows evidence to the user, and uses an LLM to write a grounded final answer with citations. Offline evaluation selects the best retrieval configuration.

## Dataset

Source: Hugging Face `rag-datasets/rag-mini-bioasq`.

Use the passage corpus together with questions, reference answers, and gold relevant passage IDs. Gold IDs are for offline scoring only, never at query time.

## Retrieval

Three retrieval modes over the same indexed passages:

1. **Lexical** search (BM25 or equivalent).
2. **Dense** vector search.
3. **Hybrid** search that combines both signals.

Query expansion runs before retrieval. Each query produces a small set of useful alternate queries; the original query is always retained.

Every search records retrieved passage IDs and scores.

## User interface

A biomedical-question search UI, hosted locally or on a static host.

The UI provides:

- Question input and a search action.
- The original query and its expansions.
- Ranked evidence passages with IDs, scores, and source links where available.
- A final LLM answer with inline citations to retrieved passage IDs.
- An explicit **insufficient evidence** state.

The final LLM call answers only from the selected retrieved passages.

## Offline evaluation

A frozen set of **100 queries** from the dataset, mixing yes/no, definition, list, treatment, mechanism, and effect questions where possible. The same 100 query IDs are used for every run.

Compare at least:

| Configuration | Required |
|---|---|
| Lexical | Yes |
| Dense | Yes |
| Hybrid | Yes |
| Hybrid + query expansion | Yes |
| Strongest variant | Yes |

## Scoring

### Traditional retrieval metrics

Using `relevant_passage_ids`, measure:

- Recall@5, Recall@10
- MRR@10
- nDCG@10
- Latency
- Cost

### LLM-judge

Use RAGAS, DeepEval, or an equivalent framework to score:

- Answer correctness
- Groundedness
- Context relevance

Judge prompt, model, and settings stay fixed across runs.

### Manual review

Inspect examples where the judge and retrieval metrics disagree. Citation validity is checked in code.

## System design

The architecture diagram covers four paths:

1. **Offline data path** — dataset loading, passage preparation, and the indexes that are built.
2. **Online query path** — query expansion, lexical and dense retrieval, hybrid fusion, selected context, and the final LLM call.
3. **User-facing path** — UI input, evidence display, final answer, and citations.
4. **Experiment path** — how configurations are logged and how the same pipeline is reused for the 100-query evaluation.

Document the main choices: vector database, embedding model, fusion method, final-answer model, and where citation validation is enforced.

## Artifacts

| Artifact | Purpose |
|---|---|
| Working UI and source | Interactive search with evidence and grounded answers |
| Frozen 100-query set + evaluation script | Reproducible comparison of retrieval configs |
| Short report with one comparison table | What won, why, latency/cost trade-off, success and failure cases |
| Selected best hybrid configuration | Recipe backed by the experiment results |
| System-design diagram | The four paths above |

## Where this lives in the repo

| Requirement | Location |
|---|---|
| Dataset + frozen 100 queries | `data/eval_queries_100.json` |
| Retrieval + expansion | `backend/app/retrieval/` |
| UI | `frontend/` |
| Evaluation runner | `backend/scripts/run_evaluation.py` |
| Scoring (metrics, judge, citations) | `backend/app/evaluation/`, `backend/app/generation/answer.py` |
| Report | `docs/evaluation_report.md` |
| Best hybrid config | `docs/best_hybrid_config.json` |
| System diagram | `docs/architecture.md` |
