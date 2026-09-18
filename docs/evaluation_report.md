# Evaluation report

Same **100 query IDs** for every configuration (seed 42). Gold `relevant_passage_ids` used only for scoring. Judge prompt, model, and temperature fixed across runs.

| Step 4 layer | Required | Status |
|---|---|---|
| Traditional retrieval metrics | Recall@5, Recall@10, MRR@10, nDCG@10, latency, cost | Done — 100 queries, table below |
| LLM-judge | Correctness, groundedness, context relevance; fixed settings | Done — 20-query subset, `granite4.1:3b via ollama`, frameworks: custom, ragas, deepeval |
| Manual review | Disagreements between judge and retrieval; citation validity in code | Done — cases below; validator in `backend/app/generation/answer.py` |

Embeddings `BAAI/bge-base-en-v1.5`. Vector store `qdrant`. Judge: `granite4.1:3b via ollama`, temperature `0.0`.

## Traditional retrieval metrics

Measured with `relevant_passage_ids`. Latency is retrieval wall time. Cost is LLM token cost (`$0` on Ollama).

## Comparison table

| config | recall@5 | recall@10 | mrr@10 | ndcg@10 | latency_ms | p95 | cost_usd/q |
|---|---|---|---|---|---|---|---|
| lexical | 0.317 | 0.391 | 0.661 | 0.481 | 56.9 | 93.3 | 0.0000 |
| dense | 0.397 | 0.469 | 0.792 | 0.588 | 32.6 | 207.9 | 0.0000 |
| hybrid_rrf | 0.395 | 0.478 | 0.764 | 0.583 | 81.4 | 125.4 | 0.0000 |
| hybrid_weighted | 0.400 | 0.495 | 0.794 | 0.607 | 82.3 | 124.4 | 0.0000 |
| hybrid_expansion | 0.425 | 0.518 | 0.826 | 0.632 | 26930.4 | 29992.7 | 0.0000 |
| best | 0.425 | 0.518 | 0.826 | 0.632 | 26930.4 | 29992.7 | 0.0000 |

`best` is the strongest variant: weighted hybrid + query expansion.

## LLM-judge evaluation

Frameworks from `JUDGE_FRAMEWORK`: custom, ragas, deepeval. Prompt/model/temperature fixed (`granite4.1:3b via ollama`, temp=0.0). Subset size 20. The 20-query judge does **not** select the winner — that is 100-query nDCG@10 then Recall@10.

| Framework | Package | Correctness | Groundedness | Context relevance |
|---|---|---|---|---|
| custom | built-in JSON judge | JSON `correctness` | JSON `groundedness` | JSON `context_relevance` |
| ragas | RAGAS 0.4.3 | `FactualCorrectness` (`mode=precision`) | `Faithfulness` | `ContextRelevance` |
| deepeval | DeepEval 4.2.3 | `AnswerRelevancyMetric` | `FaithfulnessMetric` | `ContextualRelevancyMetric` |

RAGAS precision scores whether **answer claims** are supported by the gold. It is 0 for `INSUFFICIENT_EVIDENCE` and when claim-NLI finds no overlap. Custom and DeepEval often score that sentinel as 1.0.

| config | cites valid | insufficient | n_judged |
|---|---|---|---|
| lexical | 1.000 | 0.300 | 20 |
| hybrid_weighted | 1.000 | 0.350 | 20 |
| hybrid_expansion | 1.000 | 0.300 | 20 |

### custom

| config | correctness | groundedness | context_rel | n |
|---|---|---|---|---|
| lexical | 0.963 | 0.951 | 0.960 | 20 |
| hybrid_weighted | 0.960 | 0.947 | 0.954 | 20 |
| hybrid_expansion | 0.963 | 0.951 | 0.960 | 20 |

### ragas

| config | correctness | groundedness | context_rel | n |
|---|---|---|---|---|
| lexical | 0.451 | 0.575 | 0.600 | 20 |
| hybrid_weighted | 0.439 | 0.570 | 0.600 | 20 |
| hybrid_expansion | 0.480 | 0.567 | 0.713 | 20 |

### deepeval

| config | correctness | groundedness | context_rel | n |
|---|---|---|---|---|
| lexical | 1.000 | 0.951 | 0.606 | 20 |
| hybrid_weighted | 0.950 | 0.955 | 0.606 | 20 |
| hybrid_expansion | 1.000 | 0.966 | 0.654 | 20 |


## Winner: `hybrid_expansion`
Selected by max nDCG@10 then Recall@10 (ndcg=0.632, recall@10=0.518, latency=26930.4 ms).
Trade-off: hybrid+expansion gains recall at +1 LLM call latency (and token cost on hosted providers). For interactive UI we default expansion **off** and use weighted fusion (~80 ms retrieval); enable expansion when maximising recall offline.
Code alias: `best` = weighted fusion + query expansion (same recipe as `hybrid_expansion`).

## Manual review

Cases where the judge and retrieval metrics disagree. Citation validity is checked in code.

## Success cases

- **Q2218** (`hybrid_weighted`, recall@10=1.0, correctness=1.0): How many times is CLAST faster than BLAST? — Top hit 25495907 is gold; answer cites ≈80.8×. Custom, RAGAS precision, and DeepEval all 1.0.
- **Q318** (`hybrid_weighted`, recall@10=0.167, correctness=0.95): Describe the mechanism of action of drisapersen — Antisense-oligo / exon-51 skipping answer; custom 0.95, RAGAS 0.80, DeepEval 1.0.
- **Q3383** (`hybrid_weighted`, recall@10=0.5, correctness=0.95): AhR ligands are attractive drug targets ... due to their induction of Cyp1a1, yes or no? — Grounded yes-answer with valid cites; custom 0.95, RAGAS 1.0 (was insufficient under the older gemma run).

## Failure / disagreement cases

- **Q2218** (`hybrid_expansion`, recall@10=1.0, correctness=1.0): How many times is CLAST faster than BLAST? — Recall@10=1.0 but the citation validator forced INSUFFICIENT_EVIDENCE; RAGAS precision 0, custom/DeepEval still 1.0.
- **Q85** (`hybrid_weighted`, recall@10=0.6, correctness=1.0): Which transcription factor is considered as a master regulator of lysosomal genes? — TFEB answer looks right and custom/DeepEval are 1.0, but RAGAS precision is 0 (claim NLI vs a longer gold).
- **Q4073** (`hybrid_weighted`, recall@10=0.091, correctness=1.0): The Shingrix vaccine is used to prevent what disease? — Low recall, answer INSUFFICIENT_EVIDENCE; RAGAS 0, custom/DeepEval 1.0 — those two often score the sentinel as fully correct.
- **Q4035** (`lexical`, recall@10=0.0, correctness=1.0): Which receptor is blocked by Finerenone? — No gold in top-10, INSUFFICIENT_EVIDENCE; expansion recovers a grounded mineralocorticoid-receptor answer (RAGAS 1.0).

Citation validity is enforced in `app/generation/answer.py`: any hallucinated `[id]` (or answer with zero valid cites) becomes `INSUFFICIENT_EVIDENCE`.
