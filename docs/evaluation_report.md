# Evaluation report

Protocol: **100 frozen queries** (seed 42) for every retrieval config. Embeddings `BAAI/bge-base-en-v1.5`. Vector store **Qdrant**. Judge prompt, model, and temperature (`0.0`) were identical across runs. Gold `relevant_passage_ids` are used only for scoring.

## Comparison table

| config | recall@5 | recall@10 | mrr@10 | ndcg@10 | latency_ms | p95 | cost_usd/q |
|---|---|---|---|---|---|---|---|
| lexical | 0.317 | 0.391 | 0.661 | 0.481 | 56.9 | 93.3 | 0.0000 |
| dense | 0.397 | 0.469 | 0.792 | 0.588 | 32.6 | 207.9 | 0.0000 |
| hybrid_rrf | 0.395 | 0.478 | 0.764 | 0.583 | 81.4 | 125.4 | 0.0000 |
| hybrid_weighted | 0.400 | 0.495 | 0.794 | 0.607 | 82.3 | 124.4 | 0.0000 |
| hybrid_expansion | 0.425 | 0.518 | 0.826 | 0.632 | 26930.4 | 29992.7 | 0.0000 |
| best | 0.425 | 0.518 | 0.826 | 0.632 | 26930.4 | 29992.7 | 0.0000 |

`best` is an alias of `hybrid_expansion` (weighted fusion + query expansion).

## LLM-judge (20-query subset, `gemma4:latest` via Ollama, temp=0.0)

| config | correctness | groundedness | context_rel | cites valid | insufficient | n_judged |
|---|---|---|---|---|---|---|
| lexical | 0.800 | 0.660 | 0.735 | 1.000 | 0.300 | 20 |
| hybrid_weighted | 0.750 | 0.797 | 0.795 | 0.900 | 0.250 | 20 |
| hybrid_expansion | 0.575 | 0.510 | 0.730 | 0.950 | 0.250 | 20 |
| best | 0.575 | 0.510 | 0.730 | 0.950 | 0.250 | 20 |

Rank metrics favour expansion; the judge prefers weighted hybrid for answer correctness and groundedness. Expansion buys recall with one extra LLM call (~27 s/query locally). Ollama cost is $0; OpenRouter runs record `cost_usd` from token usage.

## Winner

**`hybrid_expansion`** — highest nDCG@10 (0.632) and Recall@10 (0.518).

**Trade-off.** That gain is ~27 s/query versus ~80 ms for weighted hybrid without expansion. The product UI therefore defaults expansion **off** and weighted fusion; turn expansion on when maximising recall.

## Success cases

- **Q2218** (`hybrid_weighted`, recall@10=1.0, correctness=1.0): How many times is CLAST faster than BLAST? — Top hit 25495907 is gold; answer cites ≈80.8× with [25495907].
- **Q85** (`hybrid_weighted`, recall@10=0.667, correctness=1.0): Which transcription factor is considered as a master regulator of lysosomal genes? — TFEB grounded with valid passage IDs.
- **Q318** (`hybrid_weighted`, recall@10=0.5, correctness=1.0): Describe the mechanism of action of drisapersen — Antisense-oligo context; answer describes exon skipping.

## Failure / disagreement cases

- **Q318** (`hybrid_expansion`, recall@10=0.5, correctness=0.0): Same question — expansion polluted context; the model answered about antipsychotics.
- **Q2731** (`hybrid_expansion`, recall@10=0.588, correctness=0.0): What biologic process in the body is associated with Mast cells? — Solid recall, but the answer returned `INSUFFICIENT_EVIDENCE`.
- **Q3383** (`hybrid_expansion`, recall@10=0.5, correctness=0.0): AhR ligands / Cyp1a1 yes-or-no — hallucinated citation IDs; validator forced insufficient evidence.
- **Q4073** (`hybrid_weighted`, recall@10=0.182, correctness=1.0): The Shingrix vaccine is used to prevent what disease? — Low recall but a correct answer from partial context.

Citation checks live in `backend/app/generation/answer.py`.
