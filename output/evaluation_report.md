# Evaluation Report — BioMed Hybrid Search

Judge: `same_as_llm` temp=0.0 | Embeddings: `BAAI/bge-base-en-v1.5` | VectorDB: `qdrant`

## Comparison table

| config | recall@5 | recall@10 | mrr@10 | ndcg@10 | latency_ms | p95 | cost_usd/q |
|---|---|---|---|---|---|---|---|
| lexical | 0.317 | 0.391 | 0.661 | 0.481 | 56.9 | 93.3 | 0.0000 |
| dense | 0.397 | 0.469 | 0.792 | 0.588 | 32.6 | 207.9 | 0.0000 |
| hybrid_rrf | 0.395 | 0.478 | 0.764 | 0.583 | 81.4 | 125.4 | 0.0000 |
| hybrid_weighted | 0.400 | 0.495 | 0.794 | 0.607 | 82.3 | 124.4 | 0.0000 |
| hybrid_expansion | 0.425 | 0.518 | 0.826 | 0.632 | 26930.4 | 29992.7 | 0.0000 |
| best | 0.425 | 0.518 | 0.826 | 0.632 | 26930.4 | 29992.7 | 0.0000 |

## LLM-judge (20-query subset, `gemma4:latest via ollama`, temp=0.0)

| config | correctness | groundedness | context_rel | cites valid | insufficient | n_judged |
|---|---|---|---|---|---|---|
| lexical | 0.800 | 0.660 | 0.735 | 1.000 | 0.300 | 20 |
| hybrid_weighted | 0.750 | 0.797 | 0.795 | 0.900 | 0.250 | 20 |
| hybrid_expansion | 0.575 | 0.510 | 0.730 | 0.950 | 0.250 | 20 |
| best | 0.575 | 0.510 | 0.730 | 0.950 | 0.250 | 20 |

Note: retrieval rank metrics favour `hybrid_expansion`, but the LLM judge (answer correctness vs BioASQ gold answer, groundedness vs context, context relevance) prefers `hybrid_weighted` — expansion adds recall at the cost of answer precision and ~20 s/query local latency. Local Ollama cost is $0; OpenRouter runs accrue `cost_usd` from tracked token usage.


## Winner: `hybrid_expansion`
Selected by max nDCG@10 then Recall@10 (ndcg=0.632, recall@10=0.518, latency=26930.4 ms).
Trade-off: hybrid+expansion gains recall at +1 LLM call latency (and token cost on hosted providers). For interactive UI we default expansion **off** and use weighted fusion (~80 ms retrieval); enable expansion when maximising recall offline.
Code alias: `best` = weighted fusion + query expansion (same recipe as `hybrid_expansion` after this fix).

## Success cases

- **Q2218** (`hybrid_weighted`, recall@10=1.0, correctness=1.0): How many times is CLAST faster than BLAST? — Retrieval and judge agree: top hit 25495907 is gold; answer cites ≈80.8× with [25495907].
- **Q85** (`hybrid_weighted`, recall@10=0.667, correctness=1.0): Which transcription factor is considered as a master regulator of lysosomal genes? — TFEB correctly grounded with multiple valid passage IDs; high groundedness.
- **Q318** (`hybrid_weighted`, recall@10=0.5, correctness=1.0): Describe the mechanism of action of drisapersen — Weighted fusion retrieves antisense-oligo context; answer correctly describes exon skipping.

## Failure / disagreement cases

- **Q318** (`hybrid_expansion`, recall@10=0.5, correctness=0.0): Describe the mechanism of action of drisapersen — Disagreement: recall@10=0.5 but expansion polluted context; model answered about antipsychotics (wrong topic). Judge correctness 0.
- **Q2731** (`hybrid_expansion`, recall@10=0.588, correctness=0.0): What biologic process in the body is associated with Mast cells? — Disagreement: solid recall but answer returned INSUFFICIENT_EVIDENCE — retrieved passages not usable for a grounded claim.
- **Q3383** (`hybrid_expansion`, recall@10=0.5, correctness=0.0): AhR ligands are attractive drug targets ... due to their induction of Cyp1a1, yes or no? — Citation validity failed (hallucinated IDs) and answer drifted to CYP enzyme generalities; now forced to INSUFFICIENT_EVIDENCE by validator.
- **Q4073** (`hybrid_weighted`, recall@10=0.182, correctness=1.0): The Shingrix vaccine is used to prevent what disease? — Opposite disagreement: low recall@10 but judge correctness 1.0 — sparse gold IDs; answer still correct from partial context.

Citation validity is enforced in `app/generation/answer.py`: any hallucinated `[id]` (or answer with zero valid cites) becomes `INSUFFICIENT_EVIDENCE`.
