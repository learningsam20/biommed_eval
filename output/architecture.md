# Architecture

BioMed Hybrid Search indexes BioASQ passages once, then serves the same retrieval stack to the UI and to the 100-query benchmark.

## Diagram

```mermaid
flowchart TB
    subgraph OFFLINE["Offline data path"]
        HF["HF rag-mini-bioasq<br/>40k passages / 4.7k QA"] --> NORM["Normalise + cache<br/>data/passages.jsonl"]
        NORM --> BM25B["BM25 build<br/>rank-bm25 pickle"]
        NORM --> ENC["Batch encode<br/>sentence-transformers"]
        ENC --> PIN["Pinecone serverless<br/>auto-create + upsert"]
        PIN -.fallback.-> QDR["Qdrant local"]
    end
    subgraph ONLINE["Online query path"]
        UQ["User query"] --> EXP{"Expansion on?"}
        EXP -- yes --> EXPL["LLM expansions<br/>original + N alts"]
        EXP -- no --> QSET["queries = [original]"]
        EXPL --> QSET
        QSET --> LX["BM25 parallel"]
        QSET --> DN["Dense batch-encode"]
        BM25B --> LX
        PIN --> DN
        QDR --> DN
        LX --> DED["Dedupe max-score"]
        DN --> DED
        DED --> FUS["Fusion: weighted or RRF"]
        FUS --> CTX["Top-K context"]
        CTX --> GEN["Grounded answer LLM<br/>[id] citations"]
        GEN --> VAL{"Cited IDs ⊆ retrieved<br/>and ≥1 valid cite?"}
        VAL -- no --> INS["INSUFFICIENT_EVIDENCE"]
        VAL -- yes --> ANS["Cited answer"]
    end
    subgraph UI["User-facing path"]
        SI["SearchInput + ConfigPanel"] --> SE["POST /api/search"]
        SE --> EV["EvidencePanel<br/>id / score / source"]
        EV --> AA["POST /api/generate"]
        AA --> AP["AnswerPanel<br/>inline [id] citations<br/>or insufficient evidence"]
    end
    subgraph EXP2["Eval path"]
        CS["create_eval_set seed 42<br/>frozen 100 query IDs"] --> LOOP["For each of 6 configs<br/>same retrieve as online"]
        FUS -.-> LOOP
        LOOP --> LOG["Log per query:<br/>config, retrieved IDs + scores,<br/>latency, frozen judge settings"]
        LOG --> ART["results/config_results.json<br/>+ results/summary.json"]
        ART --> MET["Recall/MRR/nDCG/latency/cost<br/>+ LLM-judge"]
        MET --> REP["generate_report<br/>evaluation_report.md"]
    end
    ONLINE --> UI
```

Offline indexes feed both interactive search and the eval runner. `run_evaluation.py` uses the same expansion → BM25/dense → fusion path as `POST /api/search`. Judge prompt, model, and temperature stay fixed in `.env`. Every search records passage IDs and scores; aggregates become the comparison table and the selected hybrid recipe.

## Design notes

Nothing in the retrieve/generate path is hard-wired to one vendor. `.env` selects the store, embedder, and LLM; changing the embedder requires `make index`.

- **Vector store.** `VECTOR_DB=pinecone` (default) for a managed serverless index with auto-create and a dimension check. `VECTOR_DB=qdrant` for local Docker. Pinecone avoids operating 40k vectors on disk; Qdrant keeps the stack runnable without a cloud key.
- **Embeddings.** `EMBEDDING_MODEL` (default `BAAI/bge-base-en-v1.5`, 768-d) is a solid English dense baseline. Set `EMBEDDING_DIM` to match; mismatch fails fast. MiniLM / bge-large work after a re-index.
- **Fusion.** Weighted min-max is the UI default because it won nDCG vs RRF on this set. RRF remains available (`FUSION_METHOD=rrf` or the UI).
- **LLM.** `LLM_PROVIDER=openrouter` or `ollama`, same client for expansion, answers, and the judge. Cloud for quality/variety of models; Ollama for local, $0 inference.
- **Citations.** Independent of the LLM: regex `\[(\d+)\]` in `backend/app/generation/answer.py`. Bad or missing cites become `INSUFFICIENT_EVIDENCE`.
- **Latency.** Expansion off by default; evidence from `/search` appears before `/generate` returns.
