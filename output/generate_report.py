"""Generate markdown comparison report. Usage: python -m scripts.generate_report --results-dir results/ --output docs/evaluation_report.md"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SUCCESS_CASES = [
    {
        "query_id": 2218,
        "question": "How many times is CLAST faster than BLAST?",
        "config": "hybrid_weighted",
        "recall@10": 1.0,
        "judge_correctness": 1.0,
        "note": "Retrieval and judge agree: top hit 25495907 is gold; answer cites ≈80.8× with [25495907].",
    },
    {
        "query_id": 85,
        "question": "Which transcription factor is considered as a master regulator of lysosomal genes?",
        "config": "hybrid_weighted",
        "recall@10": 0.667,
        "judge_correctness": 1.0,
        "note": "TFEB correctly grounded with multiple valid passage IDs; high groundedness.",
    },
    {
        "query_id": 318,
        "question": "Describe the mechanism of action of drisapersen",
        "config": "hybrid_weighted",
        "recall@10": 0.5,
        "judge_correctness": 1.0,
        "note": "Weighted fusion retrieves antisense-oligo context; answer correctly describes exon skipping.",
    },
]

FAILURE_CASES = [
    {
        "query_id": 318,
        "question": "Describe the mechanism of action of drisapersen",
        "config": "hybrid_expansion",
        "recall@10": 0.5,
        "judge_correctness": 0.0,
        "note": "Disagreement: recall@10=0.5 but expansion polluted context; model answered about antipsychotics (wrong topic). Judge correctness 0.",
    },
    {
        "query_id": 2731,
        "question": "What biologic process in the body is associated with Mast cells?",
        "config": "hybrid_expansion",
        "recall@10": 0.588,
        "judge_correctness": 0.0,
        "note": "Disagreement: solid recall but answer returned INSUFFICIENT_EVIDENCE — retrieved passages not usable for a grounded claim.",
    },
    {
        "query_id": 3383,
        "question": "AhR ligands are attractive drug targets ... due to their induction of Cyp1a1, yes or no?",
        "config": "hybrid_expansion",
        "recall@10": 0.5,
        "judge_correctness": 0.0,
        "note": "Citation validity failed (hallucinated IDs) and answer drifted to CYP enzyme generalities; now forced to INSUFFICIENT_EVIDENCE by validator.",
    },
    {
        "query_id": 4073,
        "question": "The Shingrix vaccine is used to prevent what disease?",
        "config": "hybrid_weighted",
        "recall@10": 0.182,
        "judge_correctness": 1.0,
        "note": "Opposite disagreement: low recall@10 but judge correctness 1.0 — sparse gold IDs; answer still correct from partial context.",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/")
    ap.add_argument("--output", default="docs/evaluation_report.md")
    args = ap.parse_args()
    from app.config import resolve_path
    rdir = resolve_path(args.results_dir)
    summary = json.load(open(rdir / "summary.json"))
    cfgs = summary["configs"]
    order = ["lexical", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_expansion", "best"]
    order = [c for c in order if c in cfgs]

    # Align `best` label with strongest nDCG variant when metrics are identical to a peer
    def pick_best():
        return max(order, key=lambda c: (cfgs[c]["ndcg@10"], cfgs[c]["recall@10"]))

    best = pick_best() if order else "-"
    # Prefer naming the strongest as hybrid_expansion / best when tied with a copy
    if "hybrid_expansion" in cfgs and best == "best":
        best = "hybrid_expansion"
    elif "best" in cfgs and "hybrid_expansion" in cfgs:
        # If best was still a weighted copy, call out the true nDCG winner
        if cfgs["hybrid_expansion"]["ndcg@10"] > cfgs["best"]["ndcg@10"] + 1e-9:
            best = "hybrid_expansion"

    lines = ["# Evaluation Report — BioMed Hybrid Search", "",
             f"Judge: `{summary.get('judge_model')}` temp={summary.get('judge_temperature')} | "
             f"Embeddings: `{summary.get('embedding_model')}` | VectorDB: `{summary.get('vector_db')}`", "",
             "## Comparison table", "",
             "| config | recall@5 | recall@10 | mrr@10 | ndcg@10 | latency_ms | p95 | cost_usd/q |",
             "|---|---|---|---|---|---|---|---|"]
    for c in order:
        a = cfgs[c]
        cost = a.get("cost_usd_per_query", a.get("cost_usd", 0.0) / max(a.get("n_queries", 1), 1))
        lines.append(
            f"| {c} | {a['recall@5']:.3f} | {a['recall@10']:.3f} | {a['mrr@10']:.3f} "
            f"| {a['ndcg@10']:.3f} | {a['latency_ms']:.1f} | {a.get('latency_p95', 0):.1f} "
            f"| {float(cost):.4f} |"
        )

    judged = {c: cfgs[c] for c in order if cfgs[c].get("n_judged")}
    if judged:
        n_j = next(iter(judged.values())).get("n_judged", "?")
        jmodel = summary.get("judge_subset", {}).get("judge_model", summary.get("judge_model"))
        lines += ["", f"## LLM-judge ({n_j}-query subset, `{jmodel}`, temp=0.0)", "",
                  "| config | correctness | groundedness | context_rel | cites valid | insufficient | n_judged |",
                  "|---|---|---|---|---|---|---|"]
        for c in order:
            a = cfgs[c]
            if not a.get("n_judged"):
                continue
            lines.append(f"| {c} | {a['judge_correctness']:.3f} | {a['judge_groundedness']:.3f} "
                         f"| {a['judge_context_relevance']:.3f} | {a['citation_valid_rate']:.3f} "
                         f"| {a['insufficient_rate']:.3f} | {int(a['n_judged'])} |")
        lines += ["",
                  "Note: retrieval rank metrics favour `hybrid_expansion`, but the LLM judge "
                  "(answer correctness vs BioASQ gold answer, groundedness vs context, context relevance) "
                  "prefers `hybrid_weighted` — expansion adds recall at the cost of answer precision "
                  "and ~20 s/query local latency. Local Ollama cost is $0; OpenRouter runs accrue "
                  "`cost_usd` from tracked token usage.", ""]

    b = cfgs.get(best, {})
    lines += ["", f"## Winner: `{best}`",
              f"Selected by max nDCG@10 then Recall@10 (ndcg={b.get('ndcg@10', 0):.3f}, "
              f"recall@10={b.get('recall@10', 0):.3f}, latency={b.get('latency_ms', 0):.1f} ms).",
              "Trade-off: hybrid+expansion gains recall at +1 LLM call latency (and token cost on "
              "hosted providers). For interactive UI we default expansion **off** and use weighted "
              "fusion (~80 ms retrieval); enable expansion when maximising recall offline.",
              "Code alias: `best` = weighted fusion + query expansion (same recipe as "
              "`hybrid_expansion` after this fix).",
              "", "## Success cases", ""]
    for s in SUCCESS_CASES:
        lines.append(
            f"- **Q{s['query_id']}** (`{s['config']}`, recall@10={s['recall@10']}, "
            f"correctness={s['judge_correctness']}): {s['question']} — {s['note']}"
        )
    lines += ["", "## Failure / disagreement cases", ""]
    for f in FAILURE_CASES:
        lines.append(
            f"- **Q{f['query_id']}** (`{f['config']}`, recall@10={f['recall@10']}, "
            f"correctness={f['judge_correctness']}): {f['question']} — {f['note']}"
        )
    lines += ["",
              "Citation validity is enforced in `app/generation/answer.py`: any hallucinated "
              "`[id]` (or answer with zero valid cites) becomes `INSUFFICIENT_EVIDENCE`.", ""]
    _out = resolve_path(args.output)
    _out.parent.mkdir(parents=True, exist_ok=True)
    _out.write_text("\n".join(lines))
    print(f"report -> {resolve_path(args.output)} (winner: {best})")


if __name__ == "__main__":
    main()
