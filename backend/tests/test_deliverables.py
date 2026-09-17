import json
from pathlib import Path

from app.config import resolve_path

REQUIRED_CONFIGS = ["lexical", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_expansion", "best"]


def test_output_deliverables_present():
    out = resolve_path("output")
    for name in [
        "README.md",
        "eval_queries_100.json",
        "run_evaluation.py",
        "create_eval_set.py",
        "generate_report.py",
        "evaluation_report.md",
        "architecture.md",
        "best_hybrid_config.json",
        "results/summary.json",
    ]:
        assert (out / name).exists(), f"missing output/{name}"


def test_report_has_comparison_table_and_winner():
    text = resolve_path("output/evaluation_report.md").read_text()
    assert "Comparison table" in text
    assert "recall@5" in text and "ndcg@10" in text
    assert "Winner" in text
    assert "Success cases" in text
    assert "Failure" in text
    assert "latency" in text.lower() or "Trade-off" in text


def test_architecture_shows_four_paths():
    text = resolve_path("output/architecture.md").read_text()
    low = text.lower()
    assert "offline" in low
    assert "online" in low or "query path" in low
    assert "user-facing" in low or "ui" in low
    assert "experiment" in low
    assert "retrieved ids" in low or "retrieved IDs" in text
    assert "100" in text
    assert "same retrieve" in low or "same expansion" in low


def test_results_record_ids_and_scores():
    rdir = resolve_path("output/results")
    for cfg in REQUIRED_CONFIGS:
        rows = json.loads((rdir / f"{cfg}_results.json").read_text())
        assert len(rows) == 100, cfg
        sample = rows[0]
        assert sample["retrieved_ids"], cfg
        assert sample["retrieved_scores"], cfg
        assert sample["retrieved_scores"][0]["passage_id"] == sample["retrieved_ids"][0]
        assert "score" in sample["retrieved_scores"][0]


def test_summary_covers_required_configs():
    summary = json.loads(resolve_path("output/results/summary.json").read_text())
    cfgs = summary["configs"]
    for cfg in REQUIRED_CONFIGS:
        assert cfg in cfgs, cfg
        m = cfgs[cfg]
        for key in ("recall@5", "recall@10", "mrr@10", "ndcg@10", "latency_ms"):
            assert key in m, f"{cfg} missing {key}"


def test_best_config_backed_by_results():
    best = json.loads(resolve_path("output/best_hybrid_config.json").read_text())
    assert best["name"]
    assert best["fusion"]
    assert best["query_expansion"] is True
    assert best["metrics"]["ndcg@10"]
    summary = json.loads(resolve_path("output/results/summary.json").read_text())
    assert best["name"] in summary["configs"]
