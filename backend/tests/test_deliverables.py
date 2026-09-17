import json

from app.config import resolve_path


def test_eval_artifacts_present():
    for path in [
        "docs/evaluation_report.md",
        "docs/best_hybrid_config.json",
        "docs/architecture.md",
        "backend/scripts/run_evaluation.py",
        "data/eval_queries_100.json",
    ]:
        assert resolve_path(path).exists(), f"missing {path}"


def test_report_has_comparison_table_and_winner():
    text = resolve_path("docs/evaluation_report.md").read_text()
    assert "Comparison table" in text
    assert "recall@5" in text and "ndcg@10" in text
    assert "Winner" in text
    assert "Success cases" in text
    assert "Failure" in text
    assert "latency" in text.lower() or "Trade-off" in text


def test_architecture_shows_four_paths():
    text = resolve_path("docs/architecture.md").read_text()
    low = text.lower()
    assert "offline" in low
    assert "online" in low or "query path" in low
    assert "user-facing" in low
    assert "experiment" in low
    assert "retrieved ids" in low or "retrieved IDs" in text
    assert "100" in text
    assert "same" in low and ("pipeline" in low or "retrieve" in low or "online" in low)


def test_best_config_matches_report_winner():
    best = json.loads(resolve_path("docs/best_hybrid_config.json").read_text())
    report = resolve_path("docs/evaluation_report.md").read_text()
    assert best["name"]
    assert best["fusion"]
    assert best["query_expansion"] is True
    assert best["metrics"]["ndcg@10"]
    assert best["name"] in report
