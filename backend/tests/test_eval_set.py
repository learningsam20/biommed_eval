import json
from collections import Counter
from pathlib import Path

from app.config import cors_origins, resolve_path

REQUIRED_TYPES = {"yesno", "definition", "list", "treatment", "mechanism", "effect"}


def test_frozen_eval_set_is_100_stratified():
    path = resolve_path("data/eval_queries_100.json")
    assert path.exists(), "frozen 100-query set missing"
    rows = json.loads(path.read_text())
    assert len(rows) == 100
    ids = [r["query_id"] for r in rows]
    assert len(set(ids)) == 100
    for r in rows:
        assert r["question"] and r["answer"]
        assert isinstance(r["relevant_passage_ids"], list)
        assert r["qtype"]
    types = set(Counter(r["qtype"] for r in rows))
    assert REQUIRED_TYPES <= types


def test_cors_allows_localhost_and_loopback():
    origins = cors_origins("http://localhost:5269")
    assert "http://localhost:5269" in origins
    assert "http://127.0.0.1:5269" in origins


def test_run_evaluation_parses():
    import ast
    src = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py"
    ast.parse(src.read_text())
