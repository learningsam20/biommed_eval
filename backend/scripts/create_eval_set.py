"""Create frozen 100-query eval set. Usage: python -m scripts.create_eval_set --output data/eval_queries_100.json --seed 42 --size 100"""
import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PATTERNS = [
    ("yesno", re.compile(r"^(is|are|does|do|can|has|have|should|will|would)\b", re.I)),
    ("definition", re.compile(r"\b(what is|what are|define|definition)\b", re.I)),
    ("list", re.compile(r"\b(list|which|name|enumerate)\b", re.I)),
    ("treatment", re.compile(r"\b(treat|therapy|drug|dose|management|prevent)\b", re.I)),
    ("mechanism", re.compile(r"\b(mechanism|pathway|how does|mediate|regulat)\b", re.I)),
    ("effect", re.compile(r"\b(effect|affect|impact|outcome|risk|associated)\b", re.I)),
]


def classify(q: str) -> str:
    for label, rx in PATTERNS:
        if rx.search(q):
            return label
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/eval_queries_100.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=100)
    args = ap.parse_args()

    from datasets import load_dataset
    from app.config import get_settings
    s = get_settings()
    ds = load_dataset(s.HF_DATASET_ID, "question-answer-passages")["test"]
    buckets: dict[str, list] = {}
    for row in ds:
        buckets.setdefault(classify(row["question"]), []).append(row)

    rng = random.Random(args.seed)
    per = max(args.size // max(len(buckets), 1), 1)
    picked = []
    for label in sorted(buckets):
        pool = buckets[label][:]
        rng.shuffle(pool)
        picked.extend([(label, r) for r in pool[:per]])
    # top-up to exact size from global pool (deterministic)
    if len(picked) < args.size:
        seen = {r["id"] for _, r in picked}
        rest = [r for rows in buckets.values() for r in rows if r["id"] not in seen]
        rng.shuffle(rest)
        need = args.size - len(picked)
        picked.extend([(classify(r["question"]), r) for r in rest[:need]])
    rng.shuffle(picked)
    picked = picked[:args.size]

    def _parse_ids(v):
        if isinstance(v, list):
            return [int(x) for x in v]
        try:
            return [int(x) for x in ast.literal_eval(v)]
        except Exception:
            return []

    out = [{"query_id": int(r["id"]), "question": r["question"], "answer": r["answer"],
            "relevant_passage_ids": _parse_ids(r["relevant_passage_ids"]),
            "qtype": label} for label, r in picked]
    from app.config import resolve_path
    dest = resolve_path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    from collections import Counter
    print(f"wrote {len(out)} queries -> {dest} {dict(Counter(x['qtype'] for x in out))}")


if __name__ == "__main__":
    main()
