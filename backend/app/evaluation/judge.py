"""LLM-as-judge: custom, RAGAS, and DeepEval.

Select with JUDGE_FRAMEWORK in .env: custom | ragas | deepeval | all
(or a comma-separated list). Prompt/model/temperature stay fixed across configs.
"""
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

JUDGE_SYSTEM = (
    "You are a strict biomedical RAG evaluator. Score 0.0-1.0. "
    "Return ONLY JSON: {\"correctness\": x, \"groundedness\": y, \"context_relevance\": z}"
)

JUDGE_VERSION = "judge-v2-frameworks"
VALID_FRAMEWORKS = ("custom", "ragas", "deepeval")
JUDGE_CONTEXT_PASSAGES = 5
JUDGE_CONTEXT_CHARS = 4000
JUDGE_MAX_TOKENS = 1500
JUDGE_QUERY_TIMEOUT_S = 120
LLM_CALL_TIMEOUT_S = 20


def parse_judge_frameworks(value: str | None) -> List[str]:
    raw = (value or "custom").strip().lower()
    if raw in ("all", "*"):
        return list(VALID_FRAMEWORKS)
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    unknown = [p for p in parts if p not in VALID_FRAMEWORKS]
    if unknown:
        raise ValueError(
            f"Unknown JUDGE_FRAMEWORK {unknown}. Use custom, ragas, deepeval, or all."
        )
    out: List[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out or ["custom"]


def judge_prompt(question: str, reference: str, answer: str,
                 evidence: List[Tuple[int, str]]) -> str:
    ctx = "\n".join(f"[{pid}] {t[:JUDGE_CONTEXT_CHARS]}" for pid, t in evidence[:10])
    return (f"Question: {question}\nReference: {reference[:1500]}\n"
            f"Candidate: {answer[:2000]}\nContext:\n{ctx}\nScore correctness, "
            f"groundedness (claims supported by context), context relevance.")


@dataclass
class JudgeScores:
    correctness: float
    groundedness: float
    context_relevance: float
    framework: str = "custom"

    def as_row(self) -> Dict[str, float]:
        return {
            "correctness": float(self.correctness),
            "groundedness": float(self.groundedness),
            "context_relevance": float(self.context_relevance),
        }


def run_custom_judge(question: str, reference: str, answer: str,
                     evidence: List[Tuple[int, str]], llm_client,
                     temperature: float = 0.0) -> JudgeScores:
    import json
    try:
        raw = llm_client.chat(system=JUDGE_SYSTEM,
                              user=judge_prompt(question, reference, answer, evidence),
                              temperature=temperature)
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return JudgeScores(float(data.get("correctness", 0.0)),
                           float(data.get("groundedness", 0.0)),
                           float(data.get("context_relevance", 0.0)),
                           framework="custom")
    except Exception:
        return JudgeScores(0.0, 0.0, 0.0, framework="custom")


# Back-compat alias used by older tests/scripts
def run_judge(question: str, reference: str, answer: str,
              evidence: List[Tuple[int, str]], llm_client,
              temperature: float = 0.0) -> JudgeScores:
    return run_custom_judge(question, reference, answer, evidence, llm_client, temperature)


def _contexts(evidence: Sequence[Tuple[int, str]]) -> List[str]:
    """Top passages, each clipped to JUDGE_CONTEXT_CHARS (4k, not the 4k token cap)."""
    out: List[str] = []
    for _, text in evidence:
        if not text:
            continue
        out.append(text[:JUDGE_CONTEXT_CHARS])
        if len(out) >= JUDGE_CONTEXT_PASSAGES:
            break
    return out or [""]


def score_with_frameworks(
    frameworks: Iterable[str],
    question: str,
    reference: str,
    answer: str,
    evidence: List[Tuple[int, str]],
    llm_client,
    temperature: float = 0.0,
    deadline: float | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run each selected framework on one (question, answer, context) triple."""
    import time
    zeros = {"correctness": 0.0, "groundedness": 0.0, "context_relevance": 0.0}
    out: Dict[str, Dict[str, float]] = {}
    fw = list(frameworks)

    def leftover() -> float:
        return 999.0 if deadline is None else deadline - time.monotonic()

    def skip(name: str) -> bool:
        if leftover() < 8:
            print(f"    skip {name} ({leftover():.0f}s left)", flush=True)
            out[name] = dict(zeros)
            return True
        return False

    if "custom" in fw:
        if not skip("custom"):
            try:
                out["custom"] = run_custom_judge(
                    question, reference, answer, evidence, llm_client, temperature
                ).as_row()
            except Exception as e:
                print(f"  custom judge failed: {type(e).__name__}: {e}")
                out["custom"] = dict(zeros)
    if "ragas" in fw:
        if not skip("ragas"):
            print("    ragas ...", flush=True)
            try:
                from app.evaluation.ragas_judge import score_ragas_one
                out["ragas"] = score_ragas_one(
                    question, reference, answer, _contexts(evidence), llm_client, temperature
                )
            except Exception as e:
                print(f"  ragas judge failed: {type(e).__name__}: {e}")
                out["ragas"] = dict(zeros)
    if "deepeval" in fw:
        if not skip("deepeval"):
            print("    deepeval ...", flush=True)
            try:
                from app.evaluation.deepeval_judge import score_deepeval_one
                out["deepeval"] = score_deepeval_one(
                    question, reference, answer, _contexts(evidence), llm_client, temperature
                )
            except Exception as e:
                print(f"  deepeval judge failed: {type(e).__name__}: {e}")
                out["deepeval"] = dict(zeros)
    return out


def primary_scores(frameworks_out: Dict[str, Dict[str, float]]) -> JudgeScores:
    """Prefer custom, else first available — keeps report columns stable."""
    row = frameworks_out.get("custom") or next(iter(frameworks_out.values()), {})
    return JudgeScores(
        correctness=float(row.get("correctness", 0.0)),
        groundedness=float(row.get("groundedness", 0.0)),
        context_relevance=float(row.get("context_relevance", 0.0)),
        framework=next(iter(frameworks_out), "custom"),
    )


def aggregate_frameworks(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Average per-framework scores across judged rows."""
    import math
    buckets: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        block = row.get("judges") or {}
        for name, scores in block.items():
            buckets.setdefault(name, []).append(scores)

    def _mean(items, key):
        vals = []
        for i in items:
            try:
                v = float(i.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isnan(v):
                vals.append(v)
        return sum(vals) / len(vals) if vals else 0.0

    agg: Dict[str, Dict[str, float]] = {}
    for name, items in buckets.items():
        agg[name] = {
            "correctness": _mean(items, "correctness"),
            "groundedness": _mean(items, "groundedness"),
            "context_relevance": _mean(items, "context_relevance"),
            "n": len(items),
        }
    return agg
