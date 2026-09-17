"""LLM-judge interface (RAGAS/DeepEval-compatible prompt contract).

Judge settings (model/prompt/temperature) must be identical across configs;
they are recorded in every eval output file.
"""
from dataclasses import dataclass
from typing import List, Tuple


JUDGE_SYSTEM = (
    "You are a strict biomedical RAG evaluator. Score 0.0-1.0. "
    "Return ONLY JSON: {\"correctness\": x, \"groundedness\": y, \"context_relevance\": z}"
)

JUDGE_VERSION = "judge-v1"


def judge_prompt(question: str, reference: str, answer: str,
                 evidence: List[Tuple[int, str]]) -> str:
    ctx = "\n".join(f"[{pid}] {t[:800]}" for pid, t in evidence[:10])
    return (f"Question: {question}\nReference: {reference[:1500]}\n"
            f"Candidate: {answer[:2000]}\nContext:\n{ctx}\nScore correctness, "
            f"groundedness (claims supported by context), context relevance.")


@dataclass
class JudgeScores:
    correctness: float
    groundedness: float
    context_relevance: float


def run_judge(question: str, reference: str, answer: str,
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
                           float(data.get("context_relevance", 0.0)))
    except Exception:
        return JudgeScores(0.0, 0.0, 0.0)
