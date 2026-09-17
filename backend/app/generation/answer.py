"""Grounded answer generation + citation validation."""
import re
from typing import List, Tuple

CITATION_RE = re.compile(r"\[(?:Passage\s+)?(\d+)\]")
SENTINEL = "INSUFFICIENT_EVIDENCE"

ANSWER_SYSTEM = (
    "You are a biomedical QA assistant. Answer ONLY from the provided passages. "
    "Cite every factual claim inline with the bare passage number in square brackets, "
    "e.g. [17965226]. Never write the word Passage inside the brackets. "
    "If the passages do not support an answer, reply exactly: INSUFFICIENT_EVIDENCE"
)


def build_answer_prompt(query: str, evidence: List[Tuple[int, str]]) -> str:
    ctx = "\n\n".join(f"[Passage {pid}]\n{text[:2000]}" for pid, text in evidence)
    return f"Question: {query}\n\nRetrieved passages:\n{ctx}\n\nAnswer with inline [id] citations:"


def extract_citations(text: str) -> List[int]:
    return [int(m) for m in CITATION_RE.findall(text)]


def validate_citations(answer: str, retrieved_ids: List[int]) -> Tuple[List[int], List[int]]:
    """Return (valid, hallucinated) citation IDs."""
    retrieved = set(retrieved_ids)
    cited = extract_citations(answer)
    return ([c for c in cited if c in retrieved], [c for c in cited if c not in retrieved])


def generate_grounded_answer(query: str, evidence: List[Tuple[int, str]],
                             llm_client, temperature: float = 0.1) -> Tuple[str, List[int], bool]:
    if not evidence:
        return SENTINEL, [], True
    raw = llm_client.chat(system=ANSWER_SYSTEM,
                          user=build_answer_prompt(query, evidence),
                          temperature=temperature).strip()
    if SENTINEL in raw or not raw:
        return SENTINEL, [], True
    valid, hallucinated = validate_citations(raw, [pid for pid, _ in evidence])
    # Spec / architecture: any cite outside retrieved context → insufficient
    if hallucinated:
        return SENTINEL, [], True
    if not valid:
        # Answer claimed facts without any valid citation → insufficient
        return SENTINEL, [], True
    return raw, list(dict.fromkeys(valid)), False
