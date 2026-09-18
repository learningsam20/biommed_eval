"""LLM query expansion — original query always retained."""
import json
from typing import List


EXPANSION_SYSTEM = (
    "You generate alternate biomedical search queries. "
    "Return ONLY a JSON list of strings, no other text."
)


def expansion_prompt(query: str, count: int) -> str:
    """User prompt asking the LLM for `count` alternate search queries as JSON."""
    return (f"Original question: {query}\nGenerate {count} diverse alternate "
            f"biomedical search queries (synonyms, MeSH-like terms, rephrasings). JSON list only.")


def parse_expansions(raw: str, count: int) -> List[str]:
    """Parse a JSON list from the LLM; fall back to line-splitting."""
    txt = raw.strip()
    if txt.startswith("```"):  # strip ```json fences
        txt = txt.strip("`").strip()
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()
    try:
        items = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
        if isinstance(items, list):
            return [str(x)[:300] for x in items[:count]]
    except Exception:
        pass
    # fallback: split lines
    lines = [l.strip("- •\t ") for l in raw.splitlines() if l.strip()]
    return lines[:count]


def expand_query(query: str, llm_client, count: int = 3,
                 temperature: float = 0.3) -> List[str]:
    """Return ``[original, ...alts]``. Original is always first. On LLM failure, ``[query]``."""
    try:
        raw = llm_client.chat(system=EXPANSION_SYSTEM,
                              user=expansion_prompt(query, count),
                              temperature=temperature)
        alts = parse_expansions(raw, count)
        seen = [a for a in alts if a and a.lower() != query.lower()]
        return [query] + seen[:count]
    except Exception:
        return [query]
