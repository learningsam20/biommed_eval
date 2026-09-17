from app.retrieval.expansion import expand_query, parse_expansions


class _FakeLLM:
    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        return '["synonym query", "MeSH rephrase"]'


def test_parse_expansions_json_list():
    assert parse_expansions('["a", "b", "c"]', 2) == ["a", "b"]


def test_expand_retains_original_then_alts():
    out = expand_query("original question", _FakeLLM(), count=2)
    assert out[0] == "original question"
    assert "synonym query" in out
    assert len(out) == 3


def test_expand_drops_duplicate_of_original():
    class Dup:
        def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
            return '["original question", "other"]'

    out = expand_query("original question", Dup(), count=3)
    assert out[0] == "original question"
    assert out.count("original question") == 1
