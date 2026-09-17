from app.generation.answer import (
    SENTINEL,
    extract_citations,
    generate_grounded_answer,
    validate_citations,
)


def test_citation_validation():
    ans = "Ret is involved [123] and also [999]."
    valid, hall = validate_citations(ans, [123, 456])
    assert valid == [123] and hall == [999]
    assert extract_citations("no cites") == []


class _FakeLLM:
    def __init__(self, text: str):
        self.text = text

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        return self.text


def test_hallucinated_cite_becomes_insufficient():
    text, cites, insuf = generate_grounded_answer(
        "q", [(123, "Ret pathway text")], _FakeLLM("Ret is involved [999]."), 0.0
    )
    assert text == SENTINEL and cites == [] and insuf is True


def test_valid_cites_pass():
    text, cites, insuf = generate_grounded_answer(
        "q", [(123, "Ret pathway text")], _FakeLLM("Ret is involved [123]."), 0.0
    )
    assert not insuf and cites == [123] and "[123]" in text


def test_uncited_answer_insufficient():
    text, cites, insuf = generate_grounded_answer(
        "q", [(123, "Ret pathway text")], _FakeLLM("Ret is involved somehow."), 0.0
    )
    assert text == SENTINEL and cites == [] and insuf is True
