from app.evaluation.judge import _contexts, parse_judge_frameworks, primary_scores


def test_parse_all():
    assert parse_judge_frameworks("all") == ["custom", "ragas", "deepeval"]
    assert parse_judge_frameworks("custom, ragas") == ["custom", "ragas"]
    assert parse_judge_frameworks("") == ["custom"]


def test_parse_rejects_unknown():
    try:
        parse_judge_frameworks("foo")
        assert False
    except ValueError:
        pass


def test_primary_prefers_custom():
    scores = primary_scores({
        "ragas": {"correctness": 0.1, "groundedness": 0.1, "context_relevance": 0.1},
        "custom": {"correctness": 0.9, "groundedness": 0.8, "context_relevance": 0.7},
    })
    assert scores.correctness == 0.9


def test_latest_judge_packages():
    import deepeval
    import ragas

    ragas_parts = tuple(int(p) for p in ragas.__version__.split(".")[:2])
    deepeval_parts = tuple(int(p) for p in deepeval.__version__.split(".")[:1])
    assert ragas_parts >= (0, 4)
    assert deepeval_parts >= (4,)


def test_judge_contexts_are_capped():
    evidence = [(i, "x" * 8000) for i in range(20)]
    ctx = _contexts(evidence)
    assert len(ctx) == 5
    assert all(len(c) == 4000 for c in ctx)


def test_ragas_loop_survives_two_runs():
    from app.evaluation import ragas_judge

    async def ping():
        return 1

    loop = ragas_judge._loop()
    assert loop.run_until_complete(ping()) == 1
    assert loop.run_until_complete(ping()) == 1
    assert not loop.is_closed()
