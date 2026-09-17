from app.retrieval.hybrid import dedupe_by_best_score, reciprocal_rank_fusion, weighted_fusion


def test_rrf_prefers_top_ranks():
    a = [(1, 9.0), (2, 1.0)]
    b = [(2, 0.9), (3, 0.1)]
    ranked = reciprocal_rank_fusion([a, b], k=60, top_k=3)
    assert ranked[0][0] in (1, 2)


def test_rrf_dedupes_duplicate_ids_in_one_list():
    a = [(1, 9.0), (1, 8.0), (2, 1.0)]
    ranked = reciprocal_rank_fusion([a], k=60, top_k=3)
    assert [pid for pid, _ in ranked].count(1) == 1


def test_dedupe_keeps_max_score():
    assert dict(dedupe_by_best_score([(1, 0.2), (1, 0.9), (2, 0.1)])) == {1: 0.9, 2: 0.1}


def test_weighted_respects_weights():
    lex = [(1, 10.0), (2, 0.0)]
    den = [(1, 0.0), (2, 1.0)]
    r = weighted_fusion(lex, den, w_bm25=1.0, w_dense=0.0, top_k=2)
    assert r[0][0] == 1
