from app.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def test_recall():
    assert recall_at_k([1, 2, 3], [2, 9], 5) == 0.5
    assert mrr_at_k([9, 2, 3], [2], 10) == 0.5
    assert ndcg_at_k([2, 9], [2], 10) == 1.0
