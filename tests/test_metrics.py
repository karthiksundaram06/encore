import numpy as np

from encore.eval.metrics import beyond_accuracy, ranking_metrics
from encore.world.report import auc, ece


class _ES:
    def __init__(self, targets):
        self.t = targets

    def targets(self, i):
        return np.asarray(self.t[i])


def test_ndcg_recall_known_values():
    recs = np.array([[1, 2, 3], [4, 5, 6]])
    es = _ES([[1], [6, 7]])
    m = ranking_metrics(recs, es, [3])
    assert np.isclose(m["recall@3"], (1.0 + 0.5) / 2)
    ndcg2 = (1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert np.isclose(m["ndcg@3"], (1.0 + ndcg2) / 2)
    assert m["hr@3"] == 1.0


def test_cold_recall_and_coverage():
    recs = np.array([[0, 3], [1, 2]])
    es = _ES([[3], [0]])
    cold = np.array([False, False, False, True])
    m = ranking_metrics(recs, es, [2], cold)
    assert m["cold_recall@2"] == 1.0 and m["cold_users"] == 1
    b = beyond_accuracy(recs, 4, np.full(4, 0.25), np.eye(4), cold, k=2)
    assert b["coverage@2"] == 1.0 and b["ild@2"] == 1.0 and np.isclose(b["novelty@2"], 2.0)


def test_auc_and_ece():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert ece(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0


def test_discovery_metrics_and_repeat_share():
    from encore.eval.metrics import discovery_metrics, repeat_share

    recs = np.array([[5, 6, 1], [7, 8, 9]])
    es = _ES([[1, 5], [9]])
    hist = [np.array([1]), np.array([9])]
    m = discovery_metrics(recs, es, hist, [3], np.zeros(10, bool))
    assert m["new_users"] == 1 and m["new_recall@3"] == 1.0
    assert np.isclose(repeat_share(recs, hist, 3), (1 / 3 + 1 / 3) / 2)


def test_history_rec_orders_by_count_then_recency():
    from encore.eval.recommenders import HistoryRec

    batch = {"items": np.array([[3, 4, 3, 5, 4, 6]]), "complete": np.array([[1, 1, 1, 1, 1, 0]])}
    out = HistoryRec().recommend(batch, 4)
    assert out[0].tolist() == [4, 3, 5, -1]
    out = HistoryRec().recommend(batch, 2, exclude=[np.array([4])])
    assert out[0].tolist() == [3, 5]
