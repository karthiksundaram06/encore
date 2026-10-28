import numpy as np
import torch

from encore.sid.embeddings import l2norm, ppmi, randomized_svd
from encore.sid.rqkmeans import add_dedup_level, kmeans, rq_kmeans
from encore.sid.trie import SIDTrie


def test_kmeans_recovers_clusters():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(5, 8)) * 10
    labels = rng.integers(0, 5, 500)
    x = torch.as_tensor(centers[labels] + rng.normal(size=(500, 8)) * 0.1, dtype=torch.float32)
    _, assign, _ = kmeans(x, 5, n_iter=20, seed=1)
    a = assign.numpy()
    for c in range(5):
        assert len(np.unique(a[labels == c])) == 1


def test_rq_kmeans_reduces_residual():
    x = l2norm(np.random.default_rng(0).normal(size=(400, 16))).astype(np.float32)
    codes, books, stats = rq_kmeans(x, [8, 8, 8], n_iter=10)
    assert codes.shape == (400, 3)
    ratios = [s["residual_var_ratio"] for s in stats]
    assert ratios[0] > ratios[1] > ratios[2]


def test_dedup_makes_ids_unique_and_ranks_by_popularity():
    codes = np.array([[0, 0], [0, 0], [0, 1], [0, 0]])
    pop = np.array([1.0, 5.0, 2.0, 3.0])
    full, D = add_dedup_level(codes, pop)
    assert D == 3
    assert len({tuple(r) for r in full}) == 4
    assert full[1, -1] == 0 and full[3, -1] == 1 and full[0, -1] == 2


def test_trie_tables_and_lookup(small_trie):
    t = small_trie
    for i, row in enumerate(t.codes):
        node = 0
        for l in range(t.n_levels):
            node = t.np_child[l][node, row[l]]
            assert node >= 0
        assert node == i
    assert (t.lookup(t.codes) == np.arange(len(t.codes))).all()
    bad = t.codes.copy()
    bad[:, -1] = t.sizes[-1] + 5
    assert (t.lookup(bad) == -1).all()


def test_trie_rejects_duplicates():
    codes = np.array([[0, 1], [0, 1]])
    try:
        SIDTrie(codes, [2, 2])
    except ValueError:
        return
    raise AssertionError("duplicates should raise")


def test_ppmi_and_svd_shapes():
    import scipy.sparse as sp

    C = sp.random(50, 50, density=0.2, random_state=0, format="csr")
    C = C + C.T
    P = ppmi(C)
    assert (P.data > 0).all()
    U, S, _ = randomized_svd(P, 5)
    assert U.shape == (50, 5) and (np.diff(S) <= 1e-6).all()
