import numpy as np
import scipy.sparse as sp

from encore.utils import log


def l2norm(x, eps=1e-9):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def randomized_svd(A, k, oversample=10, n_iter=4, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[1]
    omega = rng.standard_normal((n, k + oversample)).astype(np.float32)
    Y = A @ omega
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(Y)
        Z, _ = np.linalg.qr(A.T @ Y)
        Y = A @ Z
    Q, _ = np.linalg.qr(Y)
    B = (A.T @ Q).T
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], S[:k], Vt[:k]


def positive_train_sequences(store):
    ev = store.ev
    mask = (ev["split"] == 0) & (ev["complete"] > 0)
    return ev["item"][mask].astype(np.int64), ev["uid"][mask].astype(np.int64)


def cooccurrence(items, users, n_items, window, max_pairs, seed):
    rng = np.random.default_rng(seed)
    total = sum(2 * int((users[:-o] == users[o:]).sum()) for o in range(1, window + 1))
    keep_p = min(1.0, max_pairs / max(total, 1))
    C = sp.csr_matrix((n_items, n_items), dtype=np.float32)
    for o in range(1, window + 1):
        same = users[:-o] == users[o:]
        a, b = items[:-o][same], items[o:][same]
        if keep_p < 1.0:
            s = rng.random(len(a)) < keep_p
            a, b = a[s], b[s]
        w = np.full(len(a), 1.0 / o, np.float32)
        C = C + sp.coo_matrix((w, (a, b)), shape=(n_items, n_items)).tocsr()
        C = C + sp.coo_matrix((w, (b, a)), shape=(n_items, n_items)).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    return C


def ppmi(C, alpha=0.75):
    C = C.tocoo()
    row = np.asarray(C.sum(1)).ravel()
    col = np.asarray(C.sum(0)).ravel() ** alpha
    col_total = col.sum()
    pmi = np.log(C.data * col_total / (row[C.row] * col[C.col]) + 1e-12)
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep].astype(np.float32), (C.row[keep], C.col[keep])), shape=C.shape)


def collaborative_embeddings(store, cfg):
    s = cfg.sid
    items, users = positive_train_sequences(store)
    C = cooccurrence(items, users, store.n_items, s.ppmi_window, s.ppmi_max_pairs, cfg.seed)
    P = ppmi(C)
    log(f"co-occurrence: {C.nnz:,} nonzeros, PPMI {P.nnz:,} nonzeros")
    k = min(s.collab_dim, store.n_items - 2)
    U, S, _ = randomized_svd(P, k, seed=cfg.seed)
    emb = (U * np.sqrt(S)[None]).astype(np.float32)
    has = np.asarray(P.getnnz(axis=1)).ravel() > 0
    emb[~has] = 0.0
    return emb, has


def sasrec_embeddings(cfg, paths, store):
    import torch

    ck = torch.load(paths.ckpt / "sasrec.pt", map_location="cpu", weights_only=False)
    w = ck["state_dict"]["item_emb.weight"][1:].float().numpy()
    has = ~store.is_cold
    w = w.copy()
    w[~has] = 0.0
    return w, has


def audio_pca(audio, has_audio, dim, seed):
    X = audio[has_audio].astype(np.float64)
    mu = X.mean(0)
    rng = np.random.default_rng(seed)
    sample = X[rng.choice(len(X), size=min(len(X), 200_000), replace=False)] - mu
    _, S, Vt = np.linalg.svd(sample, full_matrices=False)
    dim = min(dim, Vt.shape[0])
    out = np.zeros((len(audio), dim), np.float32)
    out[has_audio] = ((X - mu) @ Vt[:dim].T).astype(np.float32)
    explained = float((S[:dim] ** 2).sum() / (S ** 2).sum())
    return out, explained


def _bias(X):
    return np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1)


def ridge_map(X, Y, lam):
    Xb = _bias(X)
    reg = lam * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)


def fuse(audio_p, has_audio, collab, has_collab, audio_weight, lam, seed=0):
    a = l2norm(audio_p)
    c = l2norm(collab)
    both = has_audio & has_collab
    fill = has_audio & ~has_collab
    mapped_r2 = None
    if both.sum() > 2 * a.shape[1]:
        idx = np.nonzero(both)[0]
        rng = np.random.default_rng(seed)
        held = rng.random(len(idx)) < 0.1
        W = ridge_map(a[idx[~held]], c[idx[~held]], lam)
        yt = c[idx[held]]
        pred = _bias(a[idx[held]]) @ W
        mapped_r2 = float(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
        if fill.any():
            W = ridge_map(a[both], c[both], lam)
            c = c.copy()
            c[fill] = l2norm(_bias(a[fill]) @ W)
    fused = np.concatenate([np.sqrt(audio_weight) * a, np.sqrt(1 - audio_weight) * c], axis=1)
    return l2norm(fused).astype(np.float32), mapped_r2
