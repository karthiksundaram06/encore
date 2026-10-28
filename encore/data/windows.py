import numpy as np

PAD_VALUES = {"item": -1}
FLOAT_KEYS = {"ratio"}


def train_windows(store, H, rng, users=None, max_split=0, min_len=2):
    if users is None:
        users = np.arange(store.n_users)
    starts, ends = [], []
    a_all = store.user_ptr[:-1]
    n_all = store.n_upto[max_split]
    for u in users:
        a = a_all[u]
        b = a + n_all[u]
        if b - a < min_len:
            continue
        off = int(rng.integers(0, H)) if b - a > H else 0
        e = np.arange(b - off, a + min_len - 1, -H)
        if off:
            e = np.concatenate([[b], e])
        starts.append(np.maximum(a, e - H))
        ends.append(e)
    if not starts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(starts).astype(np.int64), np.concatenate(ends).astype(np.int64)


def split_windows(store, H, split_id):
    starts, ends = [], []
    sp = store.ev["split"]
    step = max(1, H // 2)
    for u in range(store.n_users):
        a = store.user_ptr[u]
        lo = a + store.n_upto[split_id - 1][u] if split_id > 0 else a
        hi = a + store.n_upto[split_id][u]
        if hi <= lo or not (sp[lo:hi] == split_id).any():
            continue
        e = np.arange(hi, lo, -step)
        starts.append(np.maximum(a, e - H))
        ends.append(e)
    if not starts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(starts).astype(np.int64), np.concatenate(ends).astype(np.int64)


def gather(store, starts, ends, H, keys=("item", "fb", "organic", "complete", "like", "dislike", "ratio", "split")):
    idx = ends[:, None] - H + np.arange(H)[None, :]
    valid = idx >= starts[:, None]
    idx = np.where(valid, idx, 0)
    out = {"valid": valid}
    for k in keys:
        arr = store.ev[k][idx]
        if k in FLOAT_KEYS:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.int64)
        out[k] = np.where(valid, arr, PAD_VALUES.get(k, 0))
    out["uid"] = store.ev["uid"][np.maximum(ends - 1, 0)].astype(np.int64)
    return out


def iterate(store, starts, ends, H, batch_size, rng=None, **kw):
    n = len(starts)
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch_size):
        b = order[i:i + batch_size]
        yield gather(store, starts[b], ends[b], H, **kw)


def n_batches(n, batch_size):
    return (n + batch_size - 1) // batch_size
