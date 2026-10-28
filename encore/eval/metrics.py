import numpy as np


def ranking_metrics(recs, eval_set, ks, is_cold=None):
    kmax = max(ks)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    acc = {f"{m}@{k}": [] for k in ks for m in ("recall", "ndcg", "hr")}
    cold = {f"cold_recall@{k}": [] for k in ks}
    for i in range(len(recs)):
        T = eval_set.targets(i)
        r = recs[i, :kmax]
        hit = np.isin(r, T) & (r >= 0)
        for k in ks:
            h = hit[:k]
            idcg = disc[: min(len(T), k)].sum()
            acc[f"ndcg@{k}"].append((h * disc[:k]).sum() / idcg)
            acc[f"recall@{k}"].append(h.sum() / len(T))
            acc[f"hr@{k}"].append(float(h.any()))
        if is_cold is not None:
            Tc = T[is_cold[T]]
            if len(Tc):
                for k in ks:
                    cold[f"cold_recall@{k}"].append(np.isin(r[:k], Tc).sum() / len(Tc))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    if is_cold is not None:
        for k, v in cold.items():
            out[k] = float(np.mean(v)) if v else float("nan")
        out["cold_users"] = len(next(iter(cold.values()))) if cold else 0
    return out


def discovery_metrics(recs, eval_set, hist_sets, ks, is_cold):
    kmax = max(ks)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    acc = {f"new_{m}@{k}": [] for k in ks for m in ("recall", "ndcg")}
    cold = {f"new_cold_recall@{k}": [] for k in ks}
    for i in range(len(recs)):
        T = eval_set.targets(i)
        T = T[~np.isin(T, hist_sets[i])]
        if len(T) == 0:
            continue
        r = recs[i, :kmax]
        hit = np.isin(r, T) & (r >= 0)
        for k in ks:
            h = hit[:k]
            acc[f"new_ndcg@{k}"].append((h * disc[:k]).sum() / disc[: min(len(T), k)].sum())
            acc[f"new_recall@{k}"].append(h.sum() / len(T))
        Tc = T[is_cold[T]]
        if len(Tc):
            for k in ks:
                cold[f"new_cold_recall@{k}"].append(np.isin(r[:k], Tc).sum() / len(Tc))
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in {**acc, **cold}.items()}
    out["new_users"] = len(acc[f"new_ndcg@{ks[0]}"])
    return out


def repeat_share(recs, hist_sets, k=10):
    shares = []
    for r, h in zip(recs[:, :k], hist_sets):
        r = r[r >= 0]
        if len(r):
            shares.append(np.isin(r, h).mean())
    return float(np.mean(shares)) if shares else float("nan")


def beyond_accuracy(recs, n_items, popularity, embeddings, is_cold, k=10):
    r = recs[:, :k]
    valid = r >= 0
    flat = r[valid]
    out = {
        f"coverage@{k}": float(len(np.unique(flat)) / n_items),
        f"novelty@{k}": float(np.mean(-np.log2(popularity[flat]))) if len(flat) else float("nan"),
        f"cold_share@{k}": float(is_cold[flat].mean()) if len(flat) else float("nan"),
    }
    if embeddings is not None:
        ild = []
        for row in r:
            row = row[row >= 0]
            if len(row) < 2:
                continue
            e = embeddings[row]
            sim = e @ e.T
            n = len(row)
            ild.append(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))
        out[f"ild@{k}"] = float(np.mean(ild)) if ild else float("nan")
    return out
