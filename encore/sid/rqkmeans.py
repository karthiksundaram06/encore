import numpy as np
import torch


def _sq_dists(x, c):
    return (x * x).sum(1, keepdim=True) - 2 * x @ c.T + (c * c).sum(1)[None]


def _assign(x, c, chunk=65536):
    out_idx = torch.empty(len(x), dtype=torch.long, device=x.device)
    out_d = torch.empty(len(x), dtype=x.dtype, device=x.device)
    for i in range(0, len(x), chunk):
        d = _sq_dists(x[i:i + chunk], c)
        out_d[i:i + chunk], out_idx[i:i + chunk] = d.min(1)
    return out_idx, out_d.clamp(min=0)


def kmeans_pp_init(x, k, gen):
    n = len(x)
    first = torch.randint(n, (1,), generator=gen, device="cpu").item()
    centers = [x[first]]
    d = ((x - x[first]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d / d.sum() if d.sum() > 0 else torch.full_like(d, 1.0 / n)
        idx = torch.multinomial(probs.cpu(), 1, generator=gen).item()
        centers.append(x[idx])
        d = torch.minimum(d, ((x - x[idx]) ** 2).sum(1))
    return torch.stack(centers)


def kmeans(x, k, n_iter=25, seed=0, init_sample=100_000):
    gen = torch.Generator().manual_seed(seed)
    n = len(x)
    k = min(k, n)
    sample = x
    if n > init_sample:
        sample = x[torch.randperm(n, generator=gen)[:init_sample].to(x.device)]
    c = kmeans_pp_init(sample, k, gen)
    for _ in range(n_iter):
        assign, dist = _assign(x, c)
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).to(x.dtype)
        empty = counts == 0
        c = torch.where(empty[:, None], c, sums / counts.clamp(min=1)[:, None])
        if empty.any():
            far = torch.topk(dist, int(empty.sum())).indices
            c[empty] = x[far]
    assign, dist = _assign(x, c)
    return c, assign, dist


def rq_kmeans(x, levels, n_iter=25, seed=0, init_sample=100_000, device="cpu"):
    r = torch.as_tensor(x, dtype=torch.float32, device=device).clone()
    total_var = float(((r - r.mean(0)) ** 2).sum(1).mean())
    codes, books, stats = [], [], []
    for li, k in enumerate(levels):
        c, a, d = kmeans(r, k, n_iter, seed + li, init_sample)
        r = r - c[a]
        counts = torch.bincount(a, minlength=k).double()
        p = counts / counts.sum()
        nz = p[p > 0]
        stats.append({
            "level": li,
            "codebook_size": k,
            "utilization": float((counts > 0).double().mean()),
            "perplexity": float(torch.exp(-(nz * nz.log()).sum())),
            "residual_var_ratio": float((r * r).sum(1).mean()) / max(total_var, 1e-12),
        })
        codes.append(a.cpu().numpy())
        books.append(c.cpu().numpy())
    return np.stack(codes, 1).astype(np.int64), books, stats


def add_dedup_level(codes, popularity):
    levels = codes.shape[1]
    radix = codes.max(0) + 1
    key = np.zeros(len(codes), np.int64)
    for l in range(levels):
        key = key * int(radix[l]) + codes[:, l]
    order = np.lexsort((-popularity, key))
    sk = key[order]
    start = np.r_[True, sk[1:] != sk[:-1]]
    group_start = np.maximum.accumulate(np.where(start, np.arange(len(sk)), 0))
    rank = np.arange(len(sk)) - group_start
    dedup = np.empty(len(codes), np.int64)
    dedup[order] = rank
    return np.concatenate([codes, dedup[:, None]], 1), int(dedup.max()) + 1
