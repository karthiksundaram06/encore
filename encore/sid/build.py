from pathlib import Path

import numpy as np

from encore.data.store import Store
from encore.sid.embeddings import audio_pca, collaborative_embeddings, fuse, l2norm, sasrec_embeddings
from encore.sid.rqkmeans import add_dedup_level, rq_kmeans
from encore.sid.trie import SIDTrie
from encore.utils import RunPaths, Timer, get_device, log, save_json


def _coherence(fused, first_code, rng, n=20000):
    a = rng.integers(len(fused), size=n)
    b = rng.integers(len(fused), size=n)
    rand = float((fused[a] * fused[b]).sum(1).mean())
    same = []
    for c in np.unique(first_code)[:256]:
        idx = np.nonzero(first_code == c)[0]
        if len(idx) < 2:
            continue
        i = rng.choice(idx, size=min(200, len(idx)))
        j = rng.choice(idx, size=len(i))
        same.append((fused[i] * fused[j]).sum(1).mean())
    return {"cos_same_first_code": float(np.mean(same)), "cos_random": rand}


def _genre_purity(raw_dir, store, codes):
    path = Path(raw_dir) / "oracle.npz"
    if not path.exists():
        return None
    import pandas as pd

    z = np.load(path)
    genre = z["item_genre"][pd.Index(z["item_raw"]).get_indexer(store.raw_item_id)]
    pur = []
    for c in np.unique(codes[:, 0]):
        g = genre[codes[:, 0] == c]
        pur.append(np.bincount(g).max() / len(g) * len(g))
    return float(np.sum(pur) / len(codes))


def build_sids(cfg):
    paths = RunPaths(cfg)
    store = Store(paths.data)
    s = cfg.sid
    device = get_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)

    source = s.collab_source
    if source == "sasrec" and not (paths.ckpt / "sasrec.pt").exists():
        log("sasrec.pt not found; falling back to PPMI collaborative embeddings")
        source = "ppmi"
    with Timer() as t_c:
        if source == "sasrec":
            collab, has_collab = sasrec_embeddings(cfg, paths, store)
        else:
            collab, has_collab = collaborative_embeddings(store, cfg)
    log(f"collaborative embeddings ({source}): {has_collab.mean():.1%} of items covered ({t_c.seconds:.1f}s)")

    audio_p, explained = audio_pca(store.audio, store.has_audio, s.audio_dim, cfg.seed)
    fused, r2 = fuse(audio_p, store.has_audio, collab, has_collab, s.audio_weight, s.ridge_lambda)
    orphan = ~(store.has_audio | has_collab)
    if orphan.any():
        fused[orphan] = l2norm(rng.standard_normal((int(orphan.sum()), fused.shape[1]))).astype(np.float32)
        log(f"{int(orphan.sum())} items have neither audio nor collaborative signal; random vectors used")
    log(f"fused embeddings: dim={fused.shape[1]}, audio PCA explained={explained:.2f}, audio->collab R2={r2}")

    with Timer() as t_k:
        codes, books, stats = rq_kmeans(
            fused, s.levels, s.kmeans_iters, cfg.seed, s.kmeans_init_sample, device=device
        )
    full, D = add_dedup_level(codes, store.train_positives)
    level_sizes = [int(x) for x in s.levels] + [D]
    trie = SIDTrie(full, level_sizes)
    collided = float((full[:, -1] > 0).mean())
    log(f"RQ-KMeans {s.levels} in {t_k.seconds:.1f}s; dedup level size {D}; {collided:.1%} items needed dedup")
    if D > s.dedup_warn:
        log(f"WARNING: dedup level size {D} > {s.dedup_warn}; consider larger codebooks")
    for st in stats:
        log(f"  level {st['level']}: utilization={st['utilization']:.2f} perplexity={st['perplexity']:.1f} "
            f"residual_var={st['residual_var_ratio']:.3f}")

    np.savez(
        paths.sid / "sid.npz",
        codes=full, level_sizes=np.array(level_sizes), fused=fused, collab=collab,
        has_collab=has_collab, audio_pca=audio_p, **{f"codebook_{i}": b for i, b in enumerate(books)},
    )
    report = {
        "n_items": int(store.n_items),
        "levels": level_sizes,
        "vocab_size": trie.vocab_size,
        "dedup_size": D,
        "items_with_dedup": collided,
        "unique_prefixes": int(len(np.unique(codes, axis=0))),
        "collab_coverage": float(has_collab.mean()),
        "audio_explained_var": explained,
        "audio_to_collab_r2": r2,
        "level_stats": stats,
        "coherence": _coherence(fused, full[:, 0], rng),
        "first_code_genre_purity": _genre_purity(cfg.paths.raw, store, full),
        "cold_items": int(store.is_cold.sum()),
    }
    save_json(report, paths.reports / "sid_report.json")
    log(f"semantic IDs saved to {paths.sid}")
    return report
