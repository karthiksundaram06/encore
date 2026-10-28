from pathlib import Path

import numpy as np
import pandas as pd

from encore.data import yambda
from encore.data.synthetic import generate
from encore.utils import RunPaths, log, save_json

FB_NONE, FB_SKIP, FB_COMPLETE, FB_LIKE, FB_DISLIKE = 0, 1, 2, 3, 4


def feedback_codes(complete, like, dislike):
    fb = np.where(complete > 0, FB_COMPLETE, FB_SKIP)
    fb = np.where(like > 0, FB_LIKE, fb)
    fb = np.where(dislike > 0, FB_DISLIKE, fb)
    return fb.astype(np.int8)


def assign_split(ts, cfg):
    unit = cfg.data.time_unit_seconds
    day = 86400 / unit
    gap = cfg.data.split.gap_minutes * 60 / unit
    t_min, t_max = int(ts.min()), int(ts.max())
    test_start = t_max - cfg.data.split.test_days * day
    val_start = test_start - cfg.data.split.val_days * day
    split = np.full(len(ts), -1, np.int8)
    split[ts < val_start - gap] = 0
    split[(ts >= val_start) & (ts < test_start - gap)] = 1
    split[ts >= test_start] = 2
    info = {
        "t_min": t_min, "t_max": t_max, "val_start": float(val_start), "test_start": float(test_start),
        "span_days": (t_max - t_min) / day, "gap_units": gap,
    }
    return split, info


def attach_feedback(listens, fb, window_units, name):
    listens[name] = np.int8(0)
    if fb is None or len(fb) == 0:
        return
    left = listens[["uid", "item_id", "timestamp"]].reset_index().sort_values("timestamp", kind="stable")
    right = fb[["uid", "item_id", "timestamp"]].sort_values("timestamp", kind="stable").assign(_hit=np.int8(1))
    merged = pd.merge_asof(
        left, right, on="timestamp", by=["uid", "item_id"],
        tolerance=int(window_units), direction="nearest",
    )
    hit = merged["_hit"].fillna(0).astype(np.int8).to_numpy()
    listens.loc[merged["index"].to_numpy(), name] = hit


def build_eval(ev, user_ptr, split_id, H, min_hist, max_users, seed):
    users, hist_rows, tgts = [], [], []
    for u in range(len(user_ptr) - 1):
        a, b = user_ptr[u], user_ptr[u + 1]
        sp = ev["split"][a:b]
        before = np.nonzero(sp < split_id)[0]
        inwin = np.nonzero(sp == split_id)[0]
        if len(before) < min_hist or len(inwin) == 0:
            continue
        w = a + inwin
        tgt = np.unique(ev["item"][w][ev["complete"][w] > 0])
        if len(tgt) == 0:
            continue
        users.append(u)
        hist_rows.append(a + before[-H:])
        tgts.append(tgt)
    rng = np.random.default_rng(seed)
    order = np.arange(len(users))
    if max_users and len(users) > max_users:
        order = np.sort(rng.choice(len(users), size=max_users, replace=False))
    n = len(order)
    out = {
        "uids": np.array([users[i] for i in order], dtype=np.int64),
        "hist_items": np.full((n, H), -1, np.int64),
        "hist_fb": np.zeros((n, H), np.int64),
        "hist_org": np.zeros((n, H), np.int64),
        "hist_complete": np.zeros((n, H), np.int64),
    }
    ptr = [0]
    flat = []
    for r, i in enumerate(order):
        idx = hist_rows[i]
        L = len(idx)
        out["hist_items"][r, H - L:] = ev["item"][idx]
        out["hist_fb"][r, H - L:] = ev["fb"][idx]
        out["hist_org"][r, H - L:] = ev["organic"][idx]
        out["hist_complete"][r, H - L:] = ev["complete"][idx]
        flat.append(tgts[i])
        ptr.append(ptr[-1] + len(tgts[i]))
    out["tgt_ptr"] = np.array(ptr, dtype=np.int64)
    out["tgt_items"] = np.concatenate(flat).astype(np.int64) if flat else np.zeros(0, np.int64)
    return out


def prepare(cfg):
    paths = RunPaths(cfg)
    raw_dir = Path(cfg.paths.raw)
    if cfg.data.source == "synthetic" and not (raw_dir / cfg.data.files.listens).exists():
        generate(cfg, raw_dir)

    # Raw events
    log("reading listens")
    listens = yambda.read_listens(cfg)
    log(f"{len(listens):,} listens, {listens.uid.nunique():,} users, {listens.item_id.nunique():,} items")
    split, tinfo = assign_split(listens.timestamp.to_numpy(), cfg)
    listens["split"] = split
    log(f"timeline spans {tinfo['span_days']:.1f} days (time_unit_seconds={cfg.data.time_unit_seconds})")
    thr = cfg.data.positive_threshold
    listens["complete"] = (listens.played_ratio_pct >= thr).astype(np.int8)

    window = cfg.data.like_window_minutes * 60 / cfg.data.time_unit_seconds
    for name in ("likes", "dislikes"):
        try:
            fb = yambda.read_feedback(cfg, name)
        except FileNotFoundError:
            log(f"{name} file missing; labels set to 0")
            fb = None
        attach_feedback(listens, fb, window, name[:-1])
        log(f"{name}: {int(listens[name[:-1]].sum()):,} listens labelled")

    # Catalog
    pos = listens.complete > 0
    counts = listens.loc[(listens.split == 0) & pos, "item_id"].value_counts()
    warm = counts[counts >= cfg.data.min_item_positives]
    if cfg.data.max_items:
        warm = warm.iloc[: cfg.data.max_items]
    warm_ids = warm.index.to_numpy()
    held = np.unique(listens.loc[(listens.split > 0) & pos, "item_id"].to_numpy())
    candidates = np.union1d(warm_ids, held)
    emb_ids, emb_vecs = yambda.read_embeddings(cfg, candidates)
    emb_index = pd.Index(emb_ids)
    cold_ids = np.setdiff1d(held, warm_ids)
    cold_ids = cold_ids[emb_index.get_indexer(cold_ids) >= 0]
    catalog = np.concatenate([warm_ids, cold_ids]).astype(np.int64)
    n_items = len(catalog)
    pos_in_emb = emb_index.get_indexer(catalog)
    audio = np.zeros((n_items, emb_vecs.shape[1]), np.float32)
    has_audio = pos_in_emb >= 0
    audio[has_audio] = emb_vecs[pos_in_emb[has_audio]]
    is_cold = np.zeros(n_items, bool)
    is_cold[len(warm_ids):] = True
    train_counts = counts.reindex(catalog).fillna(0).to_numpy().astype(np.int64)
    log(f"catalog: {len(warm_ids):,} warm + {len(cold_ids):,} cold items, {has_audio.mean():.1%} with audio")

    # Event arrays
    cat_index = pd.Index(catalog)
    listens["item"] = cat_index.get_indexer(listens.item_id.to_numpy())
    listens = listens[(listens.item >= 0) & (listens.split >= 0)]
    n_train = listens.loc[listens.split == 0].groupby("uid").size()
    keep_users = n_train[n_train >= cfg.data.min_user_train_events].index.to_numpy()
    listens = listens[listens.uid.isin(keep_users)].copy()
    user_raw = np.sort(keep_users)
    listens["u"] = pd.Index(user_raw).get_indexer(listens.uid.to_numpy())
    listens = listens.sort_values(["u", "timestamp"], kind="stable")
    ev = {
        "uid": listens.u.to_numpy(np.int32),
        "item": listens.item.to_numpy(np.int32),
        "ts": listens.timestamp.to_numpy(np.int64),
        "ratio": (listens.played_ratio_pct.to_numpy(np.float32) / 100.0).astype(np.float16),
        "complete": listens.complete.to_numpy(np.int8),
        "like": listens.like.to_numpy(np.int8),
        "dislike": listens.dislike.to_numpy(np.int8),
        "organic": listens.is_organic.to_numpy(np.int8),
        "split": listens.split.to_numpy(np.int8),
    }
    ev["fb"] = feedback_codes(ev["complete"], ev["like"], ev["dislike"])
    n_users = len(user_raw)
    user_ptr = np.concatenate([[0], np.cumsum(np.bincount(ev["uid"], minlength=n_users))]).astype(np.int64)
    log(f"kept {n_users:,} users, {len(ev['item']):,} events")

    np.savez(paths.data / "events.npz", user_ptr=user_ptr, user_raw=user_raw, **ev)
    np.savez(
        paths.data / "catalog.npz",
        raw_item_id=catalog, is_cold=is_cold, audio=audio, has_audio=has_audio, train_positives=train_counts,
    )

    # Eval sets
    H = cfg.data.eval_history
    for name, sid in (("val", 1), ("test", 2)):
        es = build_eval(ev, user_ptr, sid, H, cfg.data.eval_min_history, cfg.data.eval_max_users, cfg.seed + sid)
        np.savez(paths.data / f"eval_{name}.npz", **es)
        n_t = len(es["tgt_items"])
        cold_t = int(is_cold[es["tgt_items"]].sum()) if n_t else 0
        log(f"eval_{name}: {len(es['uids']):,} users, {n_t:,} targets ({cold_t:,} cold)")

    meta = {
        "source": cfg.data.source,
        "n_users": n_users,
        "n_items": n_items,
        "n_warm": int(len(warm_ids)),
        "n_cold": int(len(cold_ids)),
        "n_events": int(len(ev["item"])),
        "events_per_split": {str(s): int((ev["split"] == s).sum()) for s in (0, 1, 2)},
        "positive_rate": float(ev["complete"].mean()),
        "like_rate": float(ev["like"].mean()),
        "dislike_rate": float(ev["dislike"].mean()),
        "organic_rate": float(ev["organic"].mean()),
        "audio_dim": int(audio.shape[1]),
        "time": tinfo,
        "time_unit_seconds": cfg.data.time_unit_seconds,
        "eval_history": H,
    }
    save_json(meta, paths.data / "meta.json")
    log(f"prepared data in {paths.data}")
    return meta
