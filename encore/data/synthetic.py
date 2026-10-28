from pathlib import Path

import numpy as np
import pandas as pd

from encore.utils import log, save_json

TIME_UNIT = 5


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def engagement_probs(score, theta, p):
    pc = _sigmoid(p["sharpness"] * (score - theta))
    pl = _sigmoid(p["sharpness"] * (score - p["like_threshold"]))
    pd = _sigmoid(p["sharpness"] * (p["dislike_threshold"] - score))
    return pc, pl, pd


def expected_utility(pc, pl, pd, weights):
    ratio = pc * 0.75 + (1 - pc) * 0.125
    return (
        weights["complete"] * pc
        + weights["ratio"] * ratio
        + weights["like"] * pc * pl
        - weights["dislike"] * (1 - pc) * pd
    )


def true_score(u_eff, v_items, q_items, recent, items, fatigue):
    dot = np.einsum("nk,npk->np", u_eff, v_items)
    reps = (items[:, :, None] == recent[:, None, :]).sum(-1)
    return dot + q_items - fatigue * reps


def _session_times(rng, n_events, start_day, n_days, session_len):
    n_sessions = max(1, int(np.ceil(n_events / session_len)))
    starts = np.sort(rng.uniform(start_day, n_days, n_sessions)) * 86400.0
    sizes = np.full(n_sessions, n_events // n_sessions)
    sizes[: n_events % n_sessions] += 1
    out = []
    for s, n in zip(starts, sizes):
        gaps = rng.uniform(150, 300, n)
        out.append(s + np.concatenate([[0.0], np.cumsum(gaps[:-1])]))
    t = np.concatenate(out)
    t = np.minimum(t, n_days * 86400.0 - 1)
    return np.maximum.accumulate(t)


def simulate(sc, seed):
    rng = np.random.default_rng(seed)
    U, I, k, G = sc.n_users, sc.n_items, sc.latent_dim, sc.n_genres
    p = {
        "sharpness": sc.sharpness,
        "like_threshold": sc.like_threshold,
        "dislike_threshold": sc.dislike_threshold,
        "state_rate": sc.state_rate,
        "state_mix": 0.3,
        "fatigue": sc.fatigue,
        "recent_window": 10,
    }

    # Item world
    genre_w = rng.dirichlet(np.full(G, 0.8))
    item_genre = rng.choice(G, size=I, p=genre_w)
    centers = _unit(rng.normal(size=(G, k)))
    v = _unit(centers[item_genre] + 0.6 * _unit(rng.normal(size=(I, k)))) * 1.5
    q = rng.normal(0.0, 0.4, size=I)
    late = rng.random(I) < sc.late_release_frac
    release_s = np.where(late, (sc.n_days - sc.late_release_days * rng.random(I)) * 86400.0, 0.0)
    exposure = np.exp(sc.exposure_sigma * rng.normal(size=I) + 0.5 * q) * np.where(late, 4.0, 1.0)
    cdf = np.cumsum(exposure / exposure.sum())
    w_audio = rng.normal(size=(k, sc.audio_dim)) / np.sqrt(k)
    w_q = rng.normal(size=sc.audio_dim)
    audio = v @ w_audio + 0.5 * q[:, None] * w_q[None] + sc.audio_noise * rng.normal(size=(I, sc.audio_dim))
    length_s = rng.uniform(120, 330, I).astype(np.int32)

    # User world
    u = np.zeros((U, k))
    for i in range(U):
        n_g = rng.integers(1, 4)
        gs = rng.choice(G, size=n_g, p=genre_w)
        w = rng.dirichlet(np.ones(n_g))
        u[i] = (w[:, None] * centers[gs]).sum(0)
    u = _unit(u + 0.35 * _unit(rng.normal(size=(U, k)))) * 1.5
    theta = rng.normal(0.5, 0.3, U)
    u_log = _unit(u + sc.logging_noise * _unit(rng.normal(size=(U, k)))) * 1.5
    n_events = np.clip(
        rng.lognormal(np.log(sc.events_mean), 0.6, U).astype(int), sc.events_min, sc.events_max
    )
    start_day = rng.uniform(0, sc.n_days * 0.3, U)
    max_e = int(n_events.max())
    times = np.full((U, max_e), np.nan)
    for i in range(U):
        times[i, : n_events[i]] = _session_times(rng, n_events[i], start_day[i], sc.n_days, sc.session_len)

    # Rollout
    m = u.copy()
    recent = np.full((U, p["recent_window"]), -1)
    ring = np.full((U, 20), -1)
    ring_pos = np.zeros(U, dtype=int)
    rows = {k_: [] for k_ in ("uid", "t", "item", "organic", "ratio", "like", "dislike")}
    for t in range(max_e):
        act = np.nonzero(n_events > t)[0]
        n = len(act)
        now = times[act, t]
        r = rng.random(n)
        has_ring = (ring[act] >= 0).any(1)
        is_rep = (r < sc.repeat_rate) & has_ring
        is_org = (~is_rep) & (r < sc.repeat_rate + sc.organic_rate)
        is_rec = ~(is_rep | is_org)
        u_eff = (1 - p["state_mix"]) * u[act] + p["state_mix"] * m[act]
        chosen = np.zeros(n, dtype=int)

        if is_org.any():
            idx = np.nonzero(is_org)[0]
            pool = rng.integers(I, size=(len(idx), sc.organic_pool))
            s = true_score(u_eff[idx], v[pool], q[pool], recent[act[idx]], pool, p["fatigue"])
            s = np.where(release_s[pool] <= now[idx, None], s, -np.inf)
            g = rng.gumbel(size=s.shape)
            chosen[idx] = pool[np.arange(len(idx)), np.argmax(sc.organic_temperature * s + g, 1)]

        if is_rec.any():
            idx = np.nonzero(is_rec)[0]
            pool = np.searchsorted(cdf, rng.random((len(idx), sc.reco_pool)))
            pool = np.minimum(pool, I - 1)
            s = sc.logging_personalization * np.einsum("nk,npk->np", u_log[act[idx]], v[pool])
            s = s + np.log(exposure[pool]) * 0.3 + rng.gumbel(size=s.shape)
            s = np.where(release_s[pool] <= now[idx, None], s, -np.inf)
            chosen[idx] = pool[np.arange(len(idx)), np.argmax(s, 1)]

        if is_rep.any():
            idx = np.nonzero(is_rep)[0]
            rr = ring[act[idx]]
            pick = rng.random(rr.shape) * (rr >= 0)
            chosen[idx] = rr[np.arange(len(idx)), np.argmax(pick, 1)]

        s_true = true_score(u_eff, v[chosen][:, None], q[chosen][:, None], recent[act], chosen[:, None], p["fatigue"])[:, 0]
        pc, pl, pd = engagement_probs(s_true, theta[act], p)
        complete = rng.random(n) < pc
        like = complete & (rng.random(n) < pl)
        dislike = (~complete) & (rng.random(n) < pd)
        ratio = np.where(complete, rng.uniform(50, 100, n), 50 * rng.beta(1.0, 3.0, n))

        rows["uid"].append(act)
        rows["t"].append(now)
        rows["item"].append(chosen)
        rows["organic"].append(~is_rec)
        rows["ratio"].append(ratio)
        rows["like"].append(like)
        rows["dislike"].append(dislike)

        ca = act[complete]
        m[ca] = (1 - p["state_rate"]) * m[ca] + p["state_rate"] * v[chosen[complete]]
        recent[act] = np.concatenate([recent[act, 1:], chosen[:, None]], 1)
        ring[ca, ring_pos[ca] % 20] = chosen[complete]
        ring_pos[ca] += 1

    ev = {k_: np.concatenate(v_) for k_, v_ in rows.items()}
    world = {
        "v": v, "q": q, "u": u, "theta": theta, "audio": audio, "length_s": length_s,
        "release_s": release_s, "exposure": exposure, "item_genre": item_genre,
    }
    return ev, world, p


def generate(cfg, raw_dir):
    sc = cfg.data.synthetic
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log(f"simulating synthetic world: {sc.n_users} users, {sc.n_items} items")
    ev, world, params = simulate(sc, cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    item_raw = rng.choice(sc.n_items * 13, size=sc.n_items, replace=False) + 1000
    user_raw = rng.choice(sc.n_users * 7, size=sc.n_users, replace=False) + 100

    ts = (ev["t"] / TIME_UNIT).astype(np.int64)
    listens = pd.DataFrame({
        "uid": user_raw[ev["uid"]].astype(np.uint32),
        "timestamp": ts.astype(np.uint32),
        "item_id": item_raw[ev["item"]].astype(np.uint32),
        "is_organic": ev["organic"].astype(np.uint8),
        "played_ratio_pct": np.round(ev["ratio"]).astype(np.uint16),
        "track_length_seconds": world["length_s"][ev["item"]].astype(np.uint32),
    }).sort_values(["timestamp", "uid"], kind="stable")
    listens.to_parquet(raw_dir / "listens.parquet", index=False)

    for name, flag, delay in (("likes", ev["like"], 6), ("dislikes", ev["dislike"], 2)):
        df = pd.DataFrame({
            "uid": user_raw[ev["uid"][flag]].astype(np.uint32),
            "timestamp": (ts[flag] + delay).astype(np.uint32),
            "item_id": item_raw[ev["item"][flag]].astype(np.uint32),
            "is_organic": ev["organic"][flag].astype(np.uint8),
        }).sort_values("timestamp", kind="stable")
        df.to_parquet(raw_dir / f"{name}.parquet", index=False)

    audio = world["audio"].astype(np.float32)
    emb = pd.DataFrame({
        "item_id": item_raw.astype(np.uint32),
        "embed": list(audio),
        "normalized_embed": list(_unit(audio).astype(np.float32)),
    })
    emb.to_parquet(raw_dir / "embeddings.parquet", index=False)

    np.savez_compressed(
        raw_dir / "oracle.npz",
        item_raw=item_raw, user_raw=user_raw, v=world["v"], q=world["q"], u=world["u"],
        theta=world["theta"], item_genre=world["item_genre"], release_s=world["release_s"],
    )
    save_json(params, raw_dir / "oracle_params.json")
    stats = {
        "events": int(len(listens)),
        "complete_rate": float((listens.played_ratio_pct >= 50).mean()),
        "organic_rate": float(listens.is_organic.mean()),
        "likes": int(ev["like"].sum()),
        "dislikes": int(ev["dislike"].sum()),
    }
    save_json(stats, raw_dir / "synthetic_stats.json")
    log(f"synthetic data written to {raw_dir}: {stats}")
    return stats
