import numpy as np

from encore.baselines.models import SASRec, TwoTower, popularity_ranking
from encore.data.store import Store
from encore.eval.evaluate import evaluate
from encore.eval.recommenders import PopularityRec, SASRecRec, TwoTowerRec
from encore.train_loop import fit
from encore.utils import RunPaths, count_params, get_device, log, save_json


def two_tower_content(store):
    a = store.audio.astype(np.float32)
    a = a - a[store.has_audio].mean(0)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    a[~store.has_audio] = 0.0
    return a


def _val_ndcg(rec, es, store):
    res, _ = evaluate(rec, es, store, [10], batch_size=128)
    return res["ndcg@10"]


def train_baselines(cfg, only=None):
    paths = RunPaths(cfg)
    store = Store(paths.data)
    device = get_device(cfg.device)
    val = store.eval_set("val").subset(cfg.generator.eval_users, cfg.seed)
    only = only or ["popularity", "sasrec", "two_tower"]
    summary = {}

    if "popularity" in only:
        ranking = popularity_ranking(store, cfg.popularity.window_days)
        np.save(paths.ckpt / "popularity.npy", ranking[:1000])
        summary["popularity"] = _val_ndcg(PopularityRec(ranking[:1000]), val, store)
        log(f"popularity val ndcg@10={summary['popularity']:.4f}")

    if "sasrec" in only:
        c = cfg.sasrec
        model = SASRec(store.n_items, c.d_model, c.n_layers, c.n_heads, c.history + 1, c.dropout).to(device)
        log(f"SASRec: {count_params(model) / 1e6:.2f}M params")
        neg = c.num_negatives
        summary["sasrec"] = fit(
            model, lambda b: model.loss(b, neg, store.n_warm), store, c, c.history, device, cfg.amp,
            lambda m: _val_ndcg(SASRecRec(m, store.is_cold, c.history, device), val, store),
            paths.ckpt / "sasrec.pt", paths.logs / "sasrec.jsonl", cfg.seed,
        )

    if "two_tower" in only:
        c = cfg.two_tower
        content = two_tower_content(store)
        model = TwoTower(store.n_items, content, store.is_cold, store.popularity(), c.d_model,
                         c.id_dropout, c.temperature).to(device)
        log(f"Two-tower: {count_params(model) / 1e6:.2f}M params")
        summary["two_tower"] = fit(
            model, lambda b: model.loss(b, c.max_pairs_per_batch), store, c, c.history, device, False,
            lambda m: _val_ndcg(TwoTowerRec(m, c.history, device), val, store),
            paths.ckpt / "two_tower.pt", paths.logs / "two_tower.jsonl", cfg.seed,
        )

    save_json(summary, paths.reports / "baselines_val.json")
    log(f"baselines val ndcg@10: {summary}")
    return summary
