from pathlib import Path

import numpy as np
import torch

from encore.eval.metrics import beyond_accuracy, discovery_metrics, ranking_metrics, repeat_share
from encore.utils import RunPaths, Timer, get_device, load_json, log, save_json

SPLIT_ID = {"val": 1, "test": 2}


def history_sets(store, eval_set, split_id):
    out = []
    for u in eval_set.uids:
        a = store.user_ptr[u]
        out.append(np.unique(store.ev["item"][a:a + store.n_upto[split_id - 1][u]]).astype(np.int64))
    return out


def collect(rec, eval_set, k, batch_size, exclude=None):
    out = []
    for i in range(0, len(eval_set), batch_size):
        idx = np.arange(i, min(i + batch_size, len(eval_set)))
        batch = {"items": eval_set.hist_items[idx], "fb": eval_set.hist_fb[idx],
                 "complete": eval_set.hist_complete[idx]}
        if exclude is None:
            out.append(rec.recommend(batch, k))
        else:
            out.append(rec.recommend(batch, k, exclude=[exclude[j] for j in idx]))
    return np.concatenate(out)


def evaluate(rec, eval_set, store, ks, batch_size=64, embeddings=None, oracle=None, utility_fns=None,
             hist=None):
    kmax = max(ks)
    with Timer() as t:
        recs = collect(rec, eval_set, kmax, batch_size)
    res = ranking_metrics(recs, eval_set, ks, store.is_cold)
    res.update(beyond_accuracy(recs, store.n_items, store.popularity(), embeddings, store.is_cold, k=min(ks)))
    k10 = min(ks)
    if hist is not None:
        res[f"repeat_share@{k10}"] = repeat_share(recs, hist, k10)
        if getattr(rec, "discovery", True):
            new_recs = collect(rec, eval_set, kmax, batch_size, exclude=hist)
            res.update(discovery_metrics(new_recs, eval_set, hist, ks, store.is_cold))
    if oracle is not None:
        util = oracle.utility(eval_set.uids, eval_set.hist_items, eval_set.hist_complete, recs[:, :k10])
        res[f"true_utility@{k10}"] = float(np.nanmean(util))
    for name, fn in (utility_fns or {}).items():
        res[f"{name}@{k10}"] = float(fn(eval_set, recs[:, :k10]))
    res["users"] = len(eval_set)
    res["seconds"] = t.seconds
    return res, recs


def load_recommenders(cfg, paths, store, device):
    from encore.baselines.models import SASRec, TwoTower
    from encore.eval.recommenders import GeneratorRec, HistoryRec, PopularityRec, SASRecRec, TwoTowerRec
    from encore.gen.model import load_generator
    from encore.sid.trie import SIDTrie

    recs = [HistoryRec()]
    pop = paths.ckpt / "popularity.npy"
    if pop.exists():
        recs.append(PopularityRec(np.load(pop)))
    if (paths.ckpt / "sasrec.pt").exists():
        ck = torch.load(paths.ckpt / "sasrec.pt", map_location=device, weights_only=False)
        m = SASRec(**ck["hparams"]).to(device)
        m.load_state_dict(ck["state_dict"])
        recs.append(SASRecRec(m, store.is_cold, cfg.sasrec.history, device))
    if (paths.ckpt / "two_tower.pt").exists():
        from encore.baselines.train import two_tower_content

        ck = torch.load(paths.ckpt / "two_tower.pt", map_location=device, weights_only=False)
        m = TwoTower(ck["hparams"]["n_items"], two_tower_content(store), store.is_cold, store.popularity(),
                     ck["hparams"]["d_model"], ck["hparams"]["id_dropout"], ck["hparams"]["temperature"]).to(device)
        m.load_state_dict(ck["state_dict"])
        recs.append(TwoTowerRec(m, cfg.two_tower.history, device))
    if (paths.sid / "sid.npz").exists():
        trie = SIDTrie.load(paths.sid).to(device)
        modes = ("hit", "single", "ensemble", "hybrid", "oracle")
        names = ["sft"] + [f"grpo_{m}{s}" for m in modes for s in ("_best", "")]
        for name in names:
            p = paths.ckpt / f"{name}.pt"
            if p.exists():
                model, _ = load_generator(p, device)
                recs.append(GeneratorRec(model, trie, cfg.generator.history, cfg.eval.beam, device,
                                         name=f"gen_{name}", amp=cfg.amp,
                                         cache_gb=cfg.eval.get("beam_cache_gb", 3.0)))
    return recs


def _markdown(results, ks):
    k = min(ks)
    cols = [f"ndcg@{k}", f"recall@{k}", f"repeat_share@{k}", f"new_ndcg@{k}", f"new_recall@{k}",
            f"new_cold_recall@{max(ks)}", f"coverage@{k}", f"ild@{k}", f"novelty@{k}"]
    cols = [c for c in cols if any(c in r for r in results.values())]
    extra = [c for c in (f"true_utility@{k}", f"wm_utility@{k}") if any(c in r for r in results.values())]
    cols += extra
    lines = ["| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        vals = []
        for c in cols:
            v = r.get(c, float("nan"))
            vals.append("-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_eval(cfg, split="test"):
    from encore.data.oracle import Oracle
    from encore.data.store import Store

    paths = RunPaths(cfg)
    device = get_device(cfg.device)
    store = Store(paths.data)
    es = store.eval_set(split).subset(cfg.eval.max_users, cfg.seed)
    hist = history_sets(store, es, SPLIT_ID[split])
    emb = np.load(paths.sid / "sid.npz")["fused"] if (paths.sid / "sid.npz").exists() else None
    oracle = Oracle(cfg.paths.raw, store, cfg.reward.weights) if Oracle.available(cfg.paths.raw) else None
    utility_fns = {}
    if (paths.ckpt / "world_0.pt").exists():
        from encore.rl.reward import load_ensemble

        ens = load_ensemble(cfg, paths, store, device)
        utility_fns["wm_utility"] = lambda e, r: ens.eval_utility(e, r)
    results = {}
    for rec in load_recommenders(cfg, paths, store, device):
        log(f"evaluating {rec.name} on {split} ({len(es)} users)")
        res, _ = evaluate(rec, es, store, cfg.eval.ks, cfg.eval.batch_size, emb, oracle, utility_fns, hist)
        results[rec.name] = res
        k = min(cfg.eval.ks)
        log(f"  {rec.name}: ndcg@{k}={res[f'ndcg@{k}']:.4f} repeat_share@{k}={res[f'repeat_share@{k}']:.2f} "
            f"new_ndcg@{k}={res.get(f'new_ndcg@{k}', float('nan')):.4f} "
            f"new_cold_recall@{max(cfg.eval.ks)}={res.get(f'new_cold_recall@{max(cfg.eval.ks)}', float('nan')):.4f}")
    save_json(results, paths.reports / f"results_{split}.json")
    md = _markdown(results, cfg.eval.ks)
    (paths.reports / f"results_{split}.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return results


def check(cfg):
    paths = RunPaths(cfg)
    res = load_json(paths.reports / "results_test.json")
    k = min(cfg.eval.ks)
    key = f"ndcg@{k}"
    pop = res["popularity"][key]
    gen = res["gen_sft"][key]
    best = max(res[m][key] for m in ("sasrec", "two_tower") if m in res)
    log(f"check: popularity {key}={pop:.4f}, best baseline {key}={best:.4f}, gen_sft {key}={gen:.4f}")
    assert best > pop, "a learned baseline should beat popularity"
    assert gen > 0.9 * pop, "generator fell far below popularity"
    assert all(k.startswith("gen_grpo_") or k in ("history_repeat", "popularity", "sasrec", "two_tower", "gen_sft")
               for k in res)
    assert Path(paths.reports / "collapse.png").exists(), "collapse chart missing"
    log("check passed")
