import numpy as np
import torch

from encore.data.store import Store, to_torch
from encore.data.windows import iterate, split_windows
from encore.utils import RunPaths, get_device, log, save_json
from encore.world.model import HEADS


def auc(y, p):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n1 = y.sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def _predict(model, store, starts, ends, H, device):
    model.eval()
    probs, labels = [], {k: [] for k in HEADS}
    for batch in iterate(store, starts, ends, H, 256):
        b = to_torch(batch, device)
        _, _, out, y, _ = model.loss(b, {k: 1.0 for k in HEADS}, label_split=1)
        probs.append(torch.sigmoid(out).cpu().numpy())
        for k in HEADS:
            labels[k].append(y[k].cpu().numpy())
    return np.concatenate(probs), {k: np.concatenate(v) for k, v in labels.items()}


def _metrics(p, y):
    return {
        "auc_complete": auc(y["complete"], p[:, 0]),
        "auc_like": auc(y["like"], p[:, 2]),
        "auc_dislike": auc(y["dislike"], p[:, 3]),
        "mae_ratio": float(np.abs(p[:, 1] - y["ratio"]).mean()),
        "ece_complete": ece(y["complete"], p[:, 0]),
        "brier_complete": float(((p[:, 0] - y["complete"]) ** 2).mean()),
    }


def _ood_analysis(cfg, store, ens, oracle, device, n_users=300, n_cand=10):
    es = store.eval_set("val").subset(n_users, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    logged = np.full((len(es), n_cand), -1, np.int64)
    for i, u in enumerate(es.uids):
        a = store.user_ptr[u] + store.n_upto[0][u]
        b = store.user_ptr[u] + store.n_upto[1][u]
        its = store.ev["item"][a:b][:n_cand]
        logged[i, : len(its)] = its
    random = rng.integers(store.n_items, size=(len(es), n_cand))
    out = {}
    t = lambda x: torch.as_tensor(x, device=device)
    for name, cand in (("logged", logged), ("random", random)):
        U = ens.member_utilities(t(es.hist_items), t(es.hist_fb), t(es.hist_org), t(cand)).float().cpu().numpy()
        valid = cand >= 0
        mean, std = U.mean(0)[valid], U.std(0)[valid]
        row = {"ensemble_std": float(std.mean()), "member0_minus_mean": float(np.abs(U[0][valid] - mean).mean())}
        if oracle is not None:
            true = oracle.utility(es.uids, es.hist_items, es.hist_complete, cand)[valid]
            row["spearman_mean_vs_true"] = spearman(mean, true)
            row["spearman_member0_vs_true"] = spearman(U[0][valid], true)
            row["spearman_std_vs_abs_error"] = spearman(std, np.abs(mean - true))
            row["mean_abs_error"] = float(np.abs(mean - true).mean())
        out[name] = row
    return out


def world_report(cfg):
    from encore.data.oracle import Oracle
    from encore.rl.reward import load_ensemble

    paths = RunPaths(cfg)
    store = Store(paths.data)
    device = get_device(cfg.device)
    ens = load_ensemble(cfg, paths, store, device)
    H = cfg.world.history
    vs, ve = split_windows(store, H, 1)
    preds = []
    report = {"members": {}}
    for i, m in enumerate(ens.members):
        p, y = _predict(m, store, vs, ve, H, device)
        preds.append(p)
        report["members"][str(i)] = _metrics(p, y)
    if ens.audit is not None:
        p_a, y = _predict(ens.audit, store, vs, ve, H, device)
        report["members"]["audit"] = _metrics(p_a, y)
    P = np.stack(preds)
    pm = P.mean(0)
    report["ensemble"] = _metrics(pm, y)
    report["n_val_events"] = int(len(pm))
    report["uncertainty"] = {
        "spearman_std_vs_abs_error_complete": spearman(P[:, :, 0].std(0), np.abs(pm[:, 0] - y["complete"])),
        "mean_std_complete": float(P[:, :, 0].std(0).mean()),
    }
    oracle = Oracle(cfg.paths.raw, store, cfg.reward.weights) if Oracle.available(cfg.paths.raw) else None
    report["ood"] = _ood_analysis(cfg, store, ens, oracle, device)
    save_json(report, paths.reports / "world_report.json")
    e = report["ensemble"]
    log(f"world ensemble on {len(pm):,} val events: AUC complete={e['auc_complete']:.3f} like={e['auc_like']:.3f} "
        f"dislike={e['auc_dislike']:.3f} ECE={e['ece_complete']:.3f}")
    for k, v in report["ood"].items():
        log(f"  {k} candidates: " + ", ".join(f"{a}={b:.3f}" for a, b in v.items()))
    return report
