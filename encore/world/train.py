import numpy as np
import torch

from encore.data.store import Store, to_torch
from encore.data.windows import iterate, split_windows
from encore.train_loop import fit
from encore.utils import RunPaths, count_params, get_device, log, save_json, seed_all
from encore.world.model import build_listener


def listener_content(paths):
    return np.load(paths.sid / "sid.npz")["fused"]


@torch.no_grad()
def split_loss(model, store, starts, ends, H, device, weights, split_id, batch_size=256):
    model.eval()
    tot, n = 0.0, 0
    for batch in iterate(store, starts, ends, H, batch_size):
        b = to_torch(batch, device)
        loss, _, _, _, cnt = model.loss(b, weights, label_split=split_id)
        if cnt > 0:
            tot += loss.item() * cnt.item()
            n += cnt.item()
    return tot / max(n, 1)


def train_world(cfg):
    paths = RunPaths(cfg)
    store = Store(paths.data)
    device = get_device(cfg.device)
    w = cfg.world
    content = listener_content(paths)
    vs, ve = split_windows(store, w.history, 1)
    weights = dict(w.loss_weights)
    summary = []
    names = [str(m) for m in range(w.members)] + (["audit"] if w.get("audit", True) else [])
    for m, name in enumerate(names):
        seed_all(cfg.seed + 100 * (m + 1))
        rng = np.random.default_rng(cfg.seed + 100 * (m + 1))
        users = rng.integers(store.n_users, size=store.n_users) if w.bootstrap else None
        model = build_listener(cfg, store, content).to(device)
        if m == 0:
            audit = " + held-out audit" if "audit" in names else ""
            log(f"listener model: {count_params(model) / 1e6:.2f}M params x {len(names)} ({w.members} ensemble{audit})")
        best = fit(
            model, lambda b: model.loss(b, weights)[0], store, w, w.history, device, cfg.amp,
            lambda mm: split_loss(mm, store, vs, ve, w.history, device, weights, 1),
            paths.ckpt / f"world_{name}.pt", paths.logs / f"world_{name}.jsonl", cfg.seed + m,
            users=users, maximize=False,
        )
        summary.append({"member": name, "val_loss": best})
        log(f"world member {name}: best val loss {best:.4f}")
    save_json(summary, paths.reports / "world_train.json")
    return summary
