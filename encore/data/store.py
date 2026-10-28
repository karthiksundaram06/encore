from pathlib import Path

import numpy as np
import torch

from encore.utils import load_json

EVENT_KEYS = ("uid", "item", "ts", "ratio", "complete", "like", "dislike", "organic", "split", "fb")


class EvalSet:
    def __init__(self, path):
        z = np.load(path)
        self.uids = z["uids"]
        self.hist_items = z["hist_items"]
        self.hist_fb = z["hist_fb"]
        self.hist_org = z["hist_org"]
        self.hist_complete = z["hist_complete"]
        self.tgt_ptr = z["tgt_ptr"]
        self.tgt_items = z["tgt_items"]

    def __len__(self):
        return len(self.uids)

    def targets(self, i):
        return self.tgt_items[self.tgt_ptr[i]:self.tgt_ptr[i + 1]]

    def subset(self, n, seed=0):
        if n is None or n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=n, replace=False))
        out = object.__new__(EvalSet)
        out.uids = self.uids[idx]
        out.hist_items = self.hist_items[idx]
        out.hist_fb = self.hist_fb[idx]
        out.hist_org = self.hist_org[idx]
        out.hist_complete = self.hist_complete[idx]
        tg = [self.targets(i) for i in idx]
        out.tgt_ptr = np.concatenate([[0], np.cumsum([len(t) for t in tg])]).astype(np.int64)
        out.tgt_items = np.concatenate(tg) if tg else np.zeros(0, np.int64)
        return out

    def batch(self, idx, H):
        return {
            "items": self.hist_items[idx, -H:],
            "fb": self.hist_fb[idx, -H:],
            "org": self.hist_org[idx, -H:],
            "complete": self.hist_complete[idx, -H:],
            "uids": self.uids[idx],
        }


class Store:
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.dir = data_dir
        z = np.load(data_dir / "events.npz")
        self.user_ptr = z["user_ptr"]
        self.user_raw = z["user_raw"]
        self.ev = {k: z[k] for k in EVENT_KEYS}
        c = np.load(data_dir / "catalog.npz")
        self.raw_item_id = c["raw_item_id"]
        self.is_cold = c["is_cold"]
        self.audio = c["audio"]
        self.has_audio = c["has_audio"]
        self.train_positives = c["train_positives"]
        self.meta = load_json(data_dir / "meta.json")
        self.n_users = len(self.user_ptr) - 1
        self.n_items = len(self.raw_item_id)
        self.n_warm = int((~self.is_cold).sum())
        sp = self.ev["split"]
        uid = self.ev["uid"]
        self.n_upto = {s: np.bincount(uid[sp <= s], minlength=self.n_users).astype(np.int64) for s in (0, 1, 2)}

    def eval_set(self, name):
        return EvalSet(self.dir / f"eval_{name}.npz")

    def user_range(self, u, max_split=0):
        a = self.user_ptr[u]
        return a, a + self.n_upto[max_split][u]

    def popularity(self):
        p = self.train_positives.astype(np.float64) + 1.0
        return p / p.sum()


def to_torch(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items() if isinstance(v, np.ndarray)}
