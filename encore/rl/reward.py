import numpy as np
import torch

from encore.utils import autocast
from encore.world.model import load_listener, utility


class WorldEnsemble:
    def __init__(self, members, weights, H, device, audit=None, amp=True):
        self.members = members
        self.audit = audit
        self.weights = dict(weights)
        self.H = H
        self.device = device
        self.amp = amp

    @torch.no_grad()
    def _utilities(self, models, items, fb, org, cand):
        items, fb, org = items[:, -self.H:], fb[:, -self.H:], org[:, -self.H:]
        B, C = cand.shape
        c = cand.clamp(min=0)
        zero = torch.zeros_like(c)
        out = []
        for m in models:
            with autocast(self.device, self.amp):
                h = m.encode(items, fb, org)[:, -1]
                logits = m.predict(h[:, None].expand(-1, C, -1), c, zero)
            out.append(utility(logits, self.weights))
        return torch.stack(out)

    def member_utilities(self, items, fb, org, cand):
        return self._utilities(self.members, items, fb, org, cand)

    def audit_utilities(self, items, fb, org, cand):
        if self.audit is None:
            return None
        return self._utilities([self.audit], items, fb, org, cand)[0]

    @staticmethod
    def combine(U, mode, pessimism):
        if mode == "single":
            return U[0]
        return U.mean(0) - pessimism * U.std(0, unbiased=False)

    @torch.no_grad()
    def eval_utility(self, eval_set, recs, batch=256, which="mean"):
        vals = []
        for i in range(0, len(eval_set), batch):
            sl = slice(i, i + batch)
            items = torch.as_tensor(eval_set.hist_items[sl], device=self.device)
            fb = torch.as_tensor(eval_set.hist_fb[sl], device=self.device)
            org = torch.as_tensor(eval_set.hist_org[sl], device=self.device)
            cand = torch.as_tensor(recs[sl], device=self.device)
            if which == "audit":
                u = self.audit_utilities(items, fb, org, cand)
            else:
                U = self.member_utilities(items, fb, org, cand)
                u = {"mean": U.mean(0), "single": U[0], "pessimistic": U.mean(0) - U.std(0, unbiased=False),
                     "std": U.std(0, unbiased=False)}[which]
            u = u.float().masked_fill(cand < 0, float("nan"))
            vals.append(u.cpu().numpy())
        return float(np.nanmean(np.concatenate(vals)))


def load_ensemble(cfg, paths, store, device):
    content = torch.as_tensor(np.load(paths.sid / "sid.npz")["fused"], dtype=torch.float32, device=device)
    is_cold = torch.as_tensor(store.is_cold, dtype=torch.bool, device=device)
    members = []
    for m in range(cfg.world.members):
        p = paths.ckpt / f"world_{m}.pt"
        if p.exists():
            members.append(load_listener(p, content, is_cold, device))
    if not members:
        raise FileNotFoundError("no world models found; run `python -m encore world` first")
    audit = None
    if (paths.ckpt / "world_audit.pt").exists():
        audit = load_listener(paths.ckpt / "world_audit.pt", content, is_cold, device)
    ens = WorldEnsemble(members, cfg.reward.weights, cfg.world.history, device, audit, cfg.amp)
    ens.content = content
    return ens


def diversity_bonus(item_emb, items):
    e = item_emb[items.clamp(min=0)]
    sim = e @ e.transpose(1, 2)
    G = items.size(1)
    mean_other = (sim.sum(-1) - sim.diagonal(dim1=1, dim2=2)) / max(G - 1, 1)
    return 1.0 - mean_other
