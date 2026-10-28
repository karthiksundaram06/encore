import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encore.nn import CausalTransformer, init_weights


class SASRec(nn.Module):
    def __init__(self, n_items, d_model, n_layers, n_heads, max_len, dropout):
        super().__init__()
        self.hparams = dict(n_items=n_items, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                            max_len=max_len, dropout=dropout)
        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.fb_emb = nn.Embedding(5, d_model)
        self.body = CausalTransformer(d_model, n_layers, n_heads, max_len, dropout=dropout)
        init_weights(self)

    def encode(self, items, fb):
        valid = items >= 0
        x = self.item_emb(items + 1) + self.fb_emb(fb)
        h, _, _ = self.body(x, valid)
        return h

    def item_weights(self):
        return self.item_emb.weight[1:]

    def loss(self, b, num_negatives=0, n_warm=None):
        items, fb, complete = b["item"], b["fb"], b["complete"]
        h = self.encode(items[:, :-1], fb[:, :-1])
        tgt = items[:, 1:]
        m = (tgt >= 0) & (complete[:, 1:] > 0) & (items[:, :-1] >= 0)
        hs, ts = h[m], tgt[m]
        W = self.item_weights()
        if num_negatives == 0:
            return F.cross_entropy(hs @ W.T, ts)
        neg = torch.randint(n_warm or self.n_items, (num_negatives,), device=hs.device)
        pos = (hs * W[ts]).sum(-1, keepdim=True)
        negl = (hs @ W[neg].T).masked_fill(neg[None] == ts[:, None], float("-inf"))
        logits = torch.cat([pos, negl], 1)
        return F.cross_entropy(logits, torch.zeros(len(ts), dtype=torch.long, device=hs.device))

    @torch.no_grad()
    def scores(self, items, fb):
        h = self.encode(items, fb)[:, -1]
        return h @ self.item_weights().T


class TwoTower(nn.Module):
    def __init__(self, n_items, content, is_cold, popularity, d_model, id_dropout, temperature):
        super().__init__()
        self.hparams = dict(n_items=n_items, d_model=d_model, id_dropout=id_dropout,
                            temperature=temperature, content_dim=int(content.shape[1]))
        self.n_items = n_items
        self.id_dropout = id_dropout
        self.temperature = temperature
        dc = content.shape[1]
        self.hist_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.fb_emb = nn.Embedding(5, d_model)
        self.user_mlp = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.item_id = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.content_mlp = nn.Sequential(nn.Linear(dc, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.register_buffer("content", torch.as_tensor(content, dtype=torch.float32))
        self.register_buffer("is_cold", torch.as_tensor(is_cold, dtype=torch.bool))
        self.register_buffer("logq", torch.as_tensor(np.log(popularity), dtype=torch.float32))
        init_weights(self)

    def item_vecs(self, idx, train=False):
        keep = ~self.is_cold[idx]
        if train and self.id_dropout > 0:
            keep = keep & (torch.rand(idx.shape, device=idx.device) > self.id_dropout)
        v = self.item_id(idx + 1) * keep.unsqueeze(-1) + self.content_mlp(self.content[idx])
        return F.normalize(v, dim=-1)

    def user_prefix(self, items, fb, recent=5):
        valid = (items >= 0).unsqueeze(-1).float()
        x = (self.hist_emb(items + 1) + self.fb_emb(fb)) * valid
        cs = x.cumsum(1)
        cnt = valid.cumsum(1)
        mean_all = cs / cnt.clamp(min=1)
        shifted = F.pad(cs, (0, 0, recent, 0))[:, : cs.size(1)]
        cshift = F.pad(cnt, (0, 0, recent, 0))[:, : cnt.size(1)]
        mean_recent = (cs - shifted) / (cnt - cshift).clamp(min=1)
        return F.normalize(self.user_mlp(torch.cat([mean_all, mean_recent], -1)), dim=-1)

    def loss(self, b, max_pairs):
        items, fb, complete = b["item"], b["fb"], b["complete"]
        u = self.user_prefix(items[:, :-1], fb[:, :-1])
        tgt = items[:, 1:]
        m = (tgt >= 0) & (complete[:, 1:] > 0) & (items[:, :-1] >= 0)
        us, ts = u[m], tgt[m]
        if len(ts) > max_pairs:
            sel = torch.randperm(len(ts), device=ts.device)[:max_pairs]
            us, ts = us[sel], ts[sel]
        v = self.item_vecs(ts, train=True)
        logits = us @ v.T / self.temperature - self.logq[ts][None]
        same = (ts[:, None] == ts[None]) & ~torch.eye(len(ts), dtype=torch.bool, device=ts.device)
        logits = logits.masked_fill(same, float("-inf"))
        return F.cross_entropy(logits, torch.arange(len(ts), device=ts.device))

    @torch.no_grad()
    def all_item_vecs(self, chunk=65536):
        out = []
        for i in range(0, self.n_items, chunk):
            idx = torch.arange(i, min(i + chunk, self.n_items), device=self.content.device)
            out.append(self.item_vecs(idx))
        return torch.cat(out)

    @torch.no_grad()
    def scores(self, items, fb, item_vecs=None):
        u = self.user_prefix(items, fb)[:, -1]
        if item_vecs is None:
            item_vecs = self.all_item_vecs()
        return u @ item_vecs.T


def popularity_ranking(store, window_days):
    ev = store.ev
    unit = store.meta["time_unit_seconds"]
    val_start = store.meta["time"]["val_start"]
    t0 = val_start - window_days * 86400 / unit
    m = (ev["split"] == 0) & (ev["complete"] > 0) & (ev["ts"] >= t0)
    counts = np.bincount(ev["item"][m], minlength=store.n_items).astype(np.float64)
    counts += 1e-6 * store.train_positives
    return np.argsort(-counts)
