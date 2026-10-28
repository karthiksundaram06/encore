import torch
import torch.nn as nn
import torch.nn.functional as F

from encore.nn import CausalTransformer, init_weights

HEADS = ("complete", "ratio", "like", "dislike")


class ListenerModel(nn.Module):
    def __init__(self, n_items, content, is_cold, d_model, n_layers, n_heads, max_len, dropout=0.1, id_dropout=0.1,
                 prior_scale=0.0):
        super().__init__()
        content = torch.as_tensor(content, dtype=torch.float32)
        self.hparams = dict(n_items=n_items, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                            max_len=max_len, dropout=dropout, id_dropout=id_dropout, content_dim=content.shape[1],
                            prior_scale=prior_scale)
        self.id_dropout = id_dropout
        self.prior_scale = prior_scale
        self.item_id = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.content_proj = nn.Linear(content.shape[1], d_model)
        self.fb_emb = nn.Embedding(5, d_model)
        self.org_emb = nn.Embedding(2, d_model)
        self.body = CausalTransformer(d_model, n_layers, n_heads, max_len, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, len(HEADS))
        )
        self.register_buffer("content", content, persistent=False)
        self.register_buffer("is_cold", torch.as_tensor(is_cold, dtype=torch.bool), persistent=False)
        init_weights(self)
        if prior_scale > 0:
            self.item_bias = nn.Embedding(n_items + 1, len(HEADS))
            nn.init.zeros_(self.item_bias.weight)
            self.register_buffer("item_prior", torch.randn(n_items + 1, len(HEADS)) * prior_scale)
            self.prior_net = nn.Sequential(nn.Linear(4 * d_model, 2 * d_model), nn.Tanh(), nn.Linear(2 * d_model, len(HEADS)))
            for p in self.prior_net.parameters():
                nn.init.normal_(p, std=1.0 / max(1, p.shape[-1]) ** 0.5)
                p.requires_grad_(False)

    def item_repr(self, items):
        idx = items.clamp(min=0)
        keep = ~self.is_cold[idx]
        if self.training and self.id_dropout > 0:
            keep = keep & (torch.rand(idx.shape, device=idx.device) > self.id_dropout)
        return self.item_id(items + 1) * keep.unsqueeze(-1) + self.content_proj(self.content[idx])

    def encode(self, items, fb, org):
        valid = items >= 0
        x = self.item_repr(items) + self.fb_emb(fb) + self.org_emb(org)
        h, _, _ = self.body(x, valid)
        return h

    def predict(self, h, cand, cand_org):
        c = self.item_repr(cand)
        z = torch.cat([h, c, h * c, self.org_emb(cand_org)], -1)
        out = self.head(z)
        if self.prior_scale > 0:
            idx = cand + 1
            out = out + self.item_bias(idx) + self.item_prior[idx] + self.prior_scale * self.prior_net(z.detach())
        return out

    def loss(self, b, weights, label_split=None):
        items, fb, org = b["item"], b["fb"], b["organic"]
        h = self.encode(items[:, :-1], fb[:, :-1], org[:, :-1])
        cand = items[:, 1:]
        m = (cand >= 0) & (items[:, :-1] >= 0)
        if label_split is not None:
            m = m & (b["split"][:, 1:] == label_split)
        out = self.predict(h[m], cand[m], org[:, 1:][m]).float()
        y = {
            "complete": b["complete"][:, 1:][m].float(),
            "ratio": b["ratio"][:, 1:][m].float().clamp(0, 1),
            "like": b["like"][:, 1:][m].float(),
            "dislike": b["dislike"][:, 1:][m].float(),
        }
        parts = {k: F.binary_cross_entropy_with_logits(out[:, i], y[k]) for i, k in enumerate(HEADS)}
        total = sum(weights[k] * v for k, v in parts.items())
        return total, parts, out, y, m.sum()


def utility(logits, weights):
    p = torch.sigmoid(logits.float())
    return (
        weights["complete"] * p[..., 0]
        + weights["ratio"] * p[..., 1]
        + weights["like"] * p[..., 2]
        - weights["dislike"] * p[..., 3]
    )


def build_listener(cfg, store, content):
    w = cfg.world
    return ListenerModel(store.n_items, content, store.is_cold, w.d_model, w.n_layers, w.n_heads,
                         w.history + 1, w.dropout, w.id_dropout, w.get("prior_scale", 0.0))


def load_listener(path, content, is_cold, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp = dict(ck["hparams"])
    hp.pop("content_dim")
    model = ListenerModel(content=content, is_cold=is_cold, **hp).to(device)
    state = {k: v for k, v in ck["state_dict"].items() if k not in ("content", "is_cold")}
    model.load_state_dict(state)
    model.eval()
    return model
