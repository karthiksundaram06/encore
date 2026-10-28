import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_mask(valid_keys, n_query):
    B, S = valid_keys.shape
    past = S - n_query
    qi = torch.arange(n_query, device=valid_keys.device)[:, None] + past
    kj = torch.arange(S, device=valid_keys.device)[None, :]
    causal = kj <= qi
    mask = causal[None] & valid_keys[:, None, :]
    mask = mask | (kj == qi)[None]
    return mask[:, None]


class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, mask, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out), (k, v)


class Block(nn.Module):
    def __init__(self, d, n_heads, ff_mult, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = Attention(d, n_heads)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, cache=None):
        a, kv = self.attn(self.norm1(x), mask, cache)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, kv


class CausalTransformer(nn.Module):
    def __init__(self, d, n_layers, n_heads, max_len, ff_mult=4, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ff_mult, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid, cache=None, past_valid=None):
        if cache is None:
            pos = (valid.long().cumsum(1) - 1).clamp(min=0)
            keys_valid = valid
        else:
            n_past = past_valid.long().sum(1, keepdim=True)
            pos = n_past + (valid.long().cumsum(1) - 1).clamp(min=0)
            keys_valid = torch.cat([past_valid, valid], dim=1)
        pos = pos.clamp(max=self.max_len - 1)
        h = self.drop(x + self.pos_emb(pos))
        mask = attention_mask(keys_valid, x.size(1))
        new_cache = []
        for i, blk in enumerate(self.blocks):
            h, kv = blk(h, mask, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.norm(h), new_cache, keys_valid


def init_weights(module, std=0.02):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def index_cache(cache, idx):
    return [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in cache]


def repeat_cache(cache, n):
    return [(k.repeat_interleave(n, 0), v.repeat_interleave(n, 0)) for k, v in cache]
