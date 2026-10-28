import torch
import torch.nn as nn

from encore.nn import CausalTransformer, init_weights
from encore.sid.trie import BOS, PAD


class SIDGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len, dropout=0.1, ff_mult=4, n_feedback=5):
        super().__init__()
        self.hparams = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                            max_len=max_len, dropout=dropout, ff_mult=ff_mult, n_feedback=n_feedback)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.fb_emb = nn.Embedding(n_feedback, d_model)
        self.body = CausalTransformer(d_model, n_layers, n_heads, max_len, ff_mult, dropout)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        init_weights(self)
        self.head.weight = self.tok_emb.weight

    def forward(self, tokens, fb, cache=None, past_valid=None):
        valid = tokens != PAD
        x = self.tok_emb(tokens) + self.fb_emb(fb)
        h, cache, kv_valid = self.body(x, valid, cache, past_valid)
        return self.head(h), cache, kv_valid


def max_len_for(history, n_levels):
    return 1 + (history + 1) * n_levels + 1


def history_tokens(trie, items, fb, H):
    items = items[:, -H:]
    fb = fb[:, -H:]
    B, Hh = items.shape
    L1 = trie.n_levels
    valid = items >= 0
    codes = trie.tokens[items.clamp(min=0)].masked_fill(~valid.unsqueeze(-1), PAD)
    tok = torch.full((B, 1 + Hh * L1), PAD, dtype=torch.long, device=items.device)
    tok[:, 1:] = codes.reshape(B, -1)
    bos_pos = (Hh - valid.sum(1)) * L1
    tok[torch.arange(B, device=items.device), bos_pos] = BOS
    fbt = torch.zeros_like(tok)
    fbt[:, L1::L1] = fb * valid
    return tok, fbt


def sft_batch(trie, b, H):
    items, fb, complete = b["item"][:, -H:], b["fb"][:, -H:], b["complete"][:, -H:]
    tok, fbt = history_tokens(trie, items, fb, H)
    target_mask = ((items >= 0) & (complete > 0)).repeat_interleave(trie.n_levels, dim=1)
    return tok[:, :-1], fbt[:, :-1], tok[:, 1:], target_mask


def constrained_item_nll(logits, trie, items, item_mask, NEG=-1e9):
    B, H = items.shape
    L1 = trie.n_levels
    lg_all = logits[:, : H * L1].reshape(B, H, L1, -1)
    sel = item_mask & (items >= 0)
    it = items[sel]
    lg_all = lg_all[sel]
    codes = trie.codes_t[it]
    node = torch.zeros(len(it), dtype=torch.long, device=items.device)
    nll = torch.zeros(len(it), device=items.device)
    per_level = []
    for l in range(L1):
        off = trie.offsets[l]
        lg = lg_all[:, l, off:off + trie.sizes[l]].float()
        child = trie.child[l][node].long()
        lp = lg.masked_fill(child < 0, NEG).log_softmax(-1)
        c = codes[:, l]
        step = -lp.gather(1, c[:, None]).squeeze(1)
        per_level.append(step.mean().detach())
        nll = nll + step
        node = child.gather(1, c[:, None]).squeeze(1)
    return nll.mean() / L1, per_level


def build_generator(cfg, trie):
    g = cfg.generator
    return SIDGenerator(
        trie.vocab_size, g.d_model, g.n_layers, g.n_heads, max_len_for(g.history, trie.n_levels),
        g.dropout, g.ff_mult,
    )


def load_generator(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SIDGenerator(**ck["hparams"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
