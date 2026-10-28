from dataclasses import dataclass, field

import torch

from encore.nn import index_cache, repeat_cache

NEG = -1e9


@dataclass
class Steering:
    exclude_items: torch.Tensor | None = None
    first_code_bias: torch.Tensor | None = None
    temperature: float = 1.0
    extra: dict = field(default_factory=dict)


def _level_logits(logits, trie, level):
    off = trie.offsets[level]
    return logits[..., off:off + trie.sizes[level]].float()


def _excluded(cand, exclude_rows):
    ex = exclude_rows.sort(dim=1).values
    pos = torch.searchsorted(ex, cand.contiguous()).clamp(max=ex.size(1) - 1)
    return ex.gather(1, pos) == cand


@torch.no_grad()
def beam_search(model, trie, tokens, fb, beam, steering=None):
    model.eval()
    steering = steering or Steering()
    B = tokens.size(0)
    dev = tokens.device
    logits, cache, kv_valid = model(tokens, fb)
    logits = logits[:, -1]
    scores = torch.zeros(B, 1, device=dev)
    nodes = torch.zeros(B, 1, dtype=torch.long, device=dev)
    cur = 1
    for l in range(trie.n_levels):
        K = trie.sizes[l]
        lg = _level_logits(logits, trie, l) / steering.temperature
        child = trie.child[l][nodes.reshape(-1)].long()
        allowed = child >= 0
        if l == 0 and steering.first_code_bias is not None:
            bias = steering.first_code_bias
            bias = bias.repeat_interleave(cur, 0) if bias.dim() == 2 else bias[None]
            lg = lg + bias
        if l == trie.n_levels - 1 and steering.exclude_items is not None:
            allowed &= ~_excluded(child, steering.exclude_items.repeat_interleave(cur, 0))
        lp = lg.masked_fill(~allowed, NEG).log_softmax(-1).masked_fill(~allowed, float("-inf"))
        cand = (scores.reshape(-1, 1) + lp).reshape(B, cur * K)
        width = min(beam, cur * K)
        top_s, top_i = cand.topk(width, dim=1)
        origin = top_i // K
        code = top_i % K
        flat = (torch.arange(B, device=dev)[:, None] * cur + origin).reshape(-1)
        nodes = child[flat, code.reshape(-1)].reshape(B, width)
        scores = top_s
        if l < trie.n_levels - 1:
            nodes = nodes.clamp(min=0)
            cache = index_cache(cache, flat)
            kv_valid = kv_valid.index_select(0, flat)
            step = (code + trie.offsets[l]).reshape(-1, 1)
            logits, cache, kv_valid = model(step, torch.zeros_like(step), cache=cache, past_valid=kv_valid)
            logits = logits[:, -1]
        cur = width
    items = torch.where(torch.isfinite(scores), nodes, torch.full_like(nodes, -1))
    return items, scores


def beam_rows_cap(model, n_levels, history, budget_gb, half):
    hp = model.hparams
    ctx = 2 + (history + 1) * n_levels
    per_row = ctx * hp["n_layers"] * 2 * hp["d_model"] * (2 if half else 4)
    return max(1, int(budget_gb * 1e9 / (2 * per_row)))


def _slice_steering(st, sl, B):
    if st is None:
        return None
    ex, bias = st.exclude_items, st.first_code_bias
    if ex is not None and ex.dim() == 2 and ex.size(0) == B:
        ex = ex[sl]
    if bias is not None and bias.dim() == 2 and bias.size(0) == B:
        bias = bias[sl]
    return Steering(ex, bias, st.temperature, st.extra)


@torch.no_grad()
def beam_search_chunked(model, trie, tokens, fb, beam, steering=None, max_rows=None):
    B = tokens.size(0)
    if not max_rows or B * beam <= max_rows:
        return beam_search(model, trie, tokens, fb, beam, steering)
    step = max(1, max_rows // beam)
    outs, scores = [], []
    for i in range(0, B, step):
        sl = slice(i, i + step)
        o, s = beam_search(model, trie, tokens[sl], fb[sl], beam, _slice_steering(steering, sl, B))
        outs.append(o)
        scores.append(s)
    return torch.cat(outs), torch.cat(scores)


@torch.no_grad()
def sample(model, trie, tokens, fb, n_samples, temperature=1.0):
    model.eval()
    B = tokens.size(0)
    dev = tokens.device
    logits, cache, kv_valid = model(tokens, fb)
    logits = logits[:, -1].repeat_interleave(n_samples, 0)
    cache = repeat_cache(cache, n_samples)
    kv_valid = kv_valid.repeat_interleave(n_samples, 0)
    nodes = torch.zeros(B * n_samples, dtype=torch.long, device=dev)
    codes = []
    for l in range(trie.n_levels):
        lg = _level_logits(logits, trie, l) / temperature
        child = trie.child[l][nodes].long()
        lg = lg.masked_fill(child < 0, float("-inf"))
        c = torch.multinomial(lg.softmax(-1), 1).squeeze(1)
        nodes = child.gather(1, c[:, None]).squeeze(1)
        codes.append(c)
        if l < trie.n_levels - 1:
            step = (c + trie.offsets[l])[:, None]
            logits, cache, kv_valid = model(step, torch.zeros_like(step), cache=cache, past_valid=kv_valid)
            logits = logits[:, -1]
    return torch.stack(codes, 1), nodes


def code_paths(trie, codes):
    nodes = [torch.zeros(codes.size(0), dtype=torch.long, device=codes.device)]
    for l in range(trie.n_levels - 1):
        nodes.append(trie.child[l][nodes[-1], codes[:, l]].long())
    return nodes


def sequence_logprobs(model, trie, tokens, fb, codes, n_samples):
    logits, cache, kv_valid = model(tokens, fb)
    first = logits[:, -1].repeat_interleave(n_samples, 0)
    cache = repeat_cache(cache, n_samples)
    kv_valid = kv_valid.repeat_interleave(n_samples, 0)
    offs = torch.as_tensor(trie.offsets[:-1], device=codes.device)
    steps = codes[:, :-1] + offs[None]
    step_logits, _, _ = model(steps, torch.zeros_like(steps), cache=cache, past_valid=kv_valid)
    all_logits = torch.cat([first.unsqueeze(1), step_logits], 1)
    nodes = code_paths(trie, codes)
    out = []
    for l in range(trie.n_levels):
        lg = _level_logits(all_logits[:, l], trie, l)
        allowed = trie.child[l][nodes[l]] >= 0
        lp = lg.masked_fill(~allowed, NEG).log_softmax(-1)
        out.append(lp.gather(1, codes[:, l:l + 1]).squeeze(1))
    return torch.stack(out, 1)
