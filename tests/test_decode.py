import itertools

import torch

from encore.gen.decode import Steering, beam_search, sample, sequence_logprobs
from encore.gen.model import SIDGenerator, history_tokens, max_len_for
from encore.nn import CausalTransformer


def _model(trie, H=6):
    return SIDGenerator(trie.vocab_size, 32, 2, 2, max_len_for(H, trie.n_levels), dropout=0.0).eval()


def _context(trie, B=3, H=6):
    items = torch.randint(0, len(trie.codes), (B, H))
    items[0, :3] = -1
    fb = torch.randint(1, 5, (B, H))
    return history_tokens(trie, items, fb, H)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    body = CausalTransformer(16, 2, 2, 64, dropout=0.0).eval()
    x = torch.randn(2, 10, 16)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[0, :4] = False
    full, _, _ = body(x, valid)
    h, cache, kv = body(x[:, :7], valid[:, :7])
    step, _, _ = body(x[:, 7:], valid[:, 7:], cache=cache, past_valid=kv)
    assert torch.allclose(full[:, 7:], step, atol=1e-5)


def test_history_tokens_layout(small_trie):
    t = small_trie
    items = torch.tensor([[-1, -1, 3, 5]])
    fb = torch.tensor([[0, 0, 2, 4]])
    tok, fbt = history_tokens(t, items, fb, 4)
    L = t.n_levels
    assert tok[0, 2 * L].item() == 1
    assert tok[0, 2 * L + 1: 3 * L + 1].tolist() == t.tokens[3].tolist()
    assert fbt[0, 3 * L].item() == 2 and fbt[0, 4 * L].item() == 4
    assert fbt[0].sum().item() == 6


def test_beam_search_matches_exhaustive(small_trie):
    t = small_trie
    model = _model(t)
    tok, fbt = _context(t)
    n = len(t.codes)
    items, scores = beam_search(model, t, tok, fbt, beam=n * 4)
    codes = t.codes_t
    for b in range(tok.size(0)):
        lp = sequence_logprobs(model, t, tok[b:b + 1], fbt[b:b + 1], codes, n).sum(1)
        exact = lp.argsort(descending=True)[:10]
        assert items[b, :10].tolist() == exact.tolist()
        assert torch.allclose(scores[b, :10], lp[exact], atol=1e-4)


def test_beam_outputs_valid_unique(small_trie):
    t = small_trie
    items, _ = beam_search(_model(t), t, *_context(t), beam=8)
    for row in items.tolist():
        assert len(set(row)) == len(row) and all(0 <= i < len(t.codes) for i in row)


def test_exclusion_and_bias(small_trie):
    t = small_trie
    model = _model(t)
    tok, fbt = _context(t)
    first, _ = beam_search(model, t, tok, fbt, beam=5)
    excl = first[:, :2]
    items, _ = beam_search(model, t, tok, fbt, beam=5, steering=Steering(exclude_items=excl))
    for b in range(items.size(0)):
        assert not set(items[b].tolist()) & set(excl[b].tolist())
    bias = torch.full((t.sizes[0],), -50.0)
    bias[2] = 0.0
    items, _ = beam_search(model, t, tok, fbt, beam=5, steering=Steering(first_code_bias=bias))
    valid = items[items >= 0]
    assert (t.codes_t[valid, 0] == 2).all()


def test_sampling_valid_and_logprobs_normalized(small_trie):
    t = small_trie
    model = _model(t)
    tok, fbt = _context(t, B=1)
    codes, items = sample(model, t, tok, fbt, 50)
    assert (items >= 0).all()
    assert (t.codes_t[items] == codes).all()
    all_codes = t.codes_t
    lp = sequence_logprobs(model, t, tok, fbt, all_codes, len(all_codes)).sum(1)
    assert abs(lp.exp().sum().item() - 1.0) < 1e-4


def test_chunked_beam_matches_full(small_trie):
    from encore.gen.decode import beam_search_chunked

    t = small_trie
    model = _model(t)
    tok, fbt = _context(t, B=5)
    full, s_full = beam_search(model, t, tok, fbt, beam=6)
    ch, s_ch = beam_search_chunked(model, t, tok, fbt, beam=6, max_rows=12)
    assert torch.equal(full, ch)
    assert torch.allclose(s_full, s_ch, atol=1e-5)
    excl = full[:, :2]
    a, _ = beam_search(model, t, tok, fbt, beam=6, steering=Steering(exclude_items=excl))
    b, _ = beam_search_chunked(model, t, tok, fbt, beam=6, steering=Steering(exclude_items=excl), max_rows=6)
    assert torch.equal(a, b)


def test_exclusion_with_long_lists(small_trie):
    t = small_trie
    model = _model(t)
    tok, fbt = _context(t, B=2)
    excl = torch.full((2, 500), -1, dtype=torch.long)
    excl[:, :30] = torch.arange(30)
    items, _ = beam_search(model, t, tok, fbt, beam=8, steering=Steering(exclude_items=excl))
    valid = items[items >= 0]
    assert len(valid) > 0 and (valid >= 30).all()
