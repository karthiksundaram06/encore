import numpy as np
import torch

from encore.gen.decode import Steering, beam_rows_cap, beam_search_chunked
from encore.gen.model import history_tokens
from encore.utils import autocast


def _pad_topk(scores, k):
    k = min(k, scores.size(1))
    idx = scores.topk(k, dim=1)
    return torch.where(torch.isfinite(idx.values), idx.indices, torch.full_like(idx.indices, -1))


def _mask_excluded(scores, exclude):
    if not exclude:
        return scores
    rows = np.concatenate([np.full(len(e), i) for i, e in enumerate(exclude)])
    cols = np.concatenate([np.asarray(e, dtype=np.int64) for e in exclude])
    if len(cols):
        scores[torch.as_tensor(rows, device=scores.device), torch.as_tensor(cols, device=scores.device)] = float("-inf")
    return scores


def _pad_rows(rows, k):
    out = np.full((len(rows), k), -1, np.int64)
    for i, r in enumerate(rows):
        r = np.asarray(r[:k])
        out[i, : len(r)] = r
    return out


class PopularityRec:
    name = "popularity"

    def __init__(self, ranking):
        self.ranking = np.asarray(ranking)

    def recommend(self, batch, k, exclude=None):
        B = len(batch["items"])
        if exclude is None:
            return np.tile(self.ranking[:k], (B, 1))
        return _pad_rows([self.ranking[~np.isin(self.ranking, ex)] for ex in exclude], k)


class HistoryRec:
    name = "history_repeat"
    discovery = False

    def recommend(self, batch, k, exclude=None):
        rows = []
        for items, comp in zip(batch["items"], batch["complete"]):
            h = items[(items >= 0) & (comp > 0)]
            if len(h) == 0:
                rows.append([])
                continue
            uniq, first, cnt = np.unique(h[::-1], return_index=True, return_counts=True)
            rows.append(uniq[np.lexsort((first, -cnt))])
        if exclude is not None:
            rows = [r[~np.isin(r, ex)] if len(r) else r for r, ex in zip(rows, exclude)]
        return _pad_rows(rows, k)


class SASRecRec:
    name = "sasrec"

    def __init__(self, model, is_cold, H, device):
        self.model = model.eval()
        self.H = H
        self.device = device
        self.cold = torch.as_tensor(is_cold, device=device)

    @torch.no_grad()
    def recommend(self, batch, k, exclude=None):
        items = torch.as_tensor(batch["items"][:, -self.H:], device=self.device)
        fb = torch.as_tensor(batch["fb"][:, -self.H:], device=self.device)
        with autocast(self.device):
            s = self.model.scores(items, fb).float()
        s = _mask_excluded(s.masked_fill(self.cold[None], float("-inf")), exclude)
        return _pad_topk(s, k).cpu().numpy()


class TwoTowerRec:
    name = "two_tower"

    def __init__(self, model, H, device):
        self.model = model.eval()
        self.H = H
        self.device = device
        with torch.no_grad():
            self.item_vecs = model.all_item_vecs()

    @torch.no_grad()
    def recommend(self, batch, k, exclude=None):
        items = torch.as_tensor(batch["items"][:, -self.H:], device=self.device)
        fb = torch.as_tensor(batch["fb"][:, -self.H:], device=self.device)
        s = _mask_excluded(self.model.scores(items, fb, self.item_vecs).float(), exclude)
        return _pad_topk(s, k).cpu().numpy()


class GeneratorRec:
    def __init__(self, model, trie, H, beam, device, name="generator", amp=True, cache_gb=3.0):
        self.model = model.eval()
        self.trie = trie
        self.H = H
        self.beam = beam
        self.device = device
        self.name = name
        self.amp = amp
        half = amp and device.type == "cuda"
        self.max_rows = beam_rows_cap(model, trie.n_levels, H, cache_gb, half)

    @torch.no_grad()
    def recommend(self, batch, k, steering=None, return_scores=False, exclude=None):
        items = torch.as_tensor(batch["items"], device=self.device)
        fb = torch.as_tensor(batch["fb"], device=self.device)
        tok, fbt = history_tokens(self.trie, items, fb, self.H)
        steering = steering or Steering()
        if exclude is not None:
            E = max(1, max(len(e) for e in exclude))
            ex = torch.full((len(exclude), E), -1, dtype=torch.long, device=self.device)
            for i, e in enumerate(exclude):
                if len(e):
                    ex[i, : len(e)] = torch.as_tensor(np.asarray(e, dtype=np.int64), device=self.device)
            steering = Steering(ex, steering.first_code_bias, steering.temperature, steering.extra)
        with autocast(self.device, self.amp):
            out, scores = beam_search_chunked(self.model, self.trie, tok, fbt, max(self.beam, k),
                                              steering, self.max_rows)
        out = out[:, :k].cpu().numpy()
        if out.shape[1] < k:
            out = np.pad(out, ((0, 0), (0, k - out.shape[1])), constant_values=-1)
        if return_scores:
            return out, scores[:, :k].float().cpu().numpy()
        return out
