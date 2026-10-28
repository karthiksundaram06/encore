from pathlib import Path

import numpy as np
import torch

PAD, BOS = 0, 1
N_SPECIAL = 2


class SIDTrie:
    def __init__(self, codes, level_sizes):
        codes = np.asarray(codes, dtype=np.int64)
        self.codes = codes
        self.sizes = [int(s) for s in level_sizes]
        self.n_levels = len(self.sizes)
        self.offsets = [N_SPECIAL + int(sum(self.sizes[:l])) for l in range(self.n_levels)]
        self.vocab_size = N_SPECIAL + int(sum(self.sizes))
        n = len(codes)
        node = np.zeros(n, np.int64)
        n_nodes = 1
        tables = []
        for l, K in enumerate(self.sizes):
            table = np.full((n_nodes, K), -1, np.int32)
            if l < self.n_levels - 1:
                key = node * K + codes[:, l]
                uniq, inv = np.unique(key, return_inverse=True)
                table[uniq // K, uniq % K] = np.arange(len(uniq))
                node, n_nodes = inv.reshape(-1), len(uniq)
            else:
                table[node, codes[:, l]] = np.arange(n)
                check = table[node, codes[:, l]]
                if not (check == np.arange(n)).all():
                    raise ValueError("semantic IDs are not unique")
            tables.append(table)
        self.np_child = tables
        self.child = [torch.from_numpy(t) for t in tables]
        self.tokens = torch.from_numpy(codes + np.asarray(self.offsets)[None])
        self.codes_t = torch.from_numpy(codes)
        self.device = torch.device("cpu")

    def to(self, device):
        self.child = [t.to(device) for t in self.child]
        self.tokens = self.tokens.to(device)
        self.codes_t = self.codes_t.to(device)
        self.device = torch.device(device)
        return self

    def item_tokens(self, items):
        return self.tokens[items]

    def lookup(self, codes):
        codes = np.asarray(codes, dtype=np.int64)
        node = np.zeros(len(codes), np.int64)
        ok = np.ones(len(codes), bool)
        for l, K in enumerate(self.sizes):
            c = codes[:, l]
            ok &= (c >= 0) & (c < K)
            nxt = self.np_child[l][np.where(ok, node, 0), np.where(ok, c, 0)]
            ok &= nxt >= 0
            node = np.maximum(nxt, 0)
        return np.where(ok, node, -1)

    @classmethod
    def load(cls, sid_dir):
        z = np.load(Path(sid_dir) / "sid.npz")
        return cls(z["codes"], z["level_sizes"])
