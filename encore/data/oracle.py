from pathlib import Path

import numpy as np
import pandas as pd

from encore.data.synthetic import engagement_probs, expected_utility, true_score
from encore.utils import load_json


class Oracle:
    def __init__(self, raw_dir, store, weights):
        raw_dir = Path(raw_dir)
        z = np.load(raw_dir / "oracle.npz")
        self.p = load_json(raw_dir / "oracle_params.json")
        self.weights = dict(weights)
        self.v, self.q, self.u, self.theta = z["v"], z["q"], z["u"], z["theta"]
        self.item_genre = z["item_genre"]
        self.cat_to_syn = pd.Index(z["item_raw"]).get_indexer(store.raw_item_id)
        self.user_to_syn = pd.Index(z["user_raw"]).get_indexer(store.user_raw)
        assert (self.cat_to_syn >= 0).all() and (self.user_to_syn >= 0).all()

    @staticmethod
    def available(raw_dir):
        return (Path(raw_dir) / "oracle.npz").exists()

    def state(self, uids, hist_items, hist_complete):
        su = self.user_to_syn[uids]
        u = self.u[su]
        m = u.copy()
        r = self.p["state_rate"]
        for j in range(hist_items.shape[1]):
            it = hist_items[:, j]
            mask = (it >= 0) & (hist_complete[:, j] > 0)
            if mask.any():
                m[mask] = (1 - r) * m[mask] + r * self.v[self.cat_to_syn[it[mask]]]
        mix = self.p["state_mix"]
        u_eff = (1 - mix) * u + mix * m
        w = self.p["recent_window"]
        recent = hist_items[:, -w:]
        recent = np.where(recent >= 0, self.cat_to_syn[np.maximum(recent, 0)], -1)
        return u_eff, recent, self.theta[su]

    def utility(self, uids, hist_items, hist_complete, cand):
        cand = np.asarray(cand)
        u_eff, recent, theta = self.state(np.asarray(uids), np.asarray(hist_items), np.asarray(hist_complete))
        valid = cand >= 0
        sc = self.cat_to_syn[np.maximum(cand, 0)]
        s = true_score(u_eff, self.v[sc], self.q[sc], recent, sc, self.p["fatigue"])
        pc, pl, pd = engagement_probs(s, theta[:, None], self.p)
        util = expected_utility(pc, pl, pd, self.weights)
        return np.where(valid, util, np.nan)
