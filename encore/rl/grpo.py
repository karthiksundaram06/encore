import copy

import numpy as np
import torch

from encore.data.store import Store
from encore.data.windows import gather
from encore.eval.evaluate import collect, history_sets
from encore.eval.metrics import beyond_accuracy, discovery_metrics, ranking_metrics
from encore.eval.recommenders import GeneratorRec
from encore.gen.decode import sample, sequence_logprobs
from encore.gen.model import history_tokens, load_generator
from encore.rl.reward import diversity_bonus, load_ensemble
from encore.sid.trie import SIDTrie
from encore.train_loop import save_checkpoint
from encore.utils import JsonlLogger, RunPaths, Timer, autocast, get_device, log, seed_all


def sample_contexts(store, B, H, min_hist, horizon, rng):
    n_train = store.n_upto[0]
    eligible = np.nonzero(n_train > min_hist)[0]
    users = rng.choice(eligible, size=B)
    a = store.user_ptr[users]
    cut = a + rng.integers(min_hist, n_train[users])
    batch = gather(store, np.maximum(a, cut - H), cut, H)
    fut = cut[:, None] + np.arange(horizon)[None]
    ok = fut < (a + n_train[users])[:, None]
    fut = np.where(ok, fut, 0)
    pos = ok & (store.ev["complete"][fut] > 0)
    batch["future"] = np.where(pos, store.ev["item"][fut], -1).astype(np.int64)
    batch["uid"] = users.astype(np.int64)
    return batch


class Monitor:
    def __init__(self, cfg, store, trie, ens, oracle, device):
        g = cfg.grpo
        self.es = store.eval_set("val").subset(g.eval_users, cfg.seed + 7)
        self.hist = history_sets(store, self.es, 1)
        self.store = store
        self.trie = trie
        self.ens = ens
        self.oracle = oracle
        self.device = device
        self.cfg = cfg

    def __call__(self, policy):
        rec = GeneratorRec(policy, self.trie, self.cfg.generator.history, self.cfg.grpo.eval_beam,
                           self.device, amp=self.cfg.amp, cache_gb=self.cfg.eval.get("beam_cache_gb", 3.0))
        recs = collect(rec, self.es, 10, 64)
        res = ranking_metrics(recs, self.es, [10], self.store.is_cold)
        res.update(beyond_accuracy(recs, self.store.n_items, self.store.popularity(), None, self.store.is_cold, 10))
        out = {k: res[k] for k in ("ndcg@10", "recall@10", "coverage@10", "novelty@10", "cold_share@10")}
        new_recs = collect(rec, self.es, 10, 64, exclude=self.hist)
        out["new_ndcg@10"] = discovery_metrics(new_recs, self.es, self.hist, [10], self.store.is_cold)["new_ndcg@10"]
        out["sim_single"] = self.ens.eval_utility(self.es, recs, which="single")
        out["sim_mean"] = self.ens.eval_utility(self.es, recs, which="mean")
        out["sim_pessimistic"] = self.ens.eval_utility(self.es, recs, which="pessimistic")
        out["sim_std"] = self.ens.eval_utility(self.es, recs, which="std")
        if self.ens.audit is not None:
            out["sim_audit"] = self.ens.eval_utility(self.es, recs, which="audit")
        if self.oracle is not None:
            u = self.oracle.utility(self.es.uids, self.es.hist_items, self.es.hist_complete, recs)
            out["true_utility"] = float(np.nanmean(u))
        return out


def run_grpo(cfg, mode):
    from encore.data.oracle import Oracle

    paths = RunPaths(cfg)
    store = Store(paths.data)
    device = get_device(cfg.device)
    seed_all(cfg.seed)
    g = cfg.grpo
    H = cfg.generator.history
    rng = np.random.default_rng(cfg.seed)
    trie = SIDTrie.load(paths.sid).to(device)
    policy, _ = load_generator(paths.ckpt / "sft.pt", device)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ens = load_ensemble(cfg, paths, store, device)
    oracle = Oracle(cfg.paths.raw, store, cfg.reward.weights) if Oracle.available(cfg.paths.raw) else None
    if mode == "oracle" and oracle is None:
        raise ValueError("oracle reward needs a synthetic dataset")
    item_emb = ens.content
    monitor = Monitor(cfg, store, trie, ens, oracle, device)
    opt = torch.optim.AdamW(policy.parameters(), lr=g.lr, weight_decay=g.weight_decay)
    run = f"{mode}{g.get('tag') or ''}"
    logger = JsonlLogger(paths.logs / f"grpo_{run}.jsonl")
    G = g.group_size
    beta = cfg.reward.get("hybrid_beta", 0.5)
    std_floor = cfg.reward.get("hybrid_std_floor", 0.1) if mode == "hybrid" else g.get("adv_std_floor", 0.0)
    select = g.get("select_metric", "ndcg@10")
    best = {"value": -float("inf"), "step": -1}
    log(f"GRPO mode={mode}: {g.steps} steps, {g.contexts_per_step} contexts x {G} samples, "
        f"kl={g.kl_coef}, pessimism={cfg.reward.pessimism}, diversity={cfg.reward.diversity}"
        + (f", hybrid beta={beta}, std floor={std_floor}" if mode == "hybrid" else ""))

    def evaluate_now(step, extra=None):
        with Timer() as t:
            row = monitor(policy)
        row.update({"step": step, "mode": mode, "eval_seconds": t.seconds, **(extra or {})})
        logger.write({"kind": "eval", **row})
        if row[select] > best["value"]:
            best.update(value=row[select], step=step)
            save_checkpoint(policy, paths.ckpt / f"grpo_{run}_best.pt", mode=mode, steps=step,
                            select_metric=select, select_value=row[select])
        msg = " ".join(f"{k}={row[k]:.4f}" for k in ("ndcg@10", "new_ndcg@10", "sim_single", "sim_pessimistic")
                       if k in row)
        if "true_utility" in row:
            msg += f" true={row['true_utility']:.4f}"
        if "sim_audit" in row:
            msg += f" audit={row['sim_audit']:.4f}"
        log(f"[{mode}] step {step}: {msg}")

    evaluate_now(0)
    t = lambda x: torch.as_tensor(x, device=device)
    for step in range(1, g.steps + 1):
        ctx = sample_contexts(store, g.contexts_per_step, H, g.min_history, g.hit_horizon, rng)
        items, fb, org = t(ctx["item"]), t(ctx["fb"]), t(ctx["organic"])
        tok, fbt = history_tokens(trie, items, fb, H)
        with autocast(device, cfg.amp):
            codes, cand = sample(policy, trie, tok, fbt, G, g.temperature)
        cand = cand.view(-1, G)

        with torch.no_grad():
            hit = None
            if mode in ("hit", "hybrid"):
                fut = t(ctx["future"])
                hit = (cand[:, :, None] == fut[:, None, :]).any(-1).float()
            if mode == "hit":
                r = hit
            elif mode == "oracle":
                c_np = cand.cpu().numpy()
                r = t(oracle.utility(ctx["uid"], ctx["item"], ctx["complete"], c_np)).float()
            else:
                U = ens.member_utilities(items, fb, org, cand).float()
                r = ens.combine(U, "single" if mode == "single" else "ensemble", cfg.reward.pessimism)
            if cfg.reward.diversity and mode != "hit":
                r = r + cfg.reward.diversity * diversity_bonus(item_emb, cand)
            if mode == "hybrid":
                r = hit + beta * r
            r_std = r.std(1, keepdim=True)
            adv = (r - r.mean(1, keepdim=True)) / (r_std.clamp(min=std_floor) + 1e-6)
            sig = hit if hit is not None else r
            zero_var = (sig.std(1) < 1e-6).float().mean().item()
            adv = adv.reshape(-1, 1)
            with autocast(device, cfg.amp):
                ref_lp = sequence_logprobs(ref, trie, tok, fbt, codes, G).float()
                old_lp = sequence_logprobs(policy, trie, tok, fbt, codes, G).float()

        for _ in range(g.ppo_epochs):
            with autocast(device, cfg.amp):
                lp = sequence_logprobs(policy, trie, tok, fbt, codes, G).float()
            ratio = torch.exp(lp - old_lp)
            pg = -torch.minimum(ratio * adv, ratio.clamp(1 - g.clip_eps, 1 + g.clip_eps) * adv)
            d = ref_lp - lp
            kl = torch.exp(d) - d - 1
            loss = (pg + g.kl_coef * kl).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), g.max_grad_norm)
            opt.step()
        policy.eval()

        row = {
            "kind": "train", "step": step, "reward": r.mean().item(), "reward_std": r_std.mean().item(),
            "zero_var_frac": zero_var, "kl": kl.mean().item(), "loss": loss.item(), "grad_norm": float(gn),
            "nll": -lp.sum(1).mean().item(), "unique_per_group": float(np.mean([len(set(x)) for x in cand.tolist()])),
        }
        logger.write(row)
        if step % max(1, g.eval_every // 2) == 0:
            log(f"[{mode}] step {step}: reward={row['reward']:.4f} zero_var={zero_var:.2f} kl={row['kl']:.4f} "
                f"unique/group={row['unique_per_group']:.1f}")
        if step % g.eval_every == 0 or step == g.steps:
            evaluate_now(step)

    save_checkpoint(policy, paths.ckpt / f"grpo_{run}.pt", mode=mode, steps=g.steps)
    log(f"saved {paths.ckpt / f'grpo_{run}.pt'}; best {select}={best['value']:.4f} at step {best['step']} "
        f"saved as grpo_{run}_best.pt")
