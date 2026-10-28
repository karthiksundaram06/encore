import time

import numpy as np
import torch

from encore.data.store import Store
from encore.gen.decode import beam_rows_cap, beam_search_chunked
from encore.gen.model import history_tokens, load_generator
from encore.sid.trie import SIDTrie
from encore.utils import RunPaths, autocast, count_params, get_device, log, save_json


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _pick_checkpoint(paths):
    for name in ("grpo_ensemble", "sft"):
        if (paths.ckpt / f"{name}.pt").exists():
            return name
    raise FileNotFoundError("no generator checkpoint")


@torch.no_grad()
def run_bench(cfg, n_requests=200, beams=(10, 20, 50, 100), batch_sizes=(1, 32)):
    paths = RunPaths(cfg)
    device = get_device(cfg.device)
    store = Store(paths.data)
    trie = SIDTrie.load(paths.sid).to(device)
    name = _pick_checkpoint(paths)
    model, _ = load_generator(paths.ckpt / f"{name}.pt", device)
    es = store.eval_set("test")
    H = cfg.generator.history
    rng = np.random.default_rng(cfg.seed)
    max_rows = beam_rows_cap(model, trie.n_levels, H, cfg.eval.get("beam_cache_gb", 3.0),
                             cfg.amp and device.type == "cuda")
    results = {"checkpoint": name, "device": str(device), "params_m": count_params(model) / 1e6,
               "context_tokens": 1 + H * trie.n_levels, "max_beam_rows": max_rows, "runs": []}
    for bs in batch_sizes:
        for beam in beams:
            n = max(5, n_requests // bs) if bs > 1 else n_requests
            lat, pre = [], []
            for i in range(n + 5):
                idx = rng.integers(len(es), size=bs)
                items = torch.as_tensor(es.hist_items[idx], device=device)
                fb = torch.as_tensor(es.hist_fb[idx], device=device)
                _sync(device)
                t0 = time.perf_counter()
                tok, fbt = history_tokens(trie, items, fb, H)
                with autocast(device, cfg.amp):
                    model(tok, fbt)
                _sync(device)
                t1 = time.perf_counter()
                with autocast(device, cfg.amp):
                    beam_search_chunked(model, trie, tok, fbt, beam, max_rows=max_rows)
                _sync(device)
                t2 = time.perf_counter()
                if i >= 5:
                    pre.append((t1 - t0) * 1000)
                    lat.append((t2 - t1) * 1000)
            lat = np.array(lat)
            row = {
                "batch": bs, "beam": beam, "p50_ms": float(np.percentile(lat, 50)),
                "p99_ms": float(np.percentile(lat, 99)), "prefill_p50_ms": float(np.percentile(pre, 50)),
                "users_per_s": float(bs * 1000 / lat.mean()),
                "chunked": bool(bs * beam > max_rows),
            }
            results["runs"].append(row)
            log(f"batch={bs:3d} beam={beam:3d}: p50={row['p50_ms']:.1f}ms p99={row['p99_ms']:.1f}ms "
                f"(prefill {row['prefill_p50_ms']:.1f}ms) -> {row['users_per_s']:.0f} users/s"
                f"{' [chunked]' if row['chunked'] else ''}")
    save_json(results, paths.reports / "latency.json")
    return results
