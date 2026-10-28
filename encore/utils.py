import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def log(msg):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(spec="auto"):
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, enabled=True):
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def configure_threads(cfg):
    n = cfg.get("num_threads")
    if n:
        torch.set_num_threads(int(n))


class RunPaths:
    def __init__(self, cfg):
        self.root = Path(cfg.paths.run)
        self.raw = Path(cfg.paths.raw)
        self.data = self.root / "data"
        self.sid = self.root / "sid"
        self.ckpt = self.root / "ckpt"
        self.logs = self.root / "logs"
        self.reports = self.root / "reports"
        for p in (self.data, self.sid, self.ckpt, self.logs, self.reports):
            p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def load_json(path):
    return json.loads(Path(path).read_text())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


class JsonlLogger:
    def __init__(self, path, fresh=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and self.path.exists():
            self.path.unlink()

    def write(self, row):
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cosine_schedule(optimizer, warmup, total, min_ratio=0.1):
    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name or "norm" in name or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def progress(i, n, every, prefix, **metrics):
    if i % every == 0 or i == n - 1:
        parts = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        log(f"{prefix} {i + 1}/{n} {parts}")
        sys.stdout.flush()
