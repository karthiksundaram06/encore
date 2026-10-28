import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from encore.data.store import Store
from encore.eval.recommenders import GeneratorRec
from encore.gen.decode import Steering
from encore.gen.model import load_generator
from encore.sid.trie import SIDTrie
from encore.utils import RunPaths, get_device, log

STATIC = Path(__file__).parent / "static"
FB_NAMES = {1: "skip", 2: "complete", 3: "like", 4: "dislike"}


class Event(BaseModel):
    item: int
    fb: int = 2


class RecRequest(BaseModel):
    history: list[Event]
    models: list[str] | None = None
    k: int = Field(12, ge=1, le=100)
    beam: int | None = Field(None, ge=1, le=200)
    exclude_recent: int = Field(0, ge=0, le=500)
    boost_code: int | None = None
    boost: float = 3.0


def create_app(cfg):
    paths = RunPaths(cfg)
    device = get_device(cfg.device)
    store = Store(paths.data)
    trie = SIDTrie.load(paths.sid).to(device)
    es = store.eval_set("test")
    pop_rank = np.empty(store.n_items, np.int64)
    pop_rank[np.argsort(-store.train_positives)] = np.arange(store.n_items)
    H = cfg.generator.history
    models = {}
    for name in ["sft", "grpo_hybrid", "grpo_hit", "grpo_ensemble", "grpo_single", "grpo_oracle"]:
        p = paths.ckpt / f"{name}.pt"
        if p.exists():
            m, _ = load_generator(p, device)
            models[name] = GeneratorRec(m, trie, H, cfg.serve.beam, device, name=name, amp=cfg.amp,
                                        cache_gb=cfg.eval.get("beam_cache_gb", 3.0))
    if not models:
        raise RuntimeError(f"no generator checkpoints in {paths.ckpt}")
    log(f"serving models: {list(models)} on {device}")
    latencies = deque(maxlen=2000)
    app = FastAPI(title="Encore", version="0.1.0")

    def item_info(i, fb=None):
        i = int(i)
        row = {"item": i, "codes": trie.codes[i].tolist(), "cold": bool(store.is_cold[i]),
               "pop_rank": int(pop_rank[i]), "raw_id": int(store.raw_item_id[i])}
        if fb is not None:
            row["fb"] = int(fb)
            row["fb_name"] = FB_NAMES.get(int(fb), "none")
        return row

    @app.get("/health")
    def health():
        return {"status": "ok", "models": list(models), "device": str(device)}

    @app.get("/api/meta")
    def meta():
        return {
            "models": list(models), "n_items": store.n_items, "levels": trie.sizes,
            "history": H, "default_beam": cfg.serve.beam, "default_k": cfg.serve.k, "dataset": cfg.name,
        }

    @app.get("/api/users")
    def users(n: int = 8, seed: int | None = None):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(es), size=min(n, len(es)), replace=False)
        out = []
        for i in idx:
            items, fbs = es.hist_items[i], es.hist_fb[i]
            hist = [item_info(it, f) for it, f in zip(items, fbs) if it >= 0][-H:]
            out.append({"row": int(i), "uid": int(es.uids[i]), "history": hist,
                        "targets": [int(t) for t in es.targets(i)[:50]]})
        return out

    @app.get("/api/item/{i}")
    def item(i: int):
        if not 0 <= i < store.n_items:
            raise HTTPException(404, "unknown item")
        return item_info(i)

    @app.post("/api/recommend")
    def recommend(req: RecRequest):
        hist = [e for e in req.history if 0 <= e.item < store.n_items][-H:]
        items = np.full((1, H), -1, np.int64)
        fbs = np.zeros((1, H), np.int64)
        if hist:
            items[0, -len(hist):] = [e.item for e in hist]
            fbs[0, -len(hist):] = [e.fb for e in hist]
        steer = Steering()
        if req.exclude_recent and hist:
            ex = [e.item for e in hist[-req.exclude_recent:]]
            steer.exclude_items = torch.as_tensor([ex], device=device)
        if req.boost_code is not None and 0 <= req.boost_code < trie.sizes[0]:
            bias = torch.zeros(trie.sizes[0], device=device)
            bias[req.boost_code] = req.boost
            steer.first_code_bias = bias
        names = [m for m in (req.models or list(models)) if m in models]
        hist_set = {e.item for e in hist}
        result = {}
        for name in names:
            rec = models[name]
            rec.beam = req.beam or cfg.serve.beam
            t0 = time.perf_counter()
            recs, scores = rec.recommend({"items": items, "fb": fbs}, req.k, steer, return_scores=True)
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            rows = []
            for it, sc in zip(recs[0], scores[0]):
                if it < 0:
                    continue
                r = item_info(it)
                r["score"] = float(sc)
                r["in_history"] = int(it) in hist_set
                rows.append(r)
            result[name] = {"items": rows, "latency_ms": ms}
        return result

    @app.get("/api/stats")
    def stats():
        lat = np.array(latencies) if latencies else np.zeros(1)
        return {"requests": len(latencies), "p50_ms": float(np.percentile(lat, 50)),
                "p99_ms": float(np.percentile(lat, 99))}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def serve(cfg):
    import uvicorn

    app = create_app(cfg)
    port = int(os.environ.get("PORT", cfg.serve.port))
    log(f"open http://localhost:{port} for the demo")
    uvicorn.run(app, host=cfg.serve.host, port=port, log_level="warning")


def app_from_env():
    from encore.config import load_config

    return create_app(load_config(os.environ.get("ENCORE_CONFIG", "configs/tiny.yaml")))
