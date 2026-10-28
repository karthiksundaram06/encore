from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from encore.utils import log

EVENT_COLS = ("uid", "timestamp", "item_id", "is_organic", "played_ratio_pct")


def download(cfg):
    from huggingface_hub import hf_hub_download

    raw_dir = Path(cfg.paths.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, rel in cfg.data.files.items():
        target = raw_dir / rel
        if target.exists():
            log(f"{name}: already present at {target}")
            continue
        log(f"downloading {cfg.data.yambda.repo_id}/{rel}")
        hf_hub_download(
            repo_id=cfg.data.yambda.repo_id,
            repo_type="dataset",
            filename=rel,
            local_dir=str(raw_dir),
        )
    log(f"raw files in {raw_dir}")


def _path(cfg, name):
    p = Path(cfg.paths.raw) / cfg.data.files[name]
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python -m encore download --config ...` (Yambda) "
            f"or `python -m encore prepare` (synthetic generates it)."
        )
    return p


def _rename_map(cfg):
    cols = cfg.data.columns
    return {cols[c]: c for c in EVENT_COLS if c in cols}


def read_events(cfg, name, columns):
    path = _path(cfg, name)
    inv = _rename_map(cfg)
    schema = pq.read_schema(path)
    want = [src for src, dst in inv.items() if dst in columns and src in schema.names]
    df = pq.read_table(path, columns=want).to_pandas()
    df = df.rename(columns=inv)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}; has {schema.names}. Fix data.columns in the config.")
    return df


def read_listens(cfg):
    df = read_events(cfg, "listens", EVENT_COLS)
    df["uid"] = df["uid"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["is_organic"] = df["is_organic"].astype(np.int8)
    df["played_ratio_pct"] = df["played_ratio_pct"].astype(np.float32).clip(0, 100)
    return df


def read_feedback(cfg, name):
    df = read_events(cfg, name, ("uid", "timestamp", "item_id"))
    for c in ("uid", "item_id", "timestamp"):
        df[c] = df[c].astype(np.int64)
    return df


def read_embeddings(cfg, item_ids):
    path = _path(cfg, "embeddings")
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    emb_col = next((c for c in cfg.data.columns.embedding if c in names), None)
    if emb_col is None:
        raise KeyError(f"no embedding column among {cfg.data.columns.embedding} in {names}")
    id_col = cfg.data.columns.item_id
    wanted = pa.array(np.asarray(item_ids, dtype=np.int64))
    ids_out, vecs_out = [], []
    for batch in pf.iter_batches(batch_size=200_000, columns=[id_col, emb_col]):
        ids = pc.cast(batch.column(0), pa.int64())
        keep = pc.is_in(ids, value_set=wanted)
        if not pc.any(keep).as_py():
            continue
        ids = pc.filter(ids, keep).to_numpy()
        lists = pc.filter(batch.column(1), keep)
        flat = lists.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        vecs_out.append(flat.reshape(len(ids), -1))
        ids_out.append(ids)
    if not ids_out:
        raise ValueError("no embeddings matched the catalog items")
    ids = np.concatenate(ids_out)
    vecs = np.concatenate(vecs_out)
    log(f"loaded {len(ids)} embeddings of dim {vecs.shape[1]} from column '{emb_col}'")
    return ids, vecs


def inspect(cfg):
    for name in cfg.data.files:
        p = Path(cfg.paths.raw) / cfg.data.files[name]
        if not p.exists():
            log(f"{name}: MISSING ({p})")
            continue
        pf = pq.ParquetFile(p)
        log(f"{name}: {pf.metadata.num_rows:,} rows | columns {pf.schema_arrow.names}")
        head = next(pf.iter_batches(batch_size=3)).to_pandas()
        print(head.to_string(max_colwidth=40))
    try:
        ts = pq.read_table(_path(cfg, "listens"), columns=[cfg.data.columns.timestamp]).column(0)
        lo, hi = pc.min_max(ts).values()
        span = (hi.as_py() - lo.as_py()) * cfg.data.time_unit_seconds / 86400
        log(f"listens timeline spans {span:.1f} days with time_unit_seconds={cfg.data.time_unit_seconds}")
    except FileNotFoundError:
        pass
