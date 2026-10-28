# Running Encore on your own machine

This guide takes you from a fresh checkout to the real Yambda-50M results on a single GPU (written for an RTX 4090 laptop, 16 GB VRAM). Every command is `python -m encore ...`, so it works the same in PowerShell, cmd, bash and zsh.

## 0. Requirements

| What | Version / size | Check |
|---|---|---|
| Python | 3.10 – 3.12 (3.11 recommended) | `python --version` |
| NVIDIA driver | recent enough for CUDA 12.x | `nvidia-smi` |
| PyTorch | 2.2+ with CUDA | step 2 below |
| Disk | ~25 GB free (Yambda-50M raw files + embeddings + checkpoints) | |
| RAM | 32 GB recommended for Yambda-50M preprocessing; 8 GB is enough for the synthetic configs | |
| git | any | `git --version` |
| Docker Desktop | optional, only for the serving container | `docker --version` |

## 1. Get the code and make a virtual environment

```bash
cd encore            # the folder you unzipped / cloned
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows cmd: `.venv\Scripts\activate.bat`
- Linux / macOS: `source .venv/bin/activate`

## 2. Install PyTorch with CUDA, then Encore

Open https://pytorch.org/get-started/locally/, pick *Stable · your OS · Pip · Python · CUDA 12.x*, and run the command it shows (it looks like `pip install torch --index-url https://download.pytorch.org/whl/cu12X`). Then:

```bash
pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and your GPU name. If it prints `False`, the CPU wheel got installed: `pip uninstall torch` and repeat the pytorch.org command.

## 3. Smoke test (≈2 min on GPU, ≈10 min on CPU)

```bash
pytest
python -m encore pipeline --config configs/ci.yaml
python -m encore check --config configs/ci.yaml
```

`pytest` checks the parts that fail silently: KV-cache decoding equals a full forward pass, trie-constrained beam search equals exhaustive ranking, the constrained sampler's probabilities sum to 1, Semantic IDs are unique. The pipeline runs every stage on a 400-user synthetic world and `check` asserts the generator beats popularity and the chart exists.

## 4. Full synthetic run (≈5 min on GPU)

```bash
python -m encore pipeline --config configs/tiny.yaml
```

Outputs land in `runs/tiny/reports/`:

- `results_test.md` – the comparison table (popularity, SASRec, two-tower, SFT generator, GRPO variants)
- `collapse.png` – the single-simulator vs pessimistic-ensemble chart
- `zero_variance.png` – how often GRPO groups get no learning signal under each reward
- `world_report.json` – listener-model AUC/calibration and the in-distribution vs out-of-distribution analysis
- `sid_report.json`, `latency.json`

The synthetic world has a hidden "true listener", so `true_utility@10` tells you whether a policy really got better or only fooled its simulator. That is what lets you trust the method before spending GPU hours on real data.

## 5. Real data: Yambda-50M

### 5a. Download and verify the schema

```bash
python -m encore download --config configs/yambda50m.yaml
```

This pulls `flat/50m/{listens,likes,dislikes}.parquet` and `embeddings.parquet` from the `yandex/yambda` dataset on Hugging Face into `data/raw/yambda/`, then prints each file's columns, three sample rows and the timeline span.

**Check two things in that output before going further:**

1. **Column names.** The config expects `uid, timestamp, item_id, is_organic, played_ratio_pct` in listens and `item_id` plus `normalized_embed` (or `embed`) in embeddings. If the printed names differ, edit `data.columns` in `configs/base.yaml`.
2. **Timeline span.** It should print roughly 300 days. If it prints something wildly different (e.g. 60 days or 1500 days), the timestamp unit is wrong: change `data.time_unit_seconds` in `configs/yambda50m.yaml` until the span is right. The temporal split (last day = test, the day before = validation, 30-minute gaps) depends on it.

Checked against the real files: Yambda-50M has 46.5M listens over 300.9 days, with timestamps in seconds rounded to 5-second bins, so the config uses `time_unit_seconds: 1`. If the download fails with 401/403, run `huggingface-cli login` (free account) and retry; if a file path changed upstream, fix `data.files` in the config.

### 5b. Run the stages

Run them one at a time the first time, so a problem shows up early. Rough times on a laptop 4090 (estimates – your mileage will vary):

| Command | What it does | Rough time |
|---|---|---|
| `python -m encore prepare --config configs/yambda50m.yaml` | labels, catalog, global temporal split, eval sets | 10–20 min (CPU/RAM bound) |
| `python -m encore baselines --config configs/yambda50m.yaml` | popularity, SASRec, two-tower | 1–2 h |
| `python -m encore sid --config configs/yambda50m.yaml` | fused audio+collaborative embeddings → RQ-KMeans 256×256×256 + dedup | 5–15 min |
| `python -m encore sft --config configs/yambda50m.yaml` | Semantic-ID generator (≈35M params, bf16) | 2–5 h |
| `python -m encore world --config configs/yambda50m.yaml` | 5 listener models + 1 held-out audit model, then the calibration report | 1–3 h |
| `python -m encore collapse --config configs/yambda50m.yaml` | GRPO with hit / single / ensemble rewards + charts | 3–6 h |
| `python -m encore eval --config configs/yambda50m.yaml` | final test-set table | 15–30 min |
| `python -m encore bench --config configs/yambda50m.yaml` | p50/p99 latency for beam 10–100 | 5 min |

Or all at once: `python -m encore pipeline --config configs/yambda50m.yaml` (add `--start sft` to resume from a stage).

On real data there is no hidden true listener, so the collapse chart's third line becomes the **held-out audit simulator**: a listener model trained like the others but never used as a reward. Held-out NDCG@10 is the other honest signal.

### 5c. If you run out of memory

Any config value can be overridden from the command line with `--set`:

```bash
python -m encore sft --config configs/yambda50m.yaml --set generator.batch_size=24 --set generator.history=64
python -m encore baselines --config configs/yambda50m.yaml --set sasrec.num_negatives=1024
python -m encore prepare --config configs/yambda50m.yaml --set data.max_items=300000
```

## 6. The demo and the API

```bash
python -m encore serve --config configs/yambda50m.yaml      # or configs/tiny.yaml
```

Open http://localhost:8000. Pick a listener, then compare SFT and GRPO recommendations side by side. Change feedback on history items, add recommended tracks with ▶ ♥ ⏭, exclude recent listens or boost a sound cluster, and watch the recommendations and latency update. Green "listened" badges mark tracks the user actually completed in the held-out window.

API: `GET /health`, `GET /api/meta`, `GET /api/users?n=8`, `POST /api/recommend`, `GET /api/stats` (p50/p99 of served requests). Interactive docs at http://localhost:8000/docs.

Docker (CPU image; mount your run folder):

```bash
docker build -t encore .
docker run -p 8000:8000 -v "$(pwd)/runs:/app/runs" -v "$(pwd)/data:/app/data" -e ENCORE_CONFIG=configs/tiny.yaml encore
```

## 7. Putting it on GitHub

```bash
git init && git add . && git commit -m "Encore: generative recommendation with pessimistic world-model GRPO"
git branch -M main
git remote add origin https://github.com/<you>/encore.git
git push -u origin main
```

`.github/workflows/ci.yml` then runs the unit tests, the synthetic end-to-end pipeline, the regression check and a serving smoke test on every push, and uploads the reports as a build artifact.

## 8. Useful knobs

| Setting | Where | Effect |
|---|---|---|
| `reward.pessimism` | config | λ in mean − λ·std; 0 turns the ensemble into a plain average |
| `reward.diversity` | config | weight of the within-group diversity bonus |
| `grpo.kl_coef` | config | pull toward the SFT model |
| `grpo.modes` | config | which reward variants `collapse` trains |
| `sid.collab_source` | config | `sasrec` (default) or `ppmi` collaborative half of the Semantic-ID embedding |
| `sid.audio_weight` | config | audio vs collaborative balance in the fused embedding |
| `world.members` | config | ensemble size (plus one held-out audit model) |
