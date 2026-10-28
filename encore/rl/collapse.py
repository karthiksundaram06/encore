import numpy as np

from encore.utils import RunPaths, log, read_jsonl, save_json

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]

OPTIMIZED = {"single": "sim_single", "ensemble": "sim_pessimistic", "oracle": "true_utility"}
TITLES = {
    "single": "Single simulator",
    "ensemble": "Pessimistic ensemble",
    "hit": "Hit reward (logged next items)",
    "hybrid": "Hybrid: hit + world model on misses",
    "oracle": "True-listener reward (upper bound)",
}
ORDER = ("hit", "single", "ensemble", "hybrid", "oracle")


def _evals(paths, mode):
    p = paths.logs / f"grpo_{mode}.jsonl"
    if not p.exists():
        return None
    return [r for r in read_jsonl(p) if r.get("kind") == "eval"]


def _trains(paths, mode):
    p = paths.logs / f"grpo_{mode}.jsonl"
    return [r for r in read_jsonl(p) if r.get("kind") == "train"] if p.exists() else []


def _rel(vals):
    base = vals[0]
    if abs(base) < 1e-9:
        return np.zeros_like(vals)
    return 100.0 * (vals - base) / abs(base)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.axhline(0, color="#b5b4ae", linewidth=1)


def plot_collapse(paths, modes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [m for m in modes if _evals(paths, m)]
    if not modes:
        raise FileNotFoundError("no GRPO logs found")
    has_true = "true_utility" in _evals(paths, modes[0])[0]
    judge_key = "true_utility" if has_true else "sim_audit"
    judge_name = "True listener utility" if has_true else "Held-out audit simulator"
    series = [
        (SERIES[0], "optimized", "Reward being optimized"),
        (SERIES[1], "ndcg@10", "Held-out NDCG@10 (all tracks)"),
        (SERIES[3], "new_ndcg@10", "Held-out NDCG@10 (new tracks)"),
        (SERIES[2], judge_key, judge_name),
    ]
    ncols = min(3, len(modes))
    nrows = int(np.ceil(len(modes) / ncols))
    spare = nrows * ncols - len(modes)
    top = 1.0 if spare else 1 - 0.07 / nrows
    fig, grid = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows + (0 if spare else 0.4)),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    grid = grid.ravel()
    for ax in grid[len(modes):]:
        ax.set_visible(False)
    axes = grid[: len(modes)]
    summary = {}
    found = {}
    for ax, mode in zip(axes, modes):
        ev = _evals(paths, mode)
        steps = np.array([r["step"] for r in ev])
        _style(ax)
        summary[mode] = {}
        ends = []
        for color, key, label in series:
            k = OPTIMIZED.get(mode) if key == "optimized" else key
            if k is None or k not in ev[0] or (key == "optimized" and k == judge_key):
                continue
            raw = np.array([r[k] for r in ev], dtype=float)
            y = _rel(raw)
            (line,) = ax.plot(steps, y, color=color, linewidth=2, label=label, solid_capstyle="round")
            found.setdefault(label, line)
            ax.plot(steps[-1], y[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
            ends.append([y[-1], y[-1]])
            summary[mode][label] = {"start": float(raw[0]), "end": float(raw[-1]), "change_pct": float(y[-1])}
        ax.set_title(TITLES.get(mode, mode), color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("GRPO step", color=TEXT_2, fontsize=9)
        ax.set_xlim(steps[0], steps[-1] * 1.14 if steps[-1] else 1)
        ax._encore_ends = (steps[-1], ends)
    for i in range(0, len(axes), ncols):
        axes[i].set_ylabel("Change from SFT start (%)", color=TEXT_2, fontsize=9)
    lo, hi = axes[0].get_ylim()
    gap = 0.07 * (hi - lo)
    for ax in axes:
        x_end, ends = ax._encore_ends
        order = sorted(range(len(ends)), key=lambda j: ends[j][0])
        for a, b in zip(order, order[1:]):
            if ends[b][1] - ends[a][1] < gap:
                ends[b][1] = ends[a][1] + gap
        dx = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        for value, placed in ends:
            ax.text(x_end + dx, placed, f"{value:+.0f}%", va="center", fontsize=9, color=TEXT, clip_on=False)
    labels = [label for _, _, label in series if label in found]
    handles = [found[label] for label in labels]
    if spare:
        slot = grid[len(modes)]
        slot.set_visible(True)
        slot.axis("off")
        slot.legend(handles, labels, loc="center left", frameon=False, fontsize=10, labelcolor=TEXT)
        fig.tight_layout()
    else:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
                   labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, top))
    out = paths.reports / "collapse.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log(f"chart saved to {out}")
    return summary


def plot_zero_variance(paths, modes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.4), facecolor=SURFACE)
    _style(ax)
    out = {}
    for color, mode in zip(SERIES, modes):
        tr = _trains(paths, mode)
        if not tr:
            continue
        z = np.array([r["zero_var_frac"] for r in tr]) * 100
        w = max(1, len(z) // 15)
        zs = np.convolve(z, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(zs)) + w, zs, color=color, linewidth=2, label=TITLES.get(mode, mode))
        out[mode] = float(z.mean())
    ax.set_ylabel("Groups with zero reward variance (%)", color=TEXT_2, fontsize=9)
    ax.set_xlabel("GRPO step", color=TEXT_2, fontsize=9)
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    ax.set_title("Zero-advantage groups: no learning signal", color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(paths.reports / "zero_variance.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def run_collapse(cfg, skip_train=False):
    from encore.data.oracle import Oracle
    from encore.rl.grpo import run_grpo

    paths = RunPaths(cfg)
    modes = list(cfg.grpo.modes)
    if "oracle" in modes and not Oracle.available(cfg.paths.raw):
        log("no hidden true listener for this dataset; skipping the oracle reward")
        modes.remove("oracle")
    if not skip_train:
        for mode in modes:
            run_grpo(cfg, mode)
    logged = [m for m in ORDER if (paths.logs / f"grpo_{m}.jsonl").exists()]
    summary = {"curves": plot_collapse(paths, logged)}
    summary["zero_variance_pct"] = plot_zero_variance(paths, [m for m in logged if m != "oracle"])
    save_json(summary, paths.reports / "collapse.json")
    for mode, rows in summary["curves"].items():
        log(f"{mode}: " + "; ".join(f"{k} {v['change_pct']:+.1f}%" for k, v in rows.items()))
    log(f"zero-variance groups: {summary['zero_variance_pct']}")
    return summary
