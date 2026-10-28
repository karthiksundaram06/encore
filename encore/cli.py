import argparse
import sys
import time

from encore.config import load_config
from encore.utils import configure_threads, log, seed_all

STAGES = ["prepare", "baselines", "sid", "sft", "world", "collapse", "eval", "bench"]


def cmd_download(cfg, args):
    from encore.data.yambda import download, inspect

    download(cfg)
    inspect(cfg)


def cmd_inspect(cfg, args):
    from encore.data.yambda import inspect

    inspect(cfg)


def cmd_prepare(cfg, args):
    from encore.data.prepare import prepare

    prepare(cfg)


def cmd_sid(cfg, args):
    from encore.sid.build import build_sids

    build_sids(cfg)


def cmd_baselines(cfg, args):
    from encore.baselines.train import train_baselines

    only = args.only.split(",") if getattr(args, "only", None) else None
    train_baselines(cfg, only)


def cmd_sft(cfg, args):
    from encore.gen.train import train_sft

    train_sft(cfg)


def cmd_world(cfg, args):
    from encore.world.report import world_report
    from encore.world.train import train_world

    train_world(cfg)
    world_report(cfg)


def cmd_grpo(cfg, args):
    from encore.rl.grpo import run_grpo

    run_grpo(cfg, args.mode)


def cmd_collapse(cfg, args):
    from encore.rl.collapse import run_collapse

    run_collapse(cfg, skip_train=getattr(args, "plot_only", False))


def cmd_eval(cfg, args):
    from encore.eval.evaluate import run_eval

    run_eval(cfg, getattr(args, "split", "test"))


def cmd_bench(cfg, args):
    from encore.eval.bench import run_bench

    run_bench(cfg)


def cmd_serve(cfg, args):
    from encore.serve.app import serve

    serve(cfg)


def cmd_check(cfg, args):
    from encore.eval.evaluate import check

    check(cfg)


def cmd_pipeline(cfg, args):
    stages = STAGES
    if args.start:
        stages = stages[stages.index(args.start):]
    if args.stop:
        stages = stages[: stages.index(args.stop) + 1]
    t0 = time.time()
    for s in stages:
        log(f"===== stage: {s} =====")
        seed_all(cfg.seed)
        COMMANDS[s](cfg, args)
    log(f"pipeline finished in {(time.time() - t0) / 60:.1f} min")


COMMANDS = {
    "download": cmd_download,
    "inspect": cmd_inspect,
    "prepare": cmd_prepare,
    "sid": cmd_sid,
    "baselines": cmd_baselines,
    "sft": cmd_sft,
    "world": cmd_world,
    "grpo": cmd_grpo,
    "collapse": cmd_collapse,
    "eval": cmd_eval,
    "bench": cmd_bench,
    "serve": cmd_serve,
    "check": cmd_check,
    "pipeline": cmd_pipeline,
}


def build_parser():
    p = argparse.ArgumentParser(prog="encore", description="Encore: generative recommendation on Semantic IDs")
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/tiny.yaml")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
        if name == "baselines":
            sp.add_argument("--only", default=None, help="comma list of popularity,sasrec,two_tower")
        if name == "grpo":
            sp.add_argument("--mode", default="ensemble", choices=["ensemble", "single", "hit", "oracle"])
        if name == "collapse":
            sp.add_argument("--plot-only", action="store_true")
        if name == "eval":
            sp.add_argument("--split", default="test", choices=["val", "test"])
        if name == "pipeline":
            sp.add_argument("--start", choices=STAGES, default=None)
            sp.add_argument("--stop", choices=STAGES, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    configure_threads(cfg)
    seed_all(cfg.seed)
    log(f"encore {args.command} | config={args.config} ({cfg.name})")
    COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
