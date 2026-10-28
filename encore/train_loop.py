import math

import numpy as np
import torch

from encore.data.store import to_torch
from encore.data.windows import iterate, n_batches, train_windows
from encore.utils import JsonlLogger, autocast, cosine_schedule, log, make_optimizer


def save_checkpoint(model, path, **extra):
    torch.save({"hparams": model.hparams, "state_dict": model.state_dict(), **extra}, path)


def fit(model, loss_fn, store, mc, H, device, amp, eval_fn, ckpt_path, log_path, seed,
        users=None, maximize=True, max_split=0, warmup=None, extra=None):
    rng = np.random.default_rng(seed)
    starts, ends = train_windows(store, H, rng, users=users, max_split=max_split)
    steps_per_epoch = n_batches(len(starts), mc.batch_size)
    total = steps_per_epoch * mc.epochs
    opt = make_optimizer(model, mc.lr, mc.weight_decay)
    sched = cosine_schedule(opt, warmup if warmup is not None else max(1, total // 20), total)
    logger = JsonlLogger(log_path)
    best, best_epoch, bad = -math.inf, -1, 0
    patience = mc.get("patience", 10**9)
    step = 0
    for epoch in range(mc.epochs):
        model.train()
        if epoch > 0:
            starts, ends = train_windows(store, H, rng, users=users, max_split=max_split)
        losses = []
        for batch in iterate(store, starts, ends, H, mc.batch_size, rng):
            b = to_torch(batch, device)
            with autocast(device, amp):
                loss = loss_fn(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                log(f"  step {step}/{total} loss={np.mean(losses[-200:]):.4f}")
        model.eval()
        score = eval_fn(model)
        signed = score if maximize else -score
        row = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)), "val": float(score)}
        logger.write(row)
        log(f"epoch {epoch + 1}/{mc.epochs} train_loss={row['train_loss']:.4f} val={score:.4f}")
        if signed > best:
            best, best_epoch, bad = signed, epoch, 0
            save_checkpoint(model, ckpt_path, epoch=epoch, val=float(score), **(extra or {}))
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop at epoch {epoch + 1} (best epoch {best_epoch + 1})")
                break
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return best if maximize else -best
