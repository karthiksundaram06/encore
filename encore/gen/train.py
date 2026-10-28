from encore.data.store import Store
from encore.eval.evaluate import evaluate
from encore.eval.recommenders import GeneratorRec
from encore.gen.model import build_generator, constrained_item_nll, history_tokens
from encore.sid.trie import SIDTrie
from encore.train_loop import fit
from encore.utils import RunPaths, count_params, get_device, log, save_json


def train_sft(cfg):
    paths = RunPaths(cfg)
    store = Store(paths.data)
    device = get_device(cfg.device)
    trie = SIDTrie.load(paths.sid).to(device)
    g = cfg.generator
    model = build_generator(cfg, trie).to(device)
    log(f"SID generator: {count_params(model) / 1e6:.2f}M params, vocab {trie.vocab_size}, "
        f"{trie.n_levels} tokens/item, context {1 + g.history * trie.n_levels} tokens")
    val = store.eval_set("val").subset(g.eval_users, cfg.seed)

    def loss_fn(b):
        H = g.history
        items, fb, complete = b["item"][:, -H:], b["fb"][:, -H:], b["complete"][:, -H:]
        tok, fbt = history_tokens(trie, items, fb, H)
        logits, _, _ = model(tok[:, :-1], fbt[:, :-1])
        loss, _ = constrained_item_nll(logits, trie, items, complete > 0)
        return loss

    def eval_fn(m):
        rec = GeneratorRec(m, trie, g.history, g.eval_beam, device, amp=cfg.amp,
                           cache_gb=cfg.eval.get("beam_cache_gb", 3.0))
        res, _ = evaluate(rec, val, store, [10], batch_size=64)
        return res["ndcg@10"]

    best = fit(model, loss_fn, store, g, g.history, device, cfg.amp, eval_fn,
               paths.ckpt / "sft.pt", paths.logs / "sft.jsonl", cfg.seed, warmup=g.warmup_steps)
    save_json({"val_ndcg@10": best}, paths.reports / "sft_val.json")
    log(f"SFT done: best val ndcg@10={best:.4f}")
    return best
