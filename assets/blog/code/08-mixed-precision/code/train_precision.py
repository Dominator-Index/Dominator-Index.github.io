"""
Three-precision comparison training (part 08 of the Illustrated Distributed Training
series): same model, same data, same random seeds, only the precision scheme changes.
Measures the claim "mixed precision converges like fp32 at bf16 speed, pure bf16
trains badly".

Three schemes:
  fp32    params and compute all fp32 (baseline)
  mixed   fp32 params (i.e. the master copy) + autocast bf16 compute, the de facto bf16 mixed precision
  bf16    params, momentum, and compute all bf16, no fp32 master copy, the naive "drop the master copy" scheme

Two extra ledgers (both inside the mixed run, where fp32 params are a trustworthy reference):
  1. swallow fraction: every 200 steps, actually round this step's fp32 update delta
     to bf16 and count how many params satisfy (w+delta).bf16 == w.bf16. If params
     were bf16, these updates would be swallowed whole.
  2. gradient histogram: log10|g| distribution over all parameter gradients at step 500,
     compared against the representable lower bounds of fp16/bf16.

Model: char-level GPT (6 layers, width 384, ~10M params, self-contained definition,
SDPA attention so all three dtypes can run, whereas the repo's model.py hard-depends
on flash-attn which only accepts fp16/bf16), tiny-shakespeare, single GPU.
Usage: python train_precision.py --outdir ../results
"""

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ln2 = nn.LayerNorm(n_embd, bias=False)
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + self.fc_proj(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, block, n_layer=6, n_head=6, n_embd=384):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(lambda m: torch.nn.init.normal_(m.weight, 0.0, 0.02)
                   if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, idx, targets):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

BLOCK, BATCH = 256, 64
MAX_ITERS, WARMUP_ITERS = 6000, 100
LR, MIN_LR = 1e-3, 1e-4
EVAL_INTERVAL, EVAL_ITERS = 100, 20
SWALLOW_INTERVAL = 200
GRADHIST_STEP = 500
DEVICE = torch.device("cuda", 0)


def load_data():
    text = open(os.path.join(os.path.dirname(__file__), "input.txt")).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data, gen):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)


def lr_at(it):
    if it < WARMUP_ITERS:
        return LR * (it + 1) / WARMUP_ITERS
    t = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, val, autocast_ctx):
    model.eval()
    gen = torch.Generator().manual_seed(4242)  # all three runs use the same validation batches
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(val, gen)
        with autocast_ctx():
            _, loss = model(x, y)
        losses.append(loss.float().item())
    model.train()
    return sum(losses) / len(losses)


def run(scheme, train, val, vocab, outdir, tag=""):
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = GPT(vocab, BLOCK).to(DEVICE)
    if scheme == "bf16":
        model = model.bfloat16()  # params are bf16, and Adam momentum follows the param dtype so it is bf16 too

    if scheme == "mixed":
        def autocast_ctx():
            return torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    gen = torch.Generator().manual_seed(1337)  # all three runs consume the same stream of training batches

    curve, swallow_rows, hist_rows = [], [], []
    step_times = []
    train_ema = None  # EMA of train loss (0.98), shows stagnation better than val once overfitting starts
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    for it in range(MAX_ITERS):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        x, y = get_batch(train, gen)

        ev0.record()
        with autocast_ctx():
            _, loss = model(x, y)
        loss.backward()

        # ---- swallow ledger (mixed: params are fp32, simulate how many updates bf16 params would swallow) ----
        do_swallow = scheme == "mixed" and (it + 1) % SWALLOW_INTERVAL == 0
        if do_swallow:
            prev = [p.detach().clone() for p in model.parameters()]
        # ---- gradient histogram (mixed, step 500) ----
        if scheme == "mixed" and it + 1 == GRADHIST_STEP:
            g_all = torch.cat([p.grad.detach().float().abs().flatten()
                               for p in model.parameters() if p.grad is not None])
            g_all = g_all[g_all > 0]
            logg = torch.log10(g_all)
            cnt = torch.histc(logg, bins=56, min=-12, max=2)
            edges = torch.linspace(-12, 2, 57)
            for lo, c in zip(edges[:-1].tolist(), cnt.tolist()):
                hist_rows.append([round(lo, 2), int(c)])

        opt.step()
        opt.zero_grad(set_to_none=True)
        ev1.record()
        torch.cuda.synchronize()
        if it >= 20 and not do_swallow:
            step_times.append(ev0.elapsed_time(ev1))
        li = loss.float().item()
        train_ema = li if train_ema is None else 0.98 * train_ema + 0.02 * li

        if do_swallow:
            tot, sw = 0, 0
            with torch.no_grad():
                for p, w0 in zip(model.parameters(), prev):
                    delta = p.detach() - w0
                    w0b = w0.bfloat16()
                    swallowed = (w0b.float() + delta).bfloat16() == w0b
                    sw += swallowed.sum().item()
                    tot += p.numel()
            swallow_rows.append([it + 1, round(sw / tot, 4)])
            print(f"  [swallow] step {it+1}: {sw/tot:.1%}", flush=True)

        if (it + 1) % EVAL_INTERVAL == 0 or it == 0:
            vl = evaluate(model, val, autocast_ctx)
            curve.append([it + 1, round(vl, 4), round(train_ema, 4)])
            print(f"[{scheme}] step {it+1}: val {vl:.4f} train_ema {train_ema:.4f}", flush=True)

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    step_ms = sum(step_times) / len(step_times)
    final_val = curve[-1][1]

    with open(os.path.join(outdir, f"curve_{scheme}{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "val_loss", "train_ema"]); w.writerows(curve)
    if swallow_rows:
        with open(os.path.join(outdir, f"swallow{tag}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step", "swallow_frac"]); w.writerows(swallow_rows)
    if hist_rows:
        with open(os.path.join(outdir, "gradhist.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["log10_bin_lo", "count"]); w.writerows(hist_rows)
    return [scheme, round(step_ms, 2), round(peak_gib, 3), final_val]


def main():
    global MAX_ITERS, WARMUP_ITERS, LR, MIN_LR, GRADHIST_STEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../results")
    ap.add_argument("--steps", type=int, default=MAX_ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--min-lr", type=float, default=MIN_LR)
    ap.add_argument("--tag", default="")  # use --tag _lowlr for the small-LR mechanism experiment
    args = ap.parse_args()
    MAX_ITERS, WARMUP_ITERS, LR, MIN_LR = args.steps, args.warmup, args.lr, args.min_lr
    if args.tag:
        GRADHIST_STEP = -1  # record the histogram only in the main experiment
    os.makedirs(args.outdir, exist_ok=True)

    train, val, vocab = load_data()
    print(f"vocab {vocab}, train {len(train)/1e6:.2f}M chars", flush=True)

    summary = []
    for scheme in ["fp32", "mixed", "bf16"]:
        print(f"=== {scheme} ===", flush=True)
        summary.append(run(scheme, train, val, vocab, args.outdir, args.tag))
    with open(os.path.join(args.outdir, f"summary{args.tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "step_ms", "peak_mem_gib", "final_val_loss"])
        w.writerows(summary)
    for row in summary:
        print("SUMMARY:", row)


if __name__ == "__main__":
    main()
