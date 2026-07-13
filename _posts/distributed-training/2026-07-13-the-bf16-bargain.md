---
layout: post
title: "The bf16 Bargain: A Numerics Ledger For Mixed Precision"
date: 2026-07-13 09:00:00
description: "Why parameters may be bf16 while the optimizer must stay fp32, and why fp16 needs loss scaling while bf16 doesn't, with a twist where naive all-bf16 training refuses to fail at toy scale and even generalizes better, until a per-parameter audit shows 87% of late-training updates being swallowed and a small-LR run makes the disease visible."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/08/fig-1-float-formats.png
toc:
  sidebar: left
related_posts: false
---

> Traffic runs in low precision and state stays in high precision. That is why parameters may be bf16, why the optimizer must be fp32, and why fp16 needs loss scaling while bf16 doesn't. The experiment has a twist. At toy scale, naive all-bf16 training refuses to break and even generalizes *better*, yet a per-parameter audit shows 87% of late-training updates being swallowed whole, and once the learning rate drops to late-large-model levels, the model without an fp32 master simply stops moving.

## 1. The IOUs this series has accumulated

Every previous post quietly used mixed precision without ever justifying it:

- Post #0's memory ledger charges parameters at 2 bytes but optimizer state at 12. **Why do parameters get away with 2 bytes while optimizer state needs 12?**
- Post #3's ZeRO constant $$K=12$$ hides "fp32 master params + fp32 $$m$$ + fp32 $$v$$". **What is a master copy, and why store the parameters twice?**
- Post #4's FSDP used `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`. **Why does the reduction get its own dtype?**

This post pays off every IOU. The whole design fits in one sentence:

> **A single multiplication may be sloppy, but a long accumulation must be exact.**

## 2. Precision and range are different things

A float = sign bit + exponent bits + mantissa bits. Exponent bits decide **range** (how large/small a value can be), and mantissa bits decide **precision** (how many digits survive). The same 16 bits, split two ways, give two very different temperaments:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/08/fig-1-float-formats.svg" class="img-fluid rounded" zoomable=true %}

- **fp16** (5 exponent + 10 mantissa): slightly better precision (~3.3 decimal digits) but a dangerously narrow range, with max 65504 and min normal 6.1e-5. Backward passes produce gradients that naturally run small (measured in §4, 57% below 6.1e-5), and untreated they underflow toward zero. The V100-era fix is **loss scaling**. Multiply the loss by a large constant $$S$$, let the chain rule lift every gradient by $$S$$ into representable territory, and divide back before the update, with $$S$$ retuned dynamically on every inf/nan.
- **bf16** (8 exponent + 7 mantissa): the same exponent width as fp32, which means an **identical range**. Nothing under- or overflows that fp32 wouldn't, so the entire loss-scaling machinery is deleted. The price is ~2.4 decimal digits. bf16 is literally fp32 with the low 16 mantissa bits cut off, so casting up just pads zeros (**lossless**), while casting down rounds to nearest (**lossy**, and every trap in this post lives in that one direction).

After Ampere, fp16 training essentially died out, and "mixed precision" now means bf16 by default. So who repays the 2.4-digit hole? Not the storage format but **the places where addition happens**.

## 3. The precision map of one training step

Unroll a single step and every tensor's dtype has a reason:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/08/fig-2-mixed-precision-loop.svg" class="img-fluid rounded" zoomable=true %}

The left side is **flow**, meaning activations and in-flight gradients. That is per-token data, huge in volume, consumed immediately, all bf16. The right side is **state**, meaning master parameters, Adam moments, and gradient-accumulation buffers. Those are ledgers built from millions of tiny additions, all fp32. The two sides meet at exactly two cast points. After backward, bf16 gradients are **losslessly upcast** into the fp32 buffer, and after the step, fp32 master params are **rounded down** into a fresh bf16 compute copy.

Three fp32 defense lines, each better hidden than the last:

**Line 1: the Tensor Core's fp32 accumulator (hardware).** A "bf16 × bf16" matmul instruction does not add products in bf16. They enter an **fp32 accumulator**, the thousands of additions in one row-times-column all happen in fp32, and only the writeback rounds to bf16. It must work this way because an $$h=4096$$ inner product sums 4096 terms, and if you accumulate in bf16, big eats small. Once the running sum reaches 100, adding 0.01 does nothing. Here is an extreme you can verify on a CPU. **Sum 4096 copies of 0.01 term by term, and fp32 gives 40.96 while bf16 gives 4.0.** That is off by 10×, and not by gradual drift. Once the sum hits 4.0, every further 0.01 is less than half a ulp and is rounded away entirely, so the sum is stuck forever.

**Line 2: fp32 gradient accumulation (software).** Gradient accumulation (post #2's gas) is another "many small numbers being summed" crime scene. nanotron simply wrote an `FP32GradientAccumulator` (`src/nanotron/optim/gradient_accumulator.py`), where each micro-batch's bf16 gradient is upcast and added into a persistent fp32 buffer the moment it exists. This is also the gap between the ledger's $$16\Psi$$ and $$18\Psi$$ variants. Store gradients in bf16 and you save $$2\Psi$$ and halve all-reduce traffic but accumulate in bf16, or store them in fp32 and pay $$2\Psi$$ more for the numerically safest option. FSDP's `reduce_dtype=fp32` is the compromise, bf16 in memory and fp32 on the wire.

**Line 3: the fp32 master copy (this post's protagonist).** The parameter update $$w \leftarrow w + \Delta$$ is also an addition, and it is the most asymmetric one of all. Mid-to-late training, $$|\Delta|$$ is typically $$10^{-3}$$ to $$10^{-5}$$ of $$|w|$$. bf16 has 7 mantissa bits, and rounding kills everything below half a ulp:

$$
\boxed{\ |\Delta| \;<\; |w|\cdot 2^{-9} \;\approx\; \frac{|w|}{512} \quad\Longrightarrow\quad \mathrm{bf16}(w + \Delta) = \mathrm{bf16}(w)\ }
$$

Updates smaller than $$|w|/512$$ don't get smaller. They become **exactly zero**. We measured it. Start at $$w=1.0$$ and add $$10^{-4}$$ a thousand times, and fp32 reaches 1.1 while **bf16 is still exactly 1.0**. Hence the rule that updates always land on the fp32 master, whose 23 mantissa bits absorb updates down to $$|w|\cdot 2^{-24}$$ and let consecutive small updates accumulate until they surface in the projected bf16 copy, while the bf16 copy is regenerated from the master every step and is never updated in place.

### 3.1 If fp32 is stored anyway, why not compute in it?

The reverse question is natural. The master is already fp32, so why not run forward/backward on it and save the $$2\Psi$$ copy? The economics don't work. The fp32 master is touched **once** per step by an element-wise update, which is bandwidth-bound and takes milliseconds, but the bf16 copy runs **every matmul in forward and backward**, ~99% of the step's FLOPs. Measured on one RTX PRO 6000 for the same $$8192^3$$ GEMM (fig. 4, left), **fp32 gets 52.7 TFLOPS while bf16 gets 262 TFLOPS, a 5.0× gap** (tf32 133.4, fp16 209.5). The trade is to spend $$2\Psi$$ of storage and make 99% of the compute 5× faster, and on top of that halve activation memory and every byte of communication. The fp32 master is deliberately excluded from the hot path, appearing only in the one addition that needs its precision.

## 4. Experiment: three schemes to the finish line, then count the swallowed updates

A char-level GPT (6 layers, width 384, ~10M params, model definition ships with the post, SDPA attention so all three dtypes run), tiny-shakespeare, one GPU, 6000 steps, cosine LR 1e-3 → 1e-4. All three schemes consume **the same batch sequence from the same init seed**, and only the precision recipe changes:

| scheme | params | compute | Adam state | master copy |
|--------|--------|---------|------------|-------------|
| `fp32` | fp32 | fp32 | fp32 | — (params are it) |
| `mixed` | fp32 | autocast bf16 | fp32 | ✓ (the fp32 params) |
| `bf16` | bf16 | bf16 | **bf16** | ✗ (naive "everything bf16") |

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/08/fig-3-three-precisions.svg" class="img-fluid rounded" zoomable=true %}

| | fp32 | mixed | bf16 (no master) |
|---|---|---|---|
| step time | 36.5 ms | 10.2 ms (**3.6×**) | 8.9 ms |
| peak memory | 2.51 GiB | 1.64 GiB (−35%) | 1.27 GiB |
| train loss @6000 (EMA) | 0.078 | 0.079 | 0.079 |
| val loss @6000 | 4.26 | 4.24 | **3.51** |

Three readings, each ruder than the last:

1. **Mixed is bit-for-bit indistinguishable from fp32** while running 3.6× faster on 35% less memory. In panel (a) the two curves sit exactly on top of each other, 0.079 vs 0.078. The mixed-precision promise of "fp32 convergence at bf16 speed" cashes out cleanly at this scale.
2. **Naive all-bf16 does not break.** Its train loss tracks throughout, and its val loss is the *best* of the three, 3.51 vs 4.24. The reason is that 10M parameters memorizing 1.1M characters overfit long before step 6000, val turns upward from ~step 1300, and bf16's rounding noise acts as a regularizer, so it overfits slowest. The textbook says training without a master copy fails, but the toy says it's fine. Who's lying?
3. Panel (c)'s **swallow audit** provides the clue. Inside the mixed run, whose fp32 params are a trustworthy reference, every 200 steps we take the step's true update $$\Delta$$ and simulate bf16 rounding per parameter. The fraction where $$(w_{\rm bf16}+\Delta)$$ rounds back to exactly $$w_{\rm bf16}$$ climbs from ~25% at full LR to **87%** as cosine decay bites. In late training, four updates out of five would land on a bf16 weight and change nothing. The toy shrugs it off only because by then the task is already learned, so those updates carried nothing. **When updates shrink not because learning is done but because the LR schedule must decay while the task is far from learned, which is the everyday situation of large-model training, what gets swallowed is real progress.**

Panel (b) isolates exactly that regime. The same model is trained from scratch at a constant lr = 3e-5, which puts updates around and below the $$|w|/512$$ threshold, precisely the magnitude a decayed large-model LR produces. Mixed and fp32 remain bit-for-bit twins, grinding steadily down to 1.50, but **pure bf16 rides its first 500 steps of large gradients down to 2.57 and then barely moves (2.41 by step 3000)**, a 0.9-nat gap that is still widening. The master copy's value is visible in one glance. In fp32, small updates accumulate until they clear $$|w|/512$$ and surface in the projection. Without a master, each step's small update is rounded away independently, and **accumulation simply never happens**.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/08/fig-4-gemm-and-gradhist.svg" class="img-fluid rounded" zoomable=true %}

The left panel is §3.1's 5× argument. The right panel answers why fp16 needs loss scaling. It shows the distribution of every parameter gradient at step 500, and **57% sit below fp16's minimum normal 6.1e-5**, entering the subnormal zone where precision bleeds away bit by bit (0.05% flush straight to zero). That is the population loss scaling exists to relocate, because ×1024 shifts the whole distribution three decades right, back into normal territory. bf16/fp32's floor is 1.2e-38, twenty-six decades left of this histogram's tail, so nothing needs doing.

> Honest boundary: our "disease made visible" uses a small-LR isolation run, not a natural training trajectory. In real large models (GPT-3-era practice onward, and the fp16 story in the original mixed-precision paper) the damage occurs *on* the natural trajectory in mid-to-late training, because the larger the model and the longer the run, the smaller updates get relative to weights, and the further no-master training falls behind. Note also that our bf16 scheme keeps Adam's $$m, v$$ in bf16 too, and $$v$$, a running mean of squared gradients, is tinier still. Production systems that put parameters in bf16 almost never dare downgrade the state.

## 5. Reading along in real source

**PyTorch autocast**: `torch/amp` keeps an operator list where matmul/conv run bf16, softmax/layer_norm/loss stay fp32, and casts are inserted at op boundaries. You never see the casts in your code, but they are real.

**FSDP2**: `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`, used in post #4. Sharded params (state) stay in fp32, all-gathered compute copies (flow) run in bf16, and the reduce-scatter happens in fp32.

**nanotron**: `FP32GradientAccumulator` in `src/nanotron/optim/gradient_accumulator.py` is the textbook implementation of defense line 2.

**Megatron-LM**: `Float16OptimizerWithFloat16Params` in `megatron/core/optimizer/optimizer.py` does the `main_params` (fp32 master) / `model_params` (bf16) double bookkeeping, plus `--accumulate-allreduce-grads-in-fp32`.

**DeepSpeed ZeRO**: the "optimizer state" that stage 1 shards (post #3) *includes* the fp32 master, which is why ZeRO-1 saves $$(4+4+4)\Psi/N$$ while the $$2\Psi$$ bf16 copy stays replicated.

## 6. Summary, and closing the series

1. Precision and range are different purchases. fp16 spends its 16 bits on mantissa and pays with a narrow range that needs loss scaling as life support, while bf16 spends them on fp32's full range and pays with 2.4 digits and no life support, which is how the bf16 era happened.
2. The whole design fits in one line. **Flow runs bf16 (activations and in-flight gradients, per-token volume) while state stays fp32 (masters, moments, accumulation buffers, per-parameter ledgers)**, stitched by the hardware fp32 accumulator, one lossless upcast, and one per-step reprojection.
3. Why the master copy cannot be skipped is computable, because updates below $$|w|/512$$ round to exactly zero in bf16, and by the end of LR decay that is a measured 87% of updates. At toy scale the disease hides (rounding noise even regularizes), but isolate small updates and the no-master model stalls at 2.41 while mixed grinds to 1.50.
4. Post #0's $$16\Psi/18\Psi$$ now has every line item sourced: $$2\Psi$$ (bf16 params) $$+\,4\Psi$$ (fp32 master) $$+\,4\Psi+4\Psi$$ ($$m$$, $$v$$) $$+\,2\Psi|4\Psi$$ (gradients, depending on accumulation precision).

**This closes the series.** One thread ran through all nine posts. Every design decision in distributed training reduces to a handful of accounts you can check by hand, whether communication volume (post #1's bandwidth table), memory (the $$\Psi$$ ledger), bubbles (grid geometry), or numerics (this post's one-in-512). Next time you meet any parallelism scheme, may your first instinct be to price it out and run the numbers.

---

*Environment: 8× RTX PRO 6000 Blackwell (1 used), PyTorch 2.9.1, CUDA 12.8. Reproduce: `python train_precision.py` (main comparison + swallow audit + gradient histogram), `python train_precision.py --steps 3000 --warmup 0 --lr 3e-5 --min-lr 3e-5 --tag _lowlr` (small-LR mechanism isolation), and `python bench_gemm.py` (GEMM throughput). Plotting and schematic code accompanies the series.*

*All benchmark scripts, schematic generators, plotting code and raw result CSVs for this post live in [assets/blog/code/08-mixed-precision](https://github.com/Dominator-Index/Dominator-Index.github.io/tree/main/assets/blog/code/08-mixed-precision).*

---

## Appendix: The Code That Ran

Every number in this post comes from the scripts below, embedded verbatim. Plotting and schematic code plus the raw result CSVs live in the folder linked above.

<details markdown="1">
<summary><code>train_precision.py</code></summary>

```python
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
```

</details>

<details markdown="1">
<summary><code>bench_gemm.py</code></summary>

```python
"""
GEMM throughput vs precision (part 08 of the Illustrated Distributed Training series):
the same 8192^3 matmul, only the dtype changes, measured in TFLOPS. This is the
ledger behind "why 99% of the compute should run in low precision".

fp32   classic fp32 (TF32 off)
tf32   fp32 inputs, Tensor Core TF32 path (mantissa truncated to 10 bits)
bf16   bf16 inputs, fp32 accumulation (Tensor Core)
fp16   fp16 inputs, fp32 accumulation (Tensor Core)

Usage: python bench_gemm.py --out ../results/gemm.csv
"""

import argparse
import csv
import os

import torch

N = 8192
WARMUP, STEPS = 10, 50
DEVICE = torch.device("cuda", 0)


def bench(dtype, allow_tf32):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    a = torch.randn(N, N, device=DEVICE, dtype=dtype)
    b = torch.randn(N, N, device=DEVICE, dtype=dtype)
    for _ in range(WARMUP):
        a @ b
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(STEPS):
        a @ b
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / STEPS
    tflops = 2 * N**3 / (ms / 1e3) / 1e12
    return ms, tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/gemm.csv")
    args = ap.parse_args()

    rows = []
    for name, dtype, tf32 in [("fp32", torch.float32, False),
                              ("tf32", torch.float32, True),
                              ("bf16", torch.bfloat16, False),
                              ("fp16", torch.float16, False)]:
        ms, tflops = bench(dtype, tf32)
        rows.append([name, round(ms, 3), round(tflops, 1)])
        print(f"{name}: {ms:.2f} ms, {tflops:.1f} TFLOPS", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dtype", "ms", "tflops"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
```

</details>

