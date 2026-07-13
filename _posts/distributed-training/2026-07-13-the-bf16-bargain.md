---
layout: post
title: "The bf16 Bargain: A Numerics Ledger For Mixed Precision"
date: 2026-07-13 09:00:00
description: "Why parameters may be bf16 while the optimizer must stay fp32, and why fp16 needs loss scaling while bf16 doesn't — with a twist: naive all-bf16 training refuses to fail at toy scale (it even generalizes better), until a per-parameter audit shows 87% of late-training updates being swallowed, and a small-LR run makes the disease visible."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/08/fig-1-float-formats.png
toc:
  sidebar: left
related_posts: false
---

> Part 8 of **An Overview of Distributed Learning** — the closing post of the series. One post, one idea: **traffic runs in low precision, state stays in high precision — why parameters may be bf16, why the optimizer must be fp32, and why fp16 needs loss scaling while bf16 doesn't.** The experiment has a twist: at toy scale, naive all-bf16 training refuses to break — it even generalizes *better* — yet a per-parameter audit shows 87% of late-training updates being swallowed whole, and once the learning rate drops to late-large-model levels, the model without an fp32 master simply stops moving.

## 1. The IOUs this series has accumulated

Every previous post quietly used mixed precision without ever justifying it:

- Post #0's memory ledger charges parameters at 2 bytes but optimizer state at 12 — **why do parameters get away with 2 bytes while optimizer state needs 12?**
- Post #3's ZeRO constant $$K=12$$ hides "fp32 master params + fp32 $$m$$ + fp32 $$v$$" — **what is a master copy, and why store the parameters twice?**
- Post #4's FSDP `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` — **why does the reduction get its own dtype?**

This post pays off every IOU. The whole design fits in one sentence:

> **A single multiplication may be sloppy; a long accumulation must be exact.**

## 2. Precision and range are different things

A float = sign bit + exponent bits + mantissa bits. Exponent bits decide **range** (how large/small a value can be); mantissa bits decide **precision** (how many digits survive). The same 16 bits, split two ways, give two very different temperaments:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/08/fig-1-float-formats.svg" class="img-fluid rounded" zoomable=true %}

- **fp16** (5 exponent + 10 mantissa): slightly better precision (~3.3 decimal digits) but a dangerously narrow range — max 65504, min normal 6.1e-5. Backward passes produce gradients that naturally run small (measured in §4: 57% below 6.1e-5), and untreated they underflow toward zero. The V100-era fix is **loss scaling**: multiply the loss by a large constant $$S$$, let the chain rule lift every gradient by $$S$$ into representable territory, divide back before the update — with $$S$$ retuned dynamically on every inf/nan.
- **bf16** (8 exponent + 7 mantissa): the same exponent width as fp32 — **identical range**. Nothing under- or overflows that fp32 wouldn't, so the entire loss-scaling machinery is deleted. The price: ~2.4 decimal digits. bf16 is literally fp32 with the low 16 mantissa bits cut off — casting up just pads zeros (**lossless**); casting down rounds to nearest (**lossy** — every trap in this post lives in that one direction).

After Ampere, fp16 training essentially died out; "mixed precision" now means bf16 by default. So who repays the 2.4-digit hole? Not the storage format — **the places where addition happens**. Read on.

## 3. The precision map of one training step

Unroll a single step and every tensor's dtype has a reason:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/08/fig-2-mixed-precision-loop.svg" class="img-fluid rounded" zoomable=true %}

The left side is **flow**: activations and in-flight gradients — per-token data, huge volume, consumed immediately, all bf16. The right side is **state**: master parameters, Adam moments, gradient-accumulation buffers — ledgers built from millions of tiny additions, all fp32. The two sides meet at exactly two cast points: after backward, bf16 gradients are **losslessly upcast** into the fp32 buffer; after the step, fp32 master params are **rounded down** into a fresh bf16 compute copy.

Three fp32 defense lines, each better hidden than the last:

**Line 1: the Tensor Core's fp32 accumulator (hardware).** A "bf16 × bf16" matmul instruction does not add products in bf16 — they enter an **fp32 accumulator**, the thousands of additions in one row-times-column all happen in fp32, and only the writeback rounds to bf16. Why it must: an $$h=4096$$ inner product sums 4096 terms; accumulate in bf16 and "big eats small" — once the running sum reaches 100, adding 0.01 does nothing. An extreme you can verify on a CPU: **sum 4096 copies of 0.01 term by term — fp32 gives 40.96, bf16 gives 4.0.** Off by 10×, and not by gradual drift: once the sum hits 4.0, every further 0.01 is less than half a ulp and is rounded away entirely. The sum is stuck forever.

**Line 2: fp32 gradient accumulation (software).** Gradient accumulation (post #2's gas) is another "many small numbers being summed" crime scene. nanotron simply wrote an `FP32GradientAccumulator` (`src/nanotron/optim/gradient_accumulator.py`): each micro-batch's bf16 gradient is upcast and added into a persistent fp32 buffer the moment it exists. This is also the gap between the ledger's $$16\Psi$$ and $$18\Psi$$ variants: store gradients in bf16 (save $$2\Psi$$ and halve all-reduce traffic, but accumulate in bf16) or in fp32 (pay $$2\Psi$$ more, numerically safest). FSDP's `reduce_dtype=fp32` is the compromise: bf16 in memory, fp32 on the wire.

**Line 3: the fp32 master copy (this post's protagonist).** The parameter update $$w \leftarrow w + \Delta$$ is also an addition — the most asymmetric one of all. Mid-to-late training, $$|\Delta|$$ is typically $$10^{-3}$$ to $$10^{-5}$$ of $$|w|$$. bf16 has 7 mantissa bits, and rounding kills everything below half a ulp:

$$
\boxed{\ |\Delta| \;<\; |w|\cdot 2^{-9} \;\approx\; \frac{|w|}{512} \quad\Longrightarrow\quad \mathrm{bf16}(w + \Delta) = \mathrm{bf16}(w)\ }
$$

Updates smaller than $$|w|/512$$ don't get smaller — they become **exactly zero**. Measured: start at $$w=1.0$$ and add $$10^{-4}$$ a thousand times; fp32 reaches 1.1, **bf16 is still exactly 1.0**. Hence the rule: updates always land on the fp32 master (23 mantissa bits absorb updates down to $$|w|\cdot 2^{-24}$$; consecutive small updates accumulate there and surface in the projected bf16 copy once large enough), while the bf16 copy is regenerated from the master every step and is never updated in place.

### 3.1 If fp32 is stored anyway, why not compute in it?

The reverse question: the master is already fp32 — run forward/backward on it and save the $$2\Psi$$ copy? The economics don't work. The fp32 master is touched **once** per step by an element-wise update (bandwidth-bound, milliseconds); the bf16 copy runs **every matmul in forward and backward** — ~99% of the step's FLOPs. Measured on one RTX PRO 6000, the same $$8192^3$$ GEMM (fig. 4, left): **fp32 52.7 TFLOPS → bf16 262 TFLOPS, a 5.0× gap** (tf32 133.4, fp16 209.5). The trade is: spend $$2\Psi$$ of storage, make 99% of the compute 5× faster — plus halve activation memory and every byte of communication. The fp32 master is deliberately excluded from the hot path, appearing only in the one addition that needs its precision.

## 4. Experiment: three schemes to the finish line, then count the swallowed updates

A char-level GPT (6 layers, width 384, ~10M params — model definition ships with the post, SDPA attention so all three dtypes run), tiny-shakespeare, one GPU, 6000 steps, cosine LR 1e-3 → 1e-4. All three schemes consume **the same batch sequence from the same init seed**; only the precision recipe changes:

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

1. **Mixed is bit-for-bit indistinguishable from fp32** (panel (a): the two curves sit exactly on top of each other, 0.079 vs 0.078) while running 3.6× faster on 35% less memory — the mixed-precision promise, "fp32 convergence at bf16 speed", cashes out cleanly at this scale.
2. **Naive all-bf16 does not break.** Its train loss tracks throughout, and its val loss is the *best* of the three (3.51 vs 4.24 — 10M parameters memorizing 1.1M characters overfit long before step 6000; val turns upward from ~step 1300, and bf16's rounding noise acts as a regularizer, overfitting slowest). The textbook says training without a master copy fails; the toy says it's fine. Who's lying?
3. Panel (c)'s **swallow audit** provides the clue: inside the mixed run (whose fp32 params are a trustworthy reference), every 200 steps we take the step's true update $$\Delta$$ and simulate bf16 rounding per parameter — the fraction where $$(w_{\rm bf16}+\Delta)$$ rounds back to exactly $$w_{\rm bf16}$$ climbs from ~25% at full LR to **87%** as cosine decay bites. In late training, four updates out of five would land on a bf16 weight and change nothing. The toy shrugs it off only because by then the task is already learned — those updates carried nothing. **When updates shrink not because learning is done but because the LR schedule must decay while the task is far from learned — the everyday situation of large-model training — what gets swallowed is real progress.**

Panel (b) isolates exactly that regime: the same model trained from scratch at a constant lr = 3e-5 (updates around and below the $$|w|/512$$ threshold — precisely the magnitude a decayed large-model LR produces). Mixed and fp32 remain bit-for-bit twins, grinding steadily down to 1.50; **pure bf16 rides its first 500 steps of large gradients down to 2.57 and then barely moves (2.41 by step 3000)** — a 0.9-nat gap, still widening. The master copy's value is visible in one glance: in fp32, small updates accumulate until they clear $$|w|/512$$ and surface in the projection; without a master, each step's small update is rounded away independently, and **accumulation simply never happens**.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/08/fig-4-gemm-and-gradhist.svg" class="img-fluid rounded" zoomable=true %}

The left panel is §3.1's 5× argument. The right panel answers "why does fp16 need loss scaling": the distribution of every parameter gradient at step 500 — **57% sit below fp16's minimum normal 6.1e-5** (entering the subnormal zone where precision bleeds away bit by bit; 0.05% flush straight to zero). That is the population loss scaling exists to relocate: ×1024 shifts the whole distribution three decades right, back into normal territory. bf16/fp32's floor is 1.2e-38 — twenty-six decades left of this histogram's tail; nothing needs doing.

> Honest boundary: the gap between toy and production must be stated. Our "disease made visible" uses a small-LR isolation run, not a natural training trajectory; in real large models (GPT-3-era practice onward, and the fp16 story in the original mixed-precision paper) the damage occurs *on* the natural trajectory in mid-to-late training — the larger the model and the longer the run, the smaller updates get relative to weights, and the further no-master training falls behind. Note also that our bf16 scheme keeps Adam's $$m, v$$ in bf16 too ($$v$$, a running mean of squared gradients, is tinier still); production systems that put parameters in bf16 almost never dare downgrade the state.

## 5. Reading along in real source

**PyTorch autocast** — `torch/amp` keeps an operator list: matmul/conv run bf16, softmax/layer_norm/loss stay fp32, casts inserted at op boundaries. You never see the casts in your code, but they are real.

**FSDP2** — `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`, used in post #4. Sharded params (state) in fp32, all-gathered compute copies (flow) in bf16, reduce-scatter in fp32.

**nanotron** — `FP32GradientAccumulator` in `src/nanotron/optim/gradient_accumulator.py`: the textbook implementation of defense line 2.

**Megatron-LM** — `Float16OptimizerWithFloat16Params` in `megatron/core/optimizer/optimizer.py`: the `main_params` (fp32 master) / `model_params` (bf16) double bookkeeping, plus `--accumulate-allreduce-grads-in-fp32`.

**DeepSpeed ZeRO** — the "optimizer state" that stage 1 shards (post #3) *includes* the fp32 master — which is why ZeRO-1 saves $$(4+4+4)\Psi/N$$ while the $$2\Psi$$ bf16 copy stays replicated.

## 6. Summary — and closing the series

1. Precision and range are different purchases: fp16 spends its 16 bits on mantissa and pays with a narrow range (loss scaling required as life support); bf16 spends them on fp32's full range and pays with 2.4 digits (no support needed) — that is how the bf16 era happened.
2. The whole design in one line: **flow runs bf16 (activations, in-flight gradients — per-token volume), state stays fp32 (masters, moments, accumulation buffers — per-parameter ledgers)**, stitched by the hardware fp32 accumulator, one lossless upcast, and one per-step reprojection.
3. Why the master copy cannot be skipped is computable: updates below $$|w|/512$$ round to exactly zero in bf16 — 87% of updates by the end of LR decay (measured). At toy scale the disease hides (rounding noise even regularizes); isolate small updates and the no-master model stalls at 2.41 while mixed grinds to 1.50.
4. Post #0's $$16\Psi/18\Psi$$ now has every line item sourced: $$2\Psi$$ (bf16 params) $$+\,4\Psi$$ (fp32 master) $$+\,4\Psi+4\Psi$$ ($$m$$, $$v$$) $$+\,2\Psi|4\Psi$$ (gradients, depending on accumulation precision).

**This closes the series.** One thread ran through all nine posts: every design decision in distributed training reduces to a handful of accounts you can check by hand — communication volume (post #1's bandwidth table), memory (the $$\Psi$$ ledger), bubbles (grid geometry), numerics (this post's one-in-512). Next time you meet any parallelism scheme, may your first instinct be: *let me price this out and run the numbers.*

---

*Environment: 8× RTX PRO 6000 Blackwell (1 used), PyTorch 2.9.1, CUDA 12.8. Reproduce: `python train_precision.py` (main comparison + swallow audit + gradient histogram); `python train_precision.py --steps 3000 --warmup 0 --lr 3e-5 --min-lr 3e-5 --tag _lowlr` (small-LR mechanism isolation); `python bench_gemm.py` (GEMM throughput); plotting and schematic code accompanies the series.*
