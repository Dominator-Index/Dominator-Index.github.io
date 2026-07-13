---
layout: post
title: "The Geometry Of Pipeline Bubbles"
date: 2026-07-12 10:00:00
description: "The (p−1)/(m+p−1) bubble derived by counting grid cells, 1F1B's same-bubble-1/m-memory rearrangement, and a 60-line hand-written GPipe that nails the formula at m=1 — then hits the wall the formula doesn't mention."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/07/fig-1-schedules.png
toc:
  sidebar: left
related_posts: false
---

> Part 7 of **An Overview of Distributed Learning**. One post, one idea: **the pipeline bubble is decided by schedule geometry — where $$(p-1)/(m+p-1)$$ comes from, how 1F1B improves on it, and where it fails when measured.** Experiment: a hand-written GPipe (~60 lines) on 4 GPUs that reproduces the formula to 0.9% at $$m{=}1$$ — and then runs into another wall the formula never wrote down.

## 1. PP is the first parallelism with division of labor

Every dimension so far — DP/TP/SP/CP — has all GPUs doing the **same kind** of work (same layers, different data or slices). Pipeline parallelism (PP) is the first to give different GPUs **different layers**: the model is cut by depth into $$p$$ stages, and activations cross stage boundaries via p2p send/recv.

Its communication is the cheapest in the whole series: per boundary, per microbatch, one $$[b_{mb} \times h]$$ activation forward and its gradient backward — **point-to-point, not collective, independent of parameter count**. That is why post #0's iron rule puts pp on the outermost, cross-node axis: it is the least sensitive to slow interconnects.

But cheap communication buys a new disease: **division of labor creates waiting.** Stage 1 has nothing to do until stage 0 finishes; in backward, stage 0 waits for everyone. That structural idling is the bubble.

## 2. Bubble geometry: a derivation by counting grid cells

Discretize time into slots (one slot = one microbatch through one stage). Split the batch into $$m$$ microbatches; GPipe = all forwards, then all backwards:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/07/fig-1-schedules.svg" class="img-fluid rounded" zoomable=true %}

Count cells (panel (b), $$p{=}4, m{=}4$$): the first microbatch takes $$p$$ slots to fill the pipe, then one drains per slot — the forward phase spans $$m+p-1$$; backward is symmetric. Each stage does $$2m$$ slots of real work in a $$2(m+p-1)$$-slot span:

$$
\boxed{\ \text{bubble fraction} \;=\; \frac{2(m+p-1) - 2m}{2(m+p-1)} \;=\; \frac{p-1}{m+p-1}\ }
$$

Three immediate corollaries:

1. **$$m{=}1$$ is naive model parallelism** (panel (a)): bubble $$(p-1)/p$$ — 75% waste on 4 GPUs, 87.5% on 8. Cut the model without cutting the batch, and you bought $$p$$ GPUs to run one at a time.
2. **Only $$m$$ amortizes the bubble**: $$m = 4p$$ → ≈19%, $$m = 8p$$ → ≈10%. This is why PP is naturally married to gradient accumulation (post #2) — gas *is* a ready-made $$m$$.
3. **The bubble is speed-independent**: it is the *shape* of the schedule, not the quality of the implementation. Faster kernels and faster links change nothing — only a different schedule does.

## 3. 1F1B: same bubble, $$1/m$$ the memory

GPipe hides a bill: all $$m$$ microbatches' activations must survive until their backward — **activation memory O(m)**, right after §2 told us to make $$m$$ large. 1F1B (one-forward-one-backward, PipeDream-Flush) rearranges: after a warm-up of $$p-s$$ forwards, each stage alternates one forward with one available backward (panel (c) — slot positions derived from the dependency constraints, checkable):

- **Total span identical to GPipe** — the bubble formula is unchanged (count the grey cells in (b) and (c): equal).
- But at most $$p$$ microbatches are in flight at any moment — **activation memory O(p), independent of m**. Now $$m$$ can grow safely.

> The wider map, without leaving this grid: interleaved 1F1B (each GPU owns several non-adjacent stage chunks; Megatron's virtual pipeline stages) divides the bubble by the chunk count; zero-bubble schedules (ZB-H1) split backward into input-grad and weight-grad halves to fill the holes, approaching zero bubble in theory; DualPipe (DeepSeek-V3) runs the pipe bidirectionally. All of them are finer Tetris on the same grid.

## 4. Experiment: a 60-line GPipe puts the formula on the scale

Hand-written GPipe (ships with the post): 4 stages × 6 Linear+GeLU layers at width 4096, total batch fixed at 8192 rows, $$m$$ swept 1→32; `isend/recv` between stages, backward returns `ins[i].grad` upstream — a direct transcription of §2's grid.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/07/fig-2-bubble-measured.svg" class="img-fluid rounded" zoomable=true %}

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| step (ms) | 91.2 | 61.9 | 52.0 | **48.6** | 58.8 | 86.2 |
| bubble, theory | 75% | 60% | 43% | 27% | 16% | 9% |
| idle, measured | 75.9% | 71% | 53% | 53% | 54% | 61% |

Three readings:

1. **At $$m{=}1$$ the formula is exact** (75.9% vs 75%, off by 0.9%) — "naive model parallelism wastes $$(p-1)/p$$" is not rhetoric but a measurable fact.
2. **As $$m$$ grows, measurement departs from theory and floors at ~53%**, then bounces back. What the formula never wrote down shows up: every microbatch pays a p2p hop latency and kernel-launch overhead, and — subtler — **GEMM efficiency itself decays**: shrinking the microbatch from 8192 to 256 rows leaves total FLOPs unchanged but drops GPU execution efficiency by ~35% ($$m \cdot t_{\text{slot}}$$ climbs from 21.9 ms to 33.6 ms). The finer you slice, the less efficient each slice.
3. So, a third U-curve for this series (after DDP's bucket size and FSDP's reshard knob): **the bubble wants $$m$$ large, per-slice efficiency wants $$m$$ small; the optimum sits between** (here $$m{=}8$$). Production systems take both ends: raise $$m$$ *and* switch to interleaved/zero-bubble schedules to shrink the bubble's coefficient — rather than slicing microbatches ever finer.

> Honest boundary: our toy is pure GPipe with equal slots and no comm/compute overlap; production PP overlaps send/recv with compute, backward is ≈2× forward, and stages are rarely perfectly balanced (embedding/lm_head imbalance is a real tuning pain). All of that moves the U-curve's bottom; none of it dissolves the tension between bubble geometry and per-slice overhead.

## 5. Reading along in real source

**PyTorch** — `torch.distributed.pipelining`: `ScheduleGPipe` / `Schedule1F1B`, the schedule as code, cell-for-cell against panels (b)/(c).

**nanotron** — `AllForwardAllBackwardPipelineEngine` (= GPipe) and `OneForwardOneBackwardPipelineEngine` (= 1F1B) in `src/nanotron/parallel/pipeline_parallel/engine.py`, with p2p in `p2p.py`.

**Megatron-LM** — `megatron/core/pipeline_parallel/schedules.py`, including interleaved 1F1B.

**DeepSpeed** — `deepspeed/runtime/pipe/schedule.py`: schedules compiled into instruction streams (`LoadMicroBatch / ForwardPass / SendActivation / ...`) — well worth a read.

## 6. Summary

1. PP buys the cheapest communication (p2p boundary activations, parameter-independent) at the price of a structural disease: the bubble, $$\frac{p-1}{m+p-1}$$ — pure geometry, implementation-independent.
2. $$m{=}1$$ = naive model parallelism = $$(p-1)/p$$ waste, verified to 0.9%; the only free amortizer is $$m$$ (hence PP ⋈ gradient accumulation).
3. 1F1B: same bubble, activation memory O(m) → O(p) — the textbook case of "rearranging a schedule changes peak resources without changing the makespan".
4. The measured other half: per-microbatch overheads (hop latency + launches + small-GEMM inefficiency) give $$m$$ an optimum — the bubble formula is a necessary map, not the full terrain.

**Next: Mixed Precision — the numerics ledger of the bf16 era.** Every post so far quietly used bf16/fp32 mixing rules (FSDP's `MixedPrecisionPolicy`, ZeRO's fp32 masters, fp32 NCCL reductions) without ever justifying them: why parameters may be bf16 while the optimizer must stay fp32, and why fp16 needs loss scaling while bf16 doesn't — one post to pay off every precision IOU in the 16Ψ ledger.

---

*Environment: 8× RTX PRO 6000 Blackwell (4 used), PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node=4 bench_gpipe.py`; plotting and schematic code accompanies the series (the 1F1B slots are derived from dependency constraints — verify them).*
