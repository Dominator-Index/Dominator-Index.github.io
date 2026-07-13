---
layout: post
title: "The Geometry Of Pipeline Bubbles"
date: 2026-07-12 10:00:00
description: "The (p−1)/(m+p−1) bubble derived by counting grid cells, 1F1B's same-bubble-1/m-memory rearrangement, and a 60-line hand-written GPipe that nails the formula at m=1 and then hits the wall the formula doesn't mention."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/07/fig-1-schedules.png
toc:
  sidebar: left
related_posts: false
---

> The pipeline bubble is decided by schedule geometry. That geometry says where $$(p-1)/(m+p-1)$$ comes from, how 1F1B improves on it, and where it fails when measured. A hand-written GPipe (~60 lines) on 4 GPUs reproduces the formula to 0.9% at $$m{=}1$$, and then runs into another wall the formula never wrote down.

## 1. PP is the first parallelism with division of labor

Every dimension so far, DP/TP/SP/CP, has all GPUs doing the **same kind** of work (same layers, different data or slices). Pipeline parallelism (PP) is the first to give different GPUs **different layers**. The model is cut by depth into $$p$$ stages, and activations cross stage boundaries via p2p send/recv.

Its communication is the cheapest in the whole series. Per boundary, per microbatch, it sends one $$[b_{mb} \times h]$$ activation forward and its gradient backward, which is **point-to-point, not collective, independent of parameter count**. That is why post #0's iron rule puts pp on the outermost, cross-node axis. It is the least sensitive to slow interconnects.

But cheap communication buys a new disease. **Division of labor creates waiting.** Stage 1 has nothing to do until stage 0 finishes, and in backward, stage 0 waits for everyone. That structural idling is the bubble.

## 2. Bubble geometry: a derivation by counting grid cells

Discretize time into slots (one slot = one microbatch through one stage). Split the batch into $$m$$ microbatches, and GPipe runs all forwards, then all backwards:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/07/fig-1-schedules.svg" class="img-fluid rounded" zoomable=true %}

Count cells in panel (b) at $$p{=}4, m{=}4$$. The first microbatch takes $$p$$ slots to fill the pipe, then one drains per slot, so the forward phase spans $$m+p-1$$, and backward is symmetric. Each stage does $$2m$$ slots of real work in a $$2(m+p-1)$$-slot span:

$$
\boxed{\ \text{bubble fraction} \;=\; \frac{2(m+p-1) - 2m}{2(m+p-1)} \;=\; \frac{p-1}{m+p-1}\ }
$$

Three immediate corollaries:

1. **$$m{=}1$$ is naive model parallelism** (panel (a)), with bubble $$(p-1)/p$$, so 75% waste on 4 GPUs and 87.5% on 8. Cut the model without cutting the batch, and you bought $$p$$ GPUs to run one at a time.
2. **Only $$m$$ amortizes the bubble**, since $$m = 4p$$ → ≈19% and $$m = 8p$$ → ≈10%. This is why PP is naturally married to gradient accumulation (post #2), because gas *is* a ready-made $$m$$.
3. **The bubble is speed-independent**, because it is the *shape* of the schedule, not the quality of the implementation. Faster kernels and faster links change nothing. Only a different schedule does.

## 3. 1F1B: same bubble, $$1/m$$ the memory

GPipe hides a bill. All $$m$$ microbatches' activations must survive until their backward, which means **activation memory O(m)**, right after §2 told us to make $$m$$ large. 1F1B (one-forward-one-backward, PipeDream-Flush) rearranges the schedule. After a warm-up of $$p-s$$ forwards, each stage alternates one forward with one available backward (panel (c), slot positions derived from the dependency constraints, checkable):

- **The total span is identical to GPipe**, so the bubble formula is unchanged. Count the grey cells in (b) and (c) and they come out equal.
- But at most $$p$$ microbatches are in flight at any moment, which means **activation memory O(p), independent of m**. Now $$m$$ can grow safely.

> The wider map never leaves this grid. Interleaved 1F1B (each GPU owns several non-adjacent stage chunks, Megatron's virtual pipeline stages) divides the bubble by the chunk count. Zero-bubble schedules (ZB-H1) split backward into input-grad and weight-grad halves to fill the holes, approaching zero bubble in theory. DualPipe (DeepSeek-V3) runs the pipe bidirectionally. All of them are finer Tetris on the same grid.

## 4. Experiment: a 60-line GPipe puts the formula on the scale

The hand-written GPipe (ships with the post) runs 4 stages × 6 Linear+GeLU layers at width 4096, total batch fixed at 8192 rows, $$m$$ swept 1→32. `isend/recv` connects the stages and backward returns `ins[i].grad` upstream, a direct transcription of §2's grid.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/07/fig-2-bubble-measured.svg" class="img-fluid rounded" zoomable=true %}

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| step (ms) | 91.2 | 61.9 | 52.0 | **48.6** | 58.8 | 86.2 |
| bubble, theory | 75% | 60% | 43% | 27% | 16% | 9% |
| idle, measured | 75.9% | 71% | 53% | 53% | 54% | 61% |

Three readings:

1. **At $$m{=}1$$ the formula is exact** (75.9% vs 75%, off by 0.9%), so "naive model parallelism wastes $$(p-1)/p$$" is not rhetoric but a measurable fact.
2. **As $$m$$ grows, measurement departs from theory and floors at ~53%**, then bounces back. What the formula never wrote down shows up. Every microbatch pays a p2p hop latency and kernel-launch overhead, and, subtler, **GEMM efficiency itself decays**. Shrinking the microbatch from 8192 to 256 rows leaves total FLOPs unchanged but drops GPU execution efficiency by ~35% ($$m \cdot t_{\text{slot}}$$ climbs from 21.9 ms to 33.6 ms). The finer you slice, the less efficient each slice.
3. So we get a third U-curve for this series, after DDP's bucket size and FSDP's reshard knob. **The bubble wants $$m$$ large, per-slice efficiency wants $$m$$ small, and the optimum sits between** (here $$m{=}8$$). Production systems take both ends. They raise $$m$$ *and* switch to interleaved/zero-bubble schedules to shrink the bubble's coefficient, rather than slicing microbatches ever finer.

> Honest boundary: our toy is pure GPipe with equal slots and no comm/compute overlap. Production PP overlaps send/recv with compute, backward is ≈2× forward, and stages are rarely perfectly balanced (embedding/lm_head imbalance is a real tuning pain). All of that moves the U-curve's bottom, but none of it dissolves the tension between bubble geometry and per-slice overhead.

## 5. Reading along in real source

**PyTorch** ships `ScheduleGPipe` / `Schedule1F1B` in `torch.distributed.pipelining`, the schedule as code, cell-for-cell against panels (b)/(c).

**nanotron**: `AllForwardAllBackwardPipelineEngine` (= GPipe) and `OneForwardOneBackwardPipelineEngine` (= 1F1B) in `src/nanotron/parallel/pipeline_parallel/engine.py`, with p2p in `p2p.py`.

**Megatron-LM** keeps the schedules in `megatron/core/pipeline_parallel/schedules.py`, including interleaved 1F1B.

**DeepSpeed** compiles schedules into instruction streams (`LoadMicroBatch / ForwardPass / SendActivation / ...`) in `deepspeed/runtime/pipe/schedule.py`, and it is well worth a read.

## 6. Summary

1. PP buys the cheapest communication (p2p boundary activations, parameter-independent) at the price of a structural disease. The bubble fraction $$\frac{p-1}{m+p-1}$$ is pure geometry, independent of implementation.
2. $$m{=}1$$ is naive model parallelism with $$(p-1)/p$$ waste, verified to 0.9%. The only free amortizer is $$m$$, which is why PP is naturally married to gradient accumulation.
3. 1F1B keeps the same bubble while taking activation memory from O(m) to O(p). It is the textbook case of rearranging a schedule to change peak resources without changing the makespan.
4. The measurement adds the other half. Per-microbatch overheads (hop latency + launches + small-GEMM inefficiency) give $$m$$ an optimum, so the bubble formula is a necessary map, not the full terrain.

**Next comes mixed precision, the numerics ledger of the bf16 era.** Every post so far quietly used bf16/fp32 mixing rules (FSDP's `MixedPrecisionPolicy`, ZeRO's fp32 masters, fp32 NCCL reductions) without ever justifying them. Why may parameters be bf16 while the optimizer must stay fp32, and why does fp16 need loss scaling while bf16 doesn't? One post pays off every precision IOU in the 16Ψ ledger.

---

*Environment: 8× RTX PRO 6000 Blackwell (4 used), PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node=4 bench_gpipe.py`. Plotting and schematic code accompanies the series, and the 1F1B slots are derived from dependency constraints, so verify them.*
