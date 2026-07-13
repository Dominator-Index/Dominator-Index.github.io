---
layout: post
title: "Distributed Training, Illustrated #6 — Sequence & Context Parallelism: Two Ways to Cut the Sequence"
date: 2026-07-13 20:00:00
description: "Megatron-SP patches TP's activation leak (the AR = AG + RS identity, third appearance), Ring Attention attacks O(s²) long context — with a hand-written ring attention verified exact, and a scaling result that inverts post #5's."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/06/fig-2-ring-attention.png
toc:
  sidebar: left
related_posts: false
---

> Part 6 of **Distributed Training, Illustrated**. One post, one idea (with two faces): **the two schemes that cut along the sequence dimension solve two different problems** — Megatron-SP patches TP's activation leak (the AR = AG + RS identity again, its third appearance in this series), Ring Attention (CP) attacks long context's $$O(s^2)$$. Experiments: a hand-written ring attention verified exactly equivalent, which — on the very PCIe box where TP flatlined — **actually scales**.

## 1. The activation leak TP leaves behind

Post #5's TP cut the weights and compute of attention/MLP, but look at its boundaries: $$f$$'s forward **replicates** $$X$$, and $$g$$'s forward all-reduces out a **complete** $$Z$$ — **at the entry and exit of every TP region, activations are full $$[b,s,h]$$, replicated $$N$$ times.** The LayerNorm and dropout sandwiched between TP regions operate on those replicated activations: their *compute* is negligible, but their **activation memory** (LN's input must be kept for backward; dropout's mask likewise) is not reduced by one byte.

Sequence parallelism (SP, Megatron-LM 2022) observes: **LN and dropout are completely independent across tokens** — LN normalizes each token's $$h$$-vector; dropout is element-wise. Token-independent ⇒ cut along the sequence, let each rank process its own segment, and the math is unchanged.

## 2. Megatron-SP: split the all-reduce in SPACE

What remains is stitching the two layouts together: TP regions want the full $$s$$; SP regions hold $$s/N$$. The answer is the same identity — **all-reduce = all-gather + reduce-scatter** (post #1):

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/06/fig-1-megatron-sp.svg" class="img-fluid rounded" zoomable=true %}

- Plain TP: LN (full $$s$$, replicated) → **g: all-reduce** → TP region → **g: all-reduce** → LN (full $$s$$, replicated);
- TP + SP: LN (**$$s/N$$, sharded**) → **ḡ: all-gather** (assemble the sequence, enter the TP region) → TP region (unchanged) → **g̅: reduce-scatter** (sum the partial results and re-split by sequence) → LN (**$$s/N$$, sharded**).

Reconcile: one all-reduce costs $$2\frac{N-1}{N}S$$ per GPU; replacing it with AG ($$\frac{N-1}{N}S$$) + RS ($$\frac{N-1}{N}S$$) moves **not one extra byte**. Meanwhile LN/dropout activations shrink from $$[b,s,h]$$ to $$[b,\frac{s}{N},h]$$ — the last replicated activations inside the TP domain vanish.

This is the identity's third appearance, each time split differently — one of this series' own through-lines, worth laying side by side:

| | How the AR is split | What it buys |
|---|---|---|
| **ZeRO** (post #3) | in **time**: local optimizer update inserted between RS and AG | optimizer state /N |
| **FSDP** (post #4) | in time, inverted: AG moved into forward, RS into backward | resident params /N |
| **Megatron-SP** (this post) | in **space**: AG at the TP region's entry, RS at its exit | LN/dropout activations /N |

> A detail that shows why the split is *necessary*: couldn't we just recompute LN instead of storing its input? Sure — that's activation checkpointing, which pays compute for memory. SP is **free**: same communication, same compute, memory /N. The only engineering cost: LN's $$\gamma/\beta$$ gradients need a cross-rank sum — an all-reduce of one $$[h]$$ vector, latency-floor territory from post #1, ≈ 0.

## 3. CP: when the sequence itself is the enemy

SP trims what TP left behind; context parallelism (CP) faces a different magnitude of problem: **$$s$$ so large that one GPU can't even hold a single layer's activations** ($$[b,s,h]$$ at 128K tokens, and attention's $$O(s^2)$$). So cut the *entire network* along the sequence: each rank owns $$1/N$$ of the tokens, and **LN, MLP, and the QKV projections are all token-local — they compute as usual with zero communication.** In the whole Transformer, **the only place tokens interact is attention's $$QK^\top$$**. CP's entire problem compresses into one question: *my $$Q$$ is here, the other K/V segments are on other GPUs — how do we compute?*

Ring Attention's answer: **Q stays home; KV makes the round trip.**

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/06/fig-2-ring-attention.svg" class="img-fluid rounded" zoomable=true %}

Each rank keeps its $$Q_k$$; K/V blocks hop to the next rank each step (N−1 hops complete the ring). On each arriving KV block, an **online softmax** absorbs it into running statistics $$(O, m, l)$$:

$$
m_j = \max(m_{j-1},\ \mathrm{rowmax}(S_j)),\quad
l_j = l_{j-1}e^{m_{j-1}-m_j} + \textstyle\sum e^{S_j - m_j},\quad
O_j = O_{j-1}e^{m_{j-1}-m_j} + e^{S_j-m_j}V_j
$$

Then $$O/l$$ is the **exact** softmax attention — old statistics get rescaled as new maxima arrive; it is an algebraic identity, not an approximation (the very trick behind FlashAttention's tiling; CP simply places the tiles on different GPUs). Each hop moves $$2\cdot\frac{s}{N}\cdot h$$ bytes, and the next block's transfer can overlap the current block's compute.

## 4. Experiment: exactness, and "same machine — TP flatlines, CP scales"

Hand-written ring attention (core = one `blockwise_update` + one `batch_isend_irecv` ring, ~30 lines, ships with the post), $$s = 8192$$, 8 heads, non-causal, against single-GPU full softmax:

| CP | max error | compute (ms) | KV ring (ms) | total (ms) |
|----|-----------|--------------|--------------|------------|
| 2 | 4.4e-7 | 8.32 | ≈0 | 8.26 |
| 4 | 3.9e-7 | 3.95 | 1.11 | 5.07 |
| 8 | 3.4e-7 | 1.41 | 1.98 | 3.38 |

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/06/fig-3-cp-scaling.svg" class="img-fluid rounded" zoomable=true %}

Two readings:

1. **Exactness**: error pinned at 4e-7 (fp32 rounding order), independent of $$N$$ — online softmax is an identity, not an approximation;
2. **The counterpoint to post #5**: on this same NVLink-less machine, TP=8 spent 86% of its time communicating and its total never moved; CP=8 cuts the total from 8.3 ms to 3.4 ms (2.4×). The reason is scaling: **attention's compute is $$O(s^2/N)$$ — quadratic — while the KV ring moves only $$O(s/N)$$.** The longer the sequence, the more lopsided the compute/communication ratio, the closer CP gets to linear scaling. Same model, different cut, opposite communication fate — **a parallelism scheme's scalability is its compute-to-communication *scaling ratio*, not what it cuts.**

> Honest boundary: the toy is non-causal. Under a causal mask, naive contiguous chunks load-imbalance badly (the rank holding the sequence tail attends to almost all KV; the head, almost none) — production implementations (Megatron CP, zigzag ring) pair head and tail segments to rebalance. And our ring is serial; production overlaps hops with compute, hiding the communication almost entirely. Both corrections only make CP scale *better*.

## 5. Reading along in real source

- Megatron-SP: the SP conjugate pair in `megatron/core/tensor_parallel/mappings.py` (`gather_from_sequence_parallel_region` / `reduce_scatter_to_sequence_parallel_region`) — read side by side with post #5's $$f/g$$ and the identity-splitting is plain to see;
- nanotron: `TensorParallelLinearMode.REDUCE_SCATTER` (`src/nanotron/parallel/tensor_parallel/nn.py`) — SP is not a separate module but a *communication-mode switch on the TP linear layer*, a design that itself says "SP = TP's communication, rearranged";
- Ring Attention: Liu et al. 2023; production versions in Megatron's `context_parallel` (p2p ring or all-gather comm types) and flash-attn with zigzag sharding;
- Our toy is §3's three formulas transcribed — read it line-against-line with the math.

## 6. Summary

1. SP and CP both cut the sequence but for different reasons: **SP patches TP's activation leak** (replicated LN/dropout activations), **CP attacks long context** (the whole net cut by token; only attention communicates);
2. Megatron-SP = the AR = AG + RS identity split in **space** (third appearance: ZeRO in time, FSDP in time-inverted, SP in space) — zero extra communication, LN/dropout activations /N;
3. Ring attention: Q stays, KV rings, online softmax stays **exact**; tokens only interact in attention, so everything else is free;
4. Measured: same PCIe box, TP flatlines while CP speeds up 2.4× — **scalability is decided by the compute/communication scaling ratio** ($$O(s^2/N)$$ vs $$O(s/N)$$), a deeper criterion than what gets cut.

**Next: Pipeline Parallelism — the geometry of bubbles.** DP/TP/SP/CP all have every GPU doing the *same kind* of work; PP is the first to give different GPUs *different layers*, and with that comes parallel training's most famous disease: the bubble. GPipe vs 1F1B schedules, and the $$(p-1)/m$$ bubble fraction derived and measured.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node={2,4,8} bench_ring_attention.py`; plotting and schematic code accompanies the series.*
