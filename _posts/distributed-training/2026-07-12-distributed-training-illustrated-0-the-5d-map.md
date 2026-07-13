---
layout: post
title: "Distributed Training, Illustrated #0 — A Map of 5D Parallelism"
date: 2026-07-12 10:00:00
description: "Why one GPU is never enough: the 16Ψ memory ledger, the two walls, and a map of the five parallelism dimensions this series will visit one by one."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/00/fig-2-5d-map.png
toc:
  sidebar: left
related_posts: false
---

> This is the prologue of **Distributed Training, Illustrated** — a series where each post explains exactly one idea (DP/DDP, ZeRO, FSDP, TP, SP/CP, PP), with the math derived from scratch, real source code to read along, and experiments actually run on an 8-GPU machine. This post is the map: **why a single GPU cannot train a large model, and what each parallelism dimension cuts.**

## 1. The two walls

Training a large model on one GPU runs into two walls, in this order.

**The memory wall.** During training, every parameter must live in memory in several bodies at once. Let $$\Psi$$ denote the *number* of parameters (a plain count — $$\Psi = 7\times 10^9$$ for a 7B model). Under mixed precision with Adam, the ledger reads:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/00/fig-1-memory-ledger.svg" class="img-fluid rounded" zoomable=true %}

$$
\underbrace{2\Psi}_{\text{bf16 params}} + \underbrace{2\Psi}_{\text{bf16 grads}} + \underbrace{4\Psi}_{\text{fp32 master params}} + \underbrace{4\Psi + 4\Psi}_{\text{Adam } m,\ v} = \boxed{16\Psi \text{ bytes}}
$$

Mnemonic: **training a model ≈ storing 8 copies of it**, and three quarters of the bill is the fp32 optimizer state (why parameters can be bf16 while the optimizer must stay fp32 is post #8; this 12Ψ of "redundancy" is exactly what ZeRO, post #3, will shard away). For a 7B model the checkpoint is 14 GB, but the training state is **112 GB** — it does not fit in our 96 GB cards. And that is before activations, which grow with batch size × sequence length and are a separate ledger entirely.

**The time wall.** Even if it fit: total training compute is roughly $$6\Psi D$$ FLOPs ($$D$$ = training tokens; 2 for the forward pass, 4 for the backward). A 7B model on 1T tokens is $$4.2\times 10^{22}$$ FLOPs — over six years on one GPU at a realistic ~200 TFLOPS.

So we must use many GPUs. And every multi-GPU scheme ever devised is an answer to the same question:

> **What do we cut, which group of GPUs do the shards live on, and which collective operation glues them back together?**

## 2. Five dimensions, one map

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/00/fig-2-5d-map.svg" class="img-fluid rounded" zoomable=true %}

| Dimension | Cuts | Attacks which wall | The "glue" | Post |
|-----------|------|--------------------|------------|------|
| **DP** — data parallelism | the batch (model replicated) | time (throughput) | all-reduce on gradients | #2 |
| **TP** — tensor parallelism | each weight matrix, row/col-wise | memory (one layer too big) | all-reduce on activations | #5 |
| **PP** — pipeline parallelism | the layer stack, into stages | memory (too many layers) | p2p send/recv | #7 |
| **CP** — context parallelism | the sequence dimension | activation memory (long context) | ring KV exchange | #6 |
| **EP** — expert parallelism | MoE experts | MoE parameter count | all-to-all | (season 2) |

Three players that don't occupy an independent dimension but matter just as much:

- **ZeRO / FSDP** (posts #3, #4): an upgrade of DP — the model still computes as a full replica, but the redundant entries of the 16Ψ ledger (optimizer state, gradients, parameters) are sharded across the DP group. It cuts **storage**, not computation;
- **SP** — sequence parallelism (post #6): a companion to TP that shards the activations TP cannot reach (LayerNorm, dropout) along the sequence dimension;
- **Mixed precision** (post #8): the very rules that make the ledger above read bf16/fp32 in the first place.

Real large-scale training is a **product** of these dimensions: the GPUs form a multi-dimensional grid `ep × pp × dp × cp × tp`, and every GPU belongs to five communication groups at once. The composition is not arbitrary — one iron rule decides the layout: **the more often a dimension communicates, the faster the interconnect it must sit on.**

$$
\underbrace{\text{tp}}_{\text{every layer}} \;>\; \text{cp} \approx \text{zero-3} \;>\; \text{dp} \;>\; \underbrace{\text{pp}}_{\text{only at stage boundaries}}
\quad\Longrightarrow\quad
\text{tp stays inside a node; pp/dp go across nodes.}
$$

## 3. The common substrate: six collective primitives

Look at the "glue" column of the map once more: **every dimension's glue is drawn from the same small set of collective communication primitives** — broadcast, scatter, gather, all-gather, reduce-scatter, all-reduce. They are the accounting unit of this whole series: DDP's cost is one all-reduce; ZeRO's is a reduce-scatter plus an all-gather; FSDP's prefetch is a pipelined all-gather. Until you know what one all-reduce costs in bytes and microseconds, no claim of the form "X saves communication over Y" can be checked.

That is why the series opens with the primitives (post #1): what each one means, the $$2\frac{N-1}{N}S$$ derivation for ring all-reduce, and their measured bandwidth and latency on our 8-GPU machine — where you will see that the theoretical formula is not an approximation but a change of coordinates on the measured curves.

## 4. Series roadmap

1. **Collective communication primitives** — how the all-reduce bill is computed ✅
2. **Data parallelism, part 1** — from DP to DDP: bucketing and compute/comm overlap
3. **Data parallelism, part 2** — ZeRO's three-stage ledger of zero redundancy
4. **FSDP** — how PyTorch implements ZeRO-3 (FSDP1 vs FSDP2)
5. **Tensor parallelism** — Megatron's two cuts
6. **Sequence & context parallelism** — two ways to cut the sequence
7. **Pipeline parallelism** — the geometry of bubbles
8. **Mixed precision** — the numerics ledger of the bf16 era
9. **Optimizers in a sharded world** — why row-local operators are natively distributed-friendly

All experiments run on one machine: 8× RTX PRO 6000 Blackwell (96 GB), pure PCIe with no NVLink, PyTorch 2.9.1 + NCCL 2.27.5. The missing NVLink is a feature, not a bug — on a "commoner topology" the communication bottlenecks are exposed with unusual clarity.

---

*References: the HuggingFace [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook); the nanotron, Megatron-LM and DeepSpeed source trees; the 图解大模型训练 series by 猛猿. All schematics and plots in this series are original, generated by code published alongside each post.*
