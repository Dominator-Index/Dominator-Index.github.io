---
layout: post
title: "Distributed Training, Illustrated #5 — Tensor Parallelism: Megatron's Two Cuts"
date: 2026-07-13 18:00:00
description: "Why Column→Row is forced by the nonlinearity, a full backward-pass derivation showing that ALL weight gradients are communication-free, the bs/3h crossover against DP — and a hand-written TP MLP verified bit-exact against one GPU."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/05/fig-1-golden-pair.png
toc:
  sidebar: left
related_posts: false
---

> Part 5 of **Distributed Training, Illustrated**. One post, one idea: **how TP cuts the computation of a single layer, and why the Column→Row order is dictated by the nonlinearity.** Unlike most tutorials, this post derives the **backward pass in full** — revealing a beautiful fact rarely spelled out: in TP, **every weight gradient is communication-free**; communication only ever touches activations. Experiments: a hand-written Column+Row MLP verified numerically exact against a single GPU, plus the TP=2/4/8 communication share on pure PCIe.

## 1. Until now, computation was never cut

The previous four posts sharded **storage**: ZeRO/FSDP split parameters, gradients and optimizer state, but every forward pass all-gathers the parameters back — **the matrix multiply itself still runs whole on every GPU**. When a single layer's compute or activations are themselves too big or too slow, you need tensor parallelism (TP): split $$Y = XW^\top$$ — *one* matmul — across $$N$$ GPUs, each doing $$1/N$$ of it.

There are two natural directions to cut. Notation (PyTorch layout, series-wide): weight $$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$, activations $$X \in \mathbb{R}^{bs \times d_{\text{in}}}$$, the layer computes $$Y = XW^\top$$:

- **Column cut** (Megatron's `ColumnParallelLinear`): split along $$d_{\text{out}}$$; rank $$k$$ holds $$W_k = W[k\frac{d_{\text{out}}}{N}:(k{+}1)\frac{d_{\text{out}}}{N},\,:]$$ — **complete rows**. $$X$$ is replicated; the output comes out naturally column-blocked: $$Y_k = XW_k^\top$$, **zero communication**, but each rank has only $$1/N$$ of the output;
- **Row cut** (`RowParallelLinear`): split along $$d_{\text{in}}$$. This *requires* the input to arrive blocked the same way ($$X_k$$), and the output is a **partial sum**: $$Y = \sum_k X_k W_k^\top$$ — one all-reduce away from complete.

> Naming trap, resolved once: Megatron's Column/Row refer to the math convention $$Y=XW$$ with $$W\in\mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$$ (Column = the output dim); in PyTorch's `[out, in]` tensor layout that is precisely a **dim-0 (row) cut**. The one question that never misleads: **"is each row (fan-in vector) still whole?"** Column cut: yes. Row cut: no. This distinction becomes life-or-death in post #9.

## 2. The golden pair: two defects that cancel

Each cut alone has a defect: Column leaves the output blocked; Row demands a blocked input. Megatron's insight: **the two defects cancel exactly** — Column's blocked output *is* the blocked input Row wants:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/05/fig-1-golden-pair.svg" class="img-fluid rounded" zoomable=true %}

The full MLP dataflow ($$Z = \mathrm{GeLU}(XW_1^\top)W_2^\top$$):

$$
X \xrightarrow{\ f\ } \underbrace{Y_k = \mathrm{GeLU}(XW_{1,k}^\top)}_{\text{Column cut, no comm}} \longrightarrow \underbrace{Z_k = Y_k W_{2,k}^\top}_{\text{Row cut, no comm}} \xrightarrow{\ g:\ \text{all-reduce}\ } Z = \textstyle\sum_k Z_k
$$

**The GeLU lands exactly on the "each rank owns its complete block" state — apply it element-wise, done.** One all-reduce per MLP forward, at the exit. Why can't the order flip? Because GeLU is nonlinear:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/05/fig-2-why-order.svg" class="img-fluid rounded" zoomable=true %}

$$
\mathrm{GeLU}(Z_0 + Z_1) \neq \mathrm{GeLU}(Z_0) + \mathrm{GeLU}(Z_1)
$$

Row-cut first, and fc1's output is a partial sum that **must be all-reduced before the GeLU** — communication forced into the middle of the layer, two collectives instead of one. **The position of the nonlinearity dictates the cut order; that is the entire design of Megatron TP.** Attention is the same story, more natural still: each head is already an independent unit, so QKV projections are Column-cut (whole heads per rank), the output projection $$W_O$$ is Row-cut, and softmax (per-head) sits in the middle communication-free.

## 3. The backward pass: derive every gradient, see where communication really is

Most tutorials skip this part — yet it answers why TP's communication is as cheap as it is. Backward through the dataflow above (write $$\bar{A} \equiv \frac{\partial L}{\partial A}$$, given $$\bar{Z}$$; $$g$$'s backward is identity, so $$\bar{Z}$$ is replicated):

**(1) The Row layer's weight gradient — no communication:**

$$
\bar{W}_{2,k} = \bar{Z}^\top Y_k
$$

$$\bar{Z}$$ is on every rank (replicated); $$Y_k$$ lives on this rank already — both multiplicands local, **no communication needed**. Note also: $$\bar{W}_{2,k}$$ is exactly the gradient of this rank's own shard — storage and update fully local (naturally orthogonal to ZeRO/FSDP).

**(2) The intermediate activation's gradient — no communication:**

$$
\bar{Y}_k = \bar{Z}\, W_{2,k}, \qquad \bar{Y}^{\text{pre}}_k = \bar{Y}_k \odot \mathrm{GeLU}'(XW_{1,k}^\top)
$$

Uses only this rank's $$W_{2,k}$$ and this rank's activations — local.

**(3) The Column layer's weight gradient — no communication:**

$$
\bar{W}_{1,k} = (\bar{Y}^{\text{pre}}_k)^\top X
$$

$$X$$ is replicated ($$f$$'s forward is a copy); $$\bar{Y}^{\text{pre}}_k$$ is local — free again.

**(4) The input's gradient — the one and only communication:**

$$
\bar{X} = \sum_k \bar{Y}^{\text{pre}}_k W_{1,k}
$$

Each rank can compute only its own term — a **partial sum** — hence an all-reduce. That is $$f$$'s backward.

Put the four together and a clean structural fact emerges:

> **All weight gradients are communication-free; communication happens only at activation boundaries ($$f$$'s backward, $$g$$'s forward), and $$f/g$$ are conjugates — one copies forward and sums backward, the other the reverse.** The per-layer bill: 1 forward AR + 1 backward AR for the MLP, plus the same pair for attention — **4 activation all-reduces per layer per step.**

Why are weight gradients free? Because the golden pair guarantees that **each weight shard's two multiplicands — its own activation block, and the replicated boundary activation — are both local.** That is not luck; it is what "aligning the cut geometry with the dataflow" buys. This view runs deeper than memorizing "Column pairs with Row," and it generalizes: to audit any new parallelism scheme's communication, **stare at each gradient formula and ask "are the multiplicands local?"**

## 4. The bill: TP moves activations — that is its essential difference from DP

DP synchronizes **gradients** each step ($$\sim 2\Psi$$ bytes, independent of batch); TP moves **activations** $$[bs \times h]$$ every layer (proportional to batch, independent of parameter count). The ratio ($$L$$ layers, $$\Psi \approx 12Lh^2$$, 4 ARs per layer):

$$
\frac{\text{TP comm per step}}{\text{DP comm per step}} = \frac{L \cdot 4 \cdot bsh}{\Psi} = \frac{4L \cdot bsh}{12Lh^2} = \frac{bs}{3h}
$$

(the common $$2\frac{N-1}{N}$$ factor cancels; same dtype assumed.) Past $$3h$$ tokens per rank (GPT-2 Large: 3840), TP out-communicates DP — which in training is essentially always. **And TP's communication comes as serial, per-layer, latency-critical packets wedged between matmuls** (hard to overlap), while DP's is one large per-step transfer that bucketing hides (post #2). This is the quantitative version of post #0's iron rule: *tp stays inside the node.*

## 5. Experiment: exactness verified, and the PCIe reality check

A hand-written Column+Row MLP (~40 lines of core, ships with the post), GPT-2 Large geometry (h=1280, ff=5120, 4096 tokens), checked against a single GPU:

| TP | max forward error | max $$\bar{X}$$ error |
|----|----|----|
| 2 | 3.8e-6 | 2.7e-12 |
| 4 | 3.8e-6 | 2.8e-12 |
| 8 | 3.3e-6 | 2.6e-12 |

The 1e-6 forward error is fp32 summation-order rounding (all-reduce adds in a different order than a fused matmul); backward agrees to 1e-12. **TP, like DP, is an exact algorithm, not an approximation** — "zero communication in the middle" costs no precision.

Speed (one MLP forward, compute/communication decomposed):

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/05/fig-3-tp-scaling.svg" class="img-fluid rounded" zoomable=true %}

- **Compute divides perfectly by N**: 1.32 → 0.64 → 0.34 ms — mathematically TP is impeccable;
- **Communication grows with N**: the all-reduce message is constant ($$[4096\times1280]$$ fp32 = 21 MB) but the ring gets longer and crosses more NUMA boundaries: 1.07 → 1.66 → 2.09 ms. Reconciling with post #1: 21 MB in 2.09 ms → algbw ≈ 10 GB/s — precisely the measured 8-GPU all-reduce plateau ✓;
- **Total time doesn't move** (2.39 → 2.30 → 2.43 ms); the communication share climbs 45% → 72% → **86%**. On a machine without NVLink, TP hands every saved FLOP to the wires — **not TP's failure, but the measured proof of "TP must live inside a fast-interconnect domain."** On NVLink (~25× the bandwidth) the same experiment's comm term divides by ~25 and TP=8's share falls back to ~20%.

## 6. Reading along in real source

- Megatron-LM: `ColumnParallelLinear` / `RowParallelLinear` in `megatron/core/tensor_parallel/layers.py`; the $$f/g$$ operators in `mappings.py` (`copy_to_tensor_model_parallel_region` = $$f$$, `reduce_from_tensor_model_parallel_region` = $$g$$ — the names literally say copy-in / reduce-out);
- nanotron: `TensorParallelColumnLinear` (`SplitConfig(split_dim=0)` — the tensor-row cut) and `TensorParallelRowLinear` (`split_dim=1`) in `src/nanotron/parallel/tensor_parallel/nn.py`; the differentiable primitives in `distributed_differentiable_primitives.py` implement $$f/g$$ as `autograd.Function`s — §3's derivation, as code;
- Our `bench_tp.py` strips those classes to the bone: one shard per rank, one all_reduce, all of §3 reproducible in 40 lines.

## 7. Summary

1. TP is the first scheme to cut **computation**: Column (rows whole, output blocked) + Row (rows severed, output partial-summed) form the golden pair; **the nonlinearity's position dictates the order**; the middle is communication-free;
2. Deriving the backward term by term: **weight gradients are all communication-free; only activation boundaries communicate** ($$f$$-backward + $$g$$-forward), 4 activation ARs per layer per step. The general audit method: stare at each gradient formula and ask *are the multiplicands local?*;
3. TP comm ∝ activations ($$bsh$$), DP comm ∝ parameters ($$\Psi$$); ratio $$bs/3h$$ — TP is almost always the bigger bill, serial and hard to overlap → the quantitative basis for keeping tp inside nodes;
4. Measured: an exact algorithm (error = rounding order); on PCIe, compute scales perfectly ÷N while total time stays flat (comm share 86%) — the iron rule, as an experiment.

**Next: Sequence & Context Parallelism.** TP cut the weights, but each rank's activations are still the full $$[b \times s \times h]$$; the parts TP cannot reach (LayerNorm, dropout) and the long-context attention problem both call for cutting along the **sequence** dimension — two different schemes for two different problems.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node={2,4,8} bench_tp.py`; plotting and schematic code accompanies the series.*
