---
layout: post
title: "Splitting The Sequence: Megatron-SP And Ring Attention"
date: 2026-07-11 10:00:00
description: "How sequence parallelism removes the activations replicated by tensor parallelism, how Ring Attention distributes long contexts, and why context parallelism scales on the same PCIe machine where TP does not."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/06/fig-2-ring-attention.png
toc:
  sidebar: left
related_posts: false
---

> Sequence parallelism (SP) and context parallelism (CP) both divide the sequence dimension, but they solve different problems. SP removes the LayerNorm and dropout activations left replicated by tensor parallelism. CP distributes the $O(s^2)$ work and memory of long-context attention. A hand-written Ring Attention implementation matches full softmax up to rounding and scales on the same PCIe machine where TP does not.

## 1. Activations that TP leaves replicated

Post #5 showed how TP divides the weights and computation of attention and MLP layers. At TP-region boundaries, however, the forward operation of $f$ replicates $X$ and the forward operation of $g$ all-reduces a complete $Z$. Each rank therefore stores full $[b,s,h]$ activations at the entry and exit of every TP region. LayerNorm and dropout operate between these regions. Their compute cost is small, but their inputs and masks must be retained for backward, so TP does not reduce this activation memory.

Sequence parallelism (SP, Megatron-LM 2022) uses the fact that **LayerNorm and dropout are independent across tokens**. LayerNorm normalizes each token's $h$-dimensional vector, and dropout is element-wise. The sequence can therefore be divided across ranks without changing either operation.

## 2. Megatron-SP: replacing all-reduce with all-gather and reduce-scatter

TP regions use the full sequence, while SP regions store only $s/N$ tokens per rank. Converting between these layouts uses the identity from post #1: **all-reduce consists of reduce-scatter and all-gather**.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/06/fig-1-megatron-sp.svg" class="img-fluid rounded" zoomable=true %}

- **Plain TP:** LayerNorm stores the full replicated sequence → $g$ performs all-reduce around each TP region → the next LayerNorm again receives the full replicated sequence.
- **TP + SP:** LayerNorm stores $s/N$ tokens per rank → all-gather assembles the sequence before the TP region → reduce-scatter sums the TP outputs and repartitions them by sequence → the next LayerNorm again stores only $s/N$ tokens.

The communication volume is unchanged. All-reduce sends $2\frac{N-1}{N}S$ bytes per GPU, while all-gather and reduce-scatter each send $\frac{N-1}{N}S$. At the same time, LayerNorm and dropout activations shrink from $[b,s,h]$ to $[b,\frac{s}{N},h]$ on each rank.

The same decomposition has appeared in three different settings:

| | How the AR is split | What it buys |
|---|---|---|
| **ZeRO** (post #3) | in **time**: local optimizer update inserted between RS and AG | optimizer state /N |
| **FSDP** (post #4) | in time, inverted: AG moved into forward, RS into backward | resident params /N |
| **Megatron-SP** (this post) | in **space**: AG at the TP region's entry, RS at its exit | LN/dropout activations /N |

> Recomputing LayerNorm during backward would also save its stored input, but that is activation checkpointing and trades additional compute for memory. SP instead keeps the main communication and compute volumes unchanged while dividing these activations by $N$. LayerNorm's $\gamma$ and $\beta$ gradients require an additional all-reduce of one $[h]$ vector. This message is small and therefore latency-bound, as described in post #1.

## 3. Context parallelism for long sequences

SP removes activations replicated by TP. Context parallelism (CP) addresses sequences so long that one GPU cannot hold a layer's activations or attention matrix. At 128K tokens, even the $[b,s,h]$ activations can be too large, while the attention matrix grows as $O(s^2)$. CP divides the entire network along the sequence, so each rank owns $1/N$ of the tokens. LayerNorm, MLP and QKV projections operate independently on each token and remain local. Cross-token interaction occurs in attention through $QK^\top$. The remaining problem is to combine local queries with K/V blocks stored on other ranks.

Ring Attention keeps each query block local and circulates the K/V blocks around the ranks.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/06/fig-2-ring-attention.svg" class="img-fluid rounded" zoomable=true %}

Rank $k$ keeps $Q_k$. At each of the next $N-1$ steps, it receives a new K/V block from its neighbor. An **online softmax** incorporates each block into running output, maximum and normalization statistics $(O,m,l)$:

$$
m_j = \max(m_{j-1},\ \mathrm{rowmax}(S_j)),\quad
l_j = l_{j-1}e^{m_{j-1}-m_j} + \textstyle\sum e^{S_j - m_j},\quad
O_j = O_{j-1}e^{m_{j-1}-m_j} + e^{S_j-m_j}V_j
$$

After every block is processed, $O/l$ equals full softmax attention up to floating-point ordering. When a larger maximum appears, the previous statistics are rescaled before the new block is added. FlashAttention uses the same algebra within one GPU. Ring Attention distributes the blocks across GPUs. Each hop moves $2\cdot\frac{s}{N}\cdot h$ elements for K and V, and communication of the next block can overlap computation on the current block.

## 4. Experiment: numerical equivalence and CP scaling

Hand-written ring attention (core = one `blockwise_update` + one `batch_isend_irecv` ring, ~30 lines, ships with the post), $$s = 8192$$, 8 heads, non-causal, against single-GPU full softmax:

| CP | max error | compute (ms) | KV ring (ms) | total (ms) |
|----|-----------|--------------|--------------|------------|
| 2 | 4.4e-7 | 8.32 | ≈0 | 8.26 |
| 4 | 3.9e-7 | 3.95 | 1.11 | 5.07 |
| 8 | 3.4e-7 | 1.41 | 1.98 | 3.38 |

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/06/fig-3-cp-scaling.svg" class="img-fluid rounded" zoomable=true %}

The measurements show two results:

1. **Numerical equivalence:** the maximum difference remains near 4e-7 for every $N$. It comes from fp32 operation order rather than an algorithmic approximation.
2. **CP scales where TP did not:** on the same machine without NVLink, TP=8 spent 86% of its time communicating and did not reduce total time. CP=8 reduces attention time from 8.3 to 3.4 ms, a 2.4× speedup. The difference comes from asymptotic scaling. Attention compute is $O(s^2/N)$, while each rank's KV communication is $O(s/N)$. As sequence length grows, computation grows faster than communication, making the communication easier to amortize. Scalability therefore depends on how compute and communication scale relative to each other.

> **Boundary of the experiment.** This implementation is non-causal. With a causal mask and contiguous chunks, later chunks attend to more K/V positions than earlier chunks and create load imbalance. Production systems such as Megatron CP and zigzag ring attention pair early and late segments to balance the work. Our implementation also executes ring transfers serially, while production systems overlap communication with attention computation.

## 5. Reading along in real source

**Megatron-SP:** `megatron/core/tensor_parallel/mappings.py` implements the SP pair as `gather_from_sequence_parallel_region` and `reduce_scatter_to_sequence_parallel_region`. These functions are the sequence-sharded counterparts of the $f/g$ operators in post #5.

**nanotron:** `TensorParallelLinearMode.REDUCE_SCATTER` is defined in `src/nanotron/parallel/tensor_parallel/nn.py`. SP is implemented as a communication mode of the TP linear layer rather than as a separate module.

**Ring Attention** comes from Liu et al. 2023. Production versions live in Megatron's `context_parallel` (p2p ring or all-gather comm types) and in flash-attn with zigzag sharding.

**Our implementation** directly follows the three online-softmax equations in Section 3.

## 6. Summary

1. SP and CP both divide the sequence dimension but solve different problems. **SP shards LayerNorm and dropout activations left replicated by TP**, while **CP distributes long-context computation and memory** across tokens.
2. Megatron-SP replaces each relevant all-reduce with an all-gather at the TP-region entrance and a reduce-scatter at the exit. The communication volume is unchanged, while LayerNorm and dropout activation memory is divided by $N$.
3. In Ring Attention, query blocks remain local while K/V blocks circulate. Online softmax produces the same result as full attention up to floating-point ordering. Other token-local layers need no communication.
4. On the same PCIe machine, TP does not reduce total time, while CP achieves a 2.4× speedup. The difference follows from their compute-to-communication scaling: $O(s^2/N)$ compute versus $O(s/N)$ communication for CP.

**Next comes pipeline parallelism and the geometry of pipeline bubbles.** DP, TP, SP and CP assign the same type of work to every GPU. Pipeline parallelism instead assigns different layers to different stages. The next post compares GPipe and 1F1B schedules and derives and measures their bubble fraction.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node={2,4,8} bench_ring_attention.py`. Plotting and schematic code accompanies the series.*

*All benchmark scripts, schematic generators, plotting code and raw result CSVs for this post live in [assets/blog/code/06-sp-cp](https://github.com/Dominator-Index/Dominator-Index.github.io/tree/main/assets/blog/code/06-sp-cp).*

---

## Appendix: The Code That Ran

Every number in this post comes from the scripts below, embedded verbatim. Plotting and schematic code plus the raw result CSVs live in the folder linked above.

<details markdown="1">
<summary><code>bench_ring_attention.py</code></summary>

```python
"""
Toy Ring Attention implementation (part 06 of the Illustrated Distributed Training series).

Each rank holds 1/N of the sequence (its own Q_k, K_k, V_k). KV blocks travel
around the ring for N-1 hops while each rank accumulates locally with online
softmax. Mathematically this is exactly equivalent to full attention, not an
approximation.

Verification: compare the output against single-GPU full attention (fp32).
Timing: breakdown of compute (blockwise attention) vs comm (KV ring exchange).

Usage:
  torchrun --standalone --nproc_per_node={2,4,8} bench_ring_attention.py --out ../results/ringattn.csv
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist

B, NH, S_TOTAL, D = 1, 8, 8192, 64   # batch, heads, total sequence length, head dim
WARMUP, STEPS = 10, 30


def blockwise_update(O, m, l, Q, K, V):
    """Online softmax: absorb one new KV block. O: [*, sq, d], m/l: [*, sq, 1]"""
    S = Q @ K.transpose(-2, -1) / D**0.5            # [*, sq, skv]
    m_blk = S.max(dim=-1, keepdim=True).values
    m_new = torch.maximum(m, m_blk)
    scale = torch.exp(m - m_new)                     # rescale the old running stats
    p = torch.exp(S - m_new)                         # unnormalized weights of the new block
    l_new = l * scale + p.sum(dim=-1, keepdim=True)
    O_new = O * scale + p @ V
    return O_new, m_new, l_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/ringattn.csv")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    s_local = S_TOTAL // world
    torch.manual_seed(1337)  # same seed: every rank generates the same full QKV, then takes its own slice
    Qf = torch.randn(B, NH, S_TOTAL, D, device=device)
    Kf = torch.randn(B, NH, S_TOTAL, D, device=device)
    Vf = torch.randn(B, NH, S_TOTAL, D, device=device)
    sl = slice(rank * s_local, (rank + 1) * s_local)
    Q, K, V = Qf[:, :, sl].contiguous(), Kf[:, :, sl].contiguous(), Vf[:, :, sl].contiguous()

    # ---- single-GPU reference (non-causal, full softmax attention) ----
    ref = torch.softmax(Qf @ Kf.transpose(-2, -1) / D**0.5, dim=-1) @ Vf
    ref_local = ref[:, :, sl]

    nxt, prv = (rank + 1) % world, (rank - 1) % world

    def ring_attention():
        O = torch.zeros_like(Q)
        m = torch.full((B, NH, s_local, 1), -float("inf"), device=device)
        l = torch.zeros(B, NH, s_local, 1, device=device)
        k_cur, v_cur = K.clone(), V.clone()
        for step in range(world):
            if step < world - 1:  # send the current KV out first (a chance to overlap with compute, this toy version is serial)
                k_buf, v_buf = torch.empty_like(k_cur), torch.empty_like(v_cur)
                ops = [dist.P2POp(dist.isend, k_cur, nxt), dist.P2POp(dist.irecv, k_buf, prv),
                       dist.P2POp(dist.isend, v_cur, nxt), dist.P2POp(dist.irecv, v_buf, prv)]
                reqs = dist.batch_isend_irecv(ops)
            O, m, l = blockwise_update(O, m, l, Q, k_cur, v_cur)
            if step < world - 1:
                for r in reqs:
                    r.wait()
                k_cur, v_cur = k_buf, v_buf
        return O / l

    out = ring_attention()
    err = (out - ref_local).abs().max().item()

    # ---- timing breakdown ----
    def timed(fn):
        s_, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
        dist.barrier(); torch.cuda.synchronize()
        s_.record()
        for _ in range(STEPS):
            fn()
        e_.record(); torch.cuda.synchronize()
        return s_.elapsed_time(e_) / STEPS

    def compute_only():  # same number of blockwise_update calls, no communication
        O = torch.zeros_like(Q)
        m = torch.full((B, NH, s_local, 1), -float("inf"), device=device)
        l = torch.zeros(B, NH, s_local, 1, device=device)
        for _ in range(world):
            O, m, l = blockwise_update(O, m, l, Q, K, V)
        return O / l

    for _ in range(WARMUP):
        ring_attention()
    t_total = timed(ring_attention)
    t_comp = timed(compute_only)

    if rank == 0:
        row = [world, S_TOTAL, s_local, err, round(t_comp, 3), round(t_total - t_comp, 3),
               round(t_total, 3)]
        newfile = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["cp", "s_total", "s_local", "max_err", "compute_ms", "comm_ms", "total_ms"])
            w.writerow(row)
        print("ROW:", row, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

</details>

