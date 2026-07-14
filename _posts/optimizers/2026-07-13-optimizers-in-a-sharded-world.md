---
layout: post
title: "Optimizers In A Sharded World: Muon's O(mn) Vs RMNP's Zero"
date: 2026-07-13 10:00:00
description: "How the data required by an optimizer interacts with parameter sharding: RMNP uses row-local normalization and needs no communication under row sharding, while Muon gathers full matrices, producing a measured 67× preconditioning-time gap on 8 GPUs."
tags: distributed-training optimization deep-learning
categories: optimizers
thumbnail: assets/img/blog/distributed/rmnp-vs-muon/fig-1-locality-geometry.png
toc:
  sidebar: left
related_posts: false
---

> An optimizer needs communication only when its preconditioner requires data outside the local parameter shard. Muon's Newton–Schulz orthogonalization depends on the full matrix, so sharded training must gather O(mn) values per matrix and schedule expensive matrix multiplications. RMNP normalizes each row independently. Its computation aligns with the sharding used by FSDP2 and row-block tensor parallelism, so it requires no optimizer-specific communication. Across all GPT-2 Large matrices on 8 GPUs, the measured preconditioning times are **5.3 ms for RMNP and 359.5 ms for Muon, a 67× difference**. Sharded and unsharded RMNP also produce identical updates (maximum difference 0).

## 1. Why row-aligned sharding matters

Post #4 showed that FSDP2 shards each parameter along **dim-0**, so every GPU stores complete rows of each weight matrix. This layout is especially useful for an optimizer whose preconditioner also acts independently on rows: each rank can update its local rows without first reconstructing the full matrix.

Muon and RMNP are both matrix-aware optimizers. Following the theoretical convention in the RMNP paper, let the momentum matrix be $V_t\in\mathbb{R}^{m\times n}$ with $m<n$, and let $V_t=\mu V_{t-1}+G_t$. The communication arguments later in this post depend only on which rows are stored locally, not on the inequality $m<n$. The optimizers differ in how they transform $V_t$ into an update direction:

- **Muon**, introduced by Keller Jordan in 2024 and used in Moonlight/Kimi pretraining for models with at least 16B parameters, applies Newton–Schulz iterations to approximate the orthogonalized direction $D_t^{\mathrm{Muon}}=(V_tV_t^\top)^{-1/2}V_t$. A common implementation uses five iterations with three matrix multiplications per iteration.
- **RMNP** (Deng, Ouyang et al., ICML 2026) replaces the full orthogonalization with row-wise L2 normalization: $D_{t,i:}^{\mathrm{RMNP}}=V_{t,i:}/\lVert V_{t,i:}\rVert_2$. This requires only row reductions and element-wise scaling.

RMNP is motivated by the structure of Transformer curvature, not only by a different norm. Prior work finds that Transformer layer-wise Hessians are row-wise block-diagonally dominant: curvature interactions within one row are much stronger than interactions across rows. The RMNP paper also measures strong diagonal dominance in the Muon Gram matrix $V_tV_t^\top\in\mathbb{R}^{m\times m}$.

Muon uses the full Gram matrix:

$$
D_t^{\mathrm{Muon}}=(V_tV_t^\top)^{-1/2}V_t.
$$

If cross-row terms are small, retain only the diagonal entries of $V_tV_t^\top$:

$$
D_t^{\mathrm{RMNP}}=\operatorname{diag}(V_tV_t^\top)^{-1/2}V_t,
\qquad
D_{t,i:}^{\mathrm{RMNP}}=\frac{V_{t,i:}}{\lVert V_{t,i:}\rVert_2}.
$$

When $V_tV_t^\top$ is exactly diagonal, RMNP and Muon produce the same direction. As the off-diagonal terms approach zero, the two directions become asymptotically equivalent. For finite diagonal dominance, RMNP is a structured approximation: it keeps the dominant row-wise curvature and discards weaker cross-row interactions. This reduces the preconditioning cost from $O(mn\min(m,n))$ to $O(mn)$.

The linear-minimization-oracle view is complementary. Muon is steepest descent under a spectral-norm constraint, while RMNP is steepest descent under a row-wise $(q,2)$ mixed-norm constraint. That view explains their different constraint geometries. The diagonal-dominance argument explains why the simpler row-wise operator can approximate Muon's orthogonalization in Transformers.

## 2. How much of the matrix does each update require?

The dependency difference is visible from the update formulas:

$$
\text{RMNP:}\quad D_{t,ij} = \frac{V_{t,ij}}{\sqrt{\sum_{k=1}^{n} V_{t,ik}^2}}
\qquad\qquad
\text{Muon:}\quad A = V_tV_t^\top,\ A_{ij} = \langle V_{t,i,:},\, V_{t,j,:}\rangle,\ \cdots
$$

RMNP computes the denominator for row $i$ from that row alone, so each row can be processed independently. Muon's first Newton–Schulz step forms $V_tV_t^\top$, which contains inner products between every pair of rows. The resulting orthogonalization couples all rows and therefore requires global matrix information.

Now consider FSDP2 row sharding across 8 GPUs:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-1-locality-geometry.svg" class="img-fluid rounded" zoomable=true %}

For RMNP, every required row norm can be computed from one local shard. Muon's $A_{ij}$ may depend on rows stored on different GPUs, so no rank can form the full Gram matrix from local data alone. The momentum matrix must be all-gathered before orthogonalization.

## 3. Communication under four sharding layouts

The following comparison reports optimizer-specific communication per matrix, per step and per GPU. Gradient synchronization is identical for both optimizers and is excluded:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-2-comm-ledger.svg" class="img-fluid rounded" zoomable=true %}

Three cases need further explanation:

**FSDP2 and row-block TP require no RMNP communication.** Each GPU holds complete rows, so it can compute row norms and update its local weights without accessing another rank. With balanced row blocks, each rank processes about $m/N$ rows.

**Column-cut TP requires an O(m) reduction for RMNP.** Each row is divided into $n/tp$ segments, but its squared norm is a sum of local squared norms:

$$
\boxed{\ \lVert M_{i,:}\rVert^2 \;=\; \sum_{\text{rank } r}\ \underbrace{\sum_{k\,\in\,\text{cols of rank } r} M_{ik}^2}_{\text{local partial sum}}\ }
$$

Each rank computes one partial sum per row, concatenates these values across all matrices into a $[\Sigma m]$ vector, and performs one all-reduce. Across all 144 GPT-2 Large matrices, the vector contains about 410K fp32 values and produces about 3 MiB of communication per GPU. This is roughly three orders of magnitude smaller than gathering the full matrices.

**Muon requires O(mn) communication under either row or column sharding.** Newton–Schulz needs the full momentum matrix, so each rank receives $(N{-}1)mn/N$ bf16 elements through all-gather before running the $O(mn\min(m,n))$ matrix-multiplication chain. Running orthogonalization on every rank duplicates compute by a factor of $N$. Assigning each matrix to one rank avoids that duplicate compute but requires another O(mn) transfer to distribute the result.

> **Required layout.** Zero-communication RMNP requires **row-aligned sharding**. FSDP2's per-parameter dim-0 partition provides complete rows. FSDP1 and ZeRO FlatParameters split by element count and may divide a row across ranks, so boundary rows need a small cross-rank norm reduction. Muon still needs the full matrix under either layout.

## 4. One matrix, end to end

Consider the GPT-2 Large MLP up-projection `c_fc`, stored as $W\,[5120 \times 1280]$. In bf16, the full matrix occupies $5120 \cdot 1280 \cdot 2\,\text{B} = 12.5$ MiB.

When $W$ is sharded, its gradients and optimizer states use the same partition. Under FSDP2 with 8 GPUs, each rank stores:

| tensor | shape on this GPU | how it got here |
|--------|-------------------|-----------------|
| weight shard | [640 × 1280] | FSDP2's dim-0 cut, 640 whole rows |
| gradient shard | [640 × 1280] | reduce-scatter at the end of backward lands exactly these rows |
| momentum $V_t$ shard | [640 × 1280] | updated in place from the local gradient shard |
| fp32 master and Adam moments, if used | [640 × 1280] each | optimizer state always follows the parameter's cut |

Replicating $V_t$ would restore much of the memory removed by parameter sharding, so optimizer states must follow the same partition as their parameters. Under column-cut TP, a rank that owns a $[5120 \times 160]$ slice of $W$ stores the matching slice of $V_t$.

One optimizer step then proceeds as follows:

| phase | Muon | RMNP |
|-------|------|------|
| ① gradient arrives | reduce-scatter puts a [640×1280] gradient shard on each rank. DP's own cost, identical for both, not counted | same |
| ② momentum update | local, $V_k \leftarrow \mu V_k + G_k$ on the shard, 0 bytes | same, 0 bytes |
| ③ assemble what the precondition needs | NS needs the full [5120×1280] and nobody has it, so all-gather the momentum: each GPU receives $$\tfrac{N-1}{N} S = \tfrac{7}{8} \times 12.5 \approx 10.9$$ MiB | the row norms need 640 whole rows, which this rank already has, 0 bytes |
| ④ run the precondition | every rank runs NS on the full matrix, redundantly | every rank normalizes its own 640 rows |
| ⑤ write back | take your 640 rows of the NS result, update the weight shard locally, 0 bytes | the normalized rows are already local, 0 bytes |

Phase ③ creates the optimizer-specific difference. Phase ⑤ needs no additional collective because FSDP2 keeps parameters sharded between operations. The next forward pass already all-gathers each layer's current parameters, so the optimizer does not need a separate post-step synchronization.

The two factors of 2 have different sources. All-gather is one communication phase and sends $\tfrac{N-1}{N}S$ bytes per GPU. All-reduce contains reduce-scatter and all-gather and sends $2\tfrac{N-1}{N}S$ bytes per GPU, as derived in post #1. Separately, bf16 uses 2 bytes per element when converting element counts into $S$.

Summing $\tfrac{7}{8}mn\cdot 2$ bytes across all 144 GPT-2 Large matrices gives 1181 MiB per GPU and optimizer step, matching the Muon benchmark below.

**Under column-cut TP,** each rank stores $[5120 \times 160]$, so every row spans 8 ranks. Each rank computes one squared-norm partial sum per row, producing a 20 KB fp32 vector of length 5120. All-reducing this vector sends about $2 \times \tfrac{7}{8} \times 20 \approx 35$ KB per GPU. Muon gathers 10.9 MiB for the same matrix, about 320× more. RMNP communicates one scalar per row rather than every element.

## 5. Experiment: preconditioning all GPT-2 Large matrices on 8 GPUs

The benchmark includes all 144 two-dimensional matrices from GPT-2 Large, covering 708M parameters across 36 layers. Momentum uses bf16, and matrices are row-sharded across 8 GPUs in the FSDP2 style. Each of the four schemes runs a complete preconditioning step:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-3-measured.svg" class="img-fluid rounded" zoomable=true %}

| scheme | step time | comm / GPU |
|--------|-----------|------------|
| RMNP, row shard (FSDP2 native) | **5.3 ms** | **0** |
| RMNP, column-cut TP (O(m) vector) | **5.3 ms** | 2.8 MiB |
| Muon, gather + redundant NS | 359.5 ms | 1181 MiB |
| Muon, gather + owner NS + broadcast (round-robin) | 457.4 ms | 2531 MiB |

The measurements show five results:

1. **Row-sharded RMNP is 67× faster than gather-and-recompute Muon:** 5.3 versus 359.5 ms. Column-cut RMNP also takes 5.3 ms, showing that its 2.8 MiB vector all-reduce is negligible in this experiment.
2. **Muon's 359.5 ms consists of about 302.9 ms of Newton–Schulz compute and 57 ms of communication.** Moving 1181 MiB in 57 ms gives about 20.4 GiB/s, consistent with the all-gather bandwidth measured for similar message sizes in post #1. Faster matrix multiplication alone cannot remove this communication time.
3. **Assigning each matrix to one owner is slower, at 457.4 ms.** The round-robin design serializes 144 dependencies in which an owner completes Newton–Schulz and then broadcasts the result. It also increases communication to 2531 MiB per GPU. Canzona (arXiv 2602.06079) addresses this scheduling problem with balanced static assignment and an asynchronous pipeline, reporting 1.57× end-to-end speedup and 5.8× lower optimizer latency for Qwen3-32B on 256 GPUs. These mechanisms are needed because Muon requires complete matrices that are not locally available under sharding.
4. **Row-sharded RMNP needs only local operations.** It does not add a communication pattern or modify ZeRO's bucket layout:

```python
# Each rank updates its own row-aligned shard. No collectives are needed.
local_M.mul_(mu).add_(local_grad)                 # momentum (sharded)
u = F.normalize(local_M.float(), p=2, dim=-1)     # row norms: rows are whole here (fp32 sums)
local_W.add_(u.bfloat16(), alpha=-lr * scale)     # update own shard
```

5. **Sharding does not approximate RMNP.** The row-sharded update is identical to the full-matrix RMNP update, with a measured maximum difference of 0.00e+00. Every row receives the same normalization because each complete row remains local.

In the single-matrix size sweep in panel (b), the time ratio grows from 20× at 1k² to **1409×** at 8k². Muon combines O(mn) communication with O(mn·min(m,n)) Newton–Schulz compute, while RMNP remains O(mn).

> **Boundary of the experiment.** We measure only the preconditioning segment, not end-to-end training. Its contribution depends on the model and total step time. Our Newton–Schulz implementation uses eager bf16 without compilation, so optimized production kernels can reduce its compute time. The 57 ms communication component is specific to this PCIe topology and would be much smaller on NVLink, although the O(mn) volume and scheduling dependency remain. Under unsharded DDP, both optimizers require no additional communication. The difference appears when model state is sharded.

## 6. Why the gap grows with scale

1. **Communication volume.** Muon's optimizer communication scales with total parameter count Ψ. A 7B model would gather roughly 14 GB of bf16 momentum per GPU and step. Column-sharded RMNP communicates only $\Sigma m \approx \Psi/n$ scalars, and row-sharded RMNP communicates none.
2. **Critical-path position.** The optimizer step runs after backward and before the next forward, leaving little model computation with which to overlap communication. Gradient all-reduce can overlap with backward, as post #2 showed. Distributed Muon therefore needs additional scheduling to create overlap, while row-sharded RMNP adds no optimizer communication.
3. **Group size.** Post #1 measured the collective latency floor increasing from 11 to 65 µs between 2 and 8 ranks. Larger groups make Muon's gathers harder to schedule, while row-sharded RMNP remains local.

## 7. Why the row-wise approximation can preserve optimization quality

The systems advantage matters only if the row-wise approximation remains effective. The RMNP paper motivates it from the observed **row-wise block-diagonal dominance** of Transformer layer Hessians and verifies diagonal dominance in $V_tV_t^\top$ across GPT-2 and LLaMA scales. In the diagonal limit, RMNP and Muon give the same preconditioned direction. With finite off-diagonal terms, RMNP keeps the dominant row-wise contribution while omitting cross-row coupling. HTRMNP can recover part of that discarded correction by scaling with a power of the row norm, using $p=0.125$ by default. Muon's global spectral information may still help when cross-row curvature is strong. The systems distinction nevertheless remains clear: a row-local update can use the existing shard, while a whole-matrix update must reconstruct global information.

## 8. Summary

1. RMNP is a structured approximation to Muon motivated by row-wise block-diagonal dominance. For $V_t\in\mathbb{R}^{m\times n}$ with $m<n$, Muon uses $(V_tV_t^\top)^{-1/2}V_t$, while RMNP keeps only the diagonal of $V_tV_t^\top$. The two are equal when the Gram matrix is diagonal and become asymptotically equivalent as its off-diagonal terms vanish.
2. The optimizer's communication depends on whether its required data is local to each shard. Muon needs complete matrices and adds O(mn) communication plus scheduling. RMNP needs one row at a time, so it adds no communication under FSDP2 or row-block TP and an O(m) reduction under column-cut TP.
3. Across all GPT-2 Large matrices on 8 GPUs, RMNP takes 5.3 ms and Muon takes 359.5 ms, a **67× difference**. Muon's time consists of about 303 ms of compute and 57 ms of communication. The owner-based variant is slower because it serializes matrix assignments and broadcasts.
4. Row-sharded RMNP produces the same update as unsharded RMNP. Its advantage grows with model size because Muon's communication scales with Ψ and lies on the optimizer critical path, while RMNP remains local under row-aligned sharding.

*(Series background: the memory/Ψ ledger in post #0, the collective-bandwidth table in post #1, FSDP2's dim-0 sharding in post #4, TP's row/column cuts in post #5, the bf16/fp32 conventions in post #8.)*

---

*Environment: 8× RTX PRO 6000 Blackwell (96 GB), PCIe without NVLink, PyTorch 2.9.1 and NCCL 2.27.5. Reproduce with `torchrun --standalone --nproc_per_node=8 bench_dist_opt.py` for the four schemes, size sweep and exactness check. Schematic and plotting code accompany the post.*

*All benchmark scripts, schematic generators, plotting code and raw result CSVs for this post live in [assets/blog/code/rmnp-vs-muon](https://github.com/Dominator-Index/Dominator-Index.github.io/tree/main/assets/blog/code/rmnp-vs-muon).*

---

## Appendix: The Code That Ran

Every number in this post comes from the scripts below, embedded verbatim. Plotting and schematic code plus the raw result CSVs live in the folder linked above.

<details markdown="1">
<summary><code>bench_dist_opt.py</code></summary>

```python
"""
Measured communication ledger of distributed optimizer preconditioning (standalone
RMNP vs Muon post).

Setup: FSDP2-style dim-0 (row) sharding, all 2D matrices of GPT-2 Large (36 layers
x 4 each, hidden 1280), 8 GPUs. Only the precondition segment of the optimizer step
is measured (momentum update + normalization/NS + writing back params). Gradient
synchronization is excluded (identical for both).

Four schemes:
  rmnp_local    RMNP under row sharding: each GPU normalizes its own complete row block, 0 communication
  rmnp_colcut   RMNP under column-cut TP: local partial sums of squares + one [sum_m] vector all-reduce
  muon_sc       Muon synchronous compute: all-gather momentum into the full matrix, every GPU redundantly runs NS, take back its row block
  muon_rr       Muon round-robin: all-gather + only the owner runs NS + broadcast the update
                (2x communication, 1/N compute, the "amortized" route in the Moonlight/Canzona sense)

Also: a square-matrix size sweep (1k/2k/4k/8k) and a numerical check that sharded
RMNP == full-matrix RMNP.

Usage: torchrun --standalone --nproc_per_node=8 bench_dist_opt.py --out ../results
"""

import argparse
import csv
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

# GPT-2 Large: hidden 1280, 36 layers, 4 2D matrices per layer ([out, in])
H, L = 1280, 36
LAYER_SHAPES = [(3 * H, H), (H, H), (4 * H, H), (H, 4 * H)]  # qkv/attn.proj/c_fc/c_proj
SWEEP_SIZES = [1024, 2048, 4096, 8192]
NS_STEPS = 5
MU = 0.95
WARMUP, STEPS = 3, 10


def newtonschulz5(G, steps=NS_STEPS, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Bench:
    """Sharded state for one set of GPT-2 Large matrices (row-sharded and column-sharded copies, bf16)."""

    def __init__(self, device, world, rank):
        self.device, self.world, self.rank = device, world, rank
        self.shapes = LAYER_SHAPES * L
        g = torch.Generator(device="cpu").manual_seed(1337)
        # row sharding (FSDP2): [m/N, n] per GPU, column sharding (TP RowLinear): [m, n/N] per GPU
        self.row_W, self.row_M, self.row_G = [], [], []
        self.col_M, self.col_G = [], []
        for (m, n) in self.shapes:
            self.row_W.append(torch.randn(m // world, n, generator=g).bfloat16().to(device))
            self.row_M.append(torch.randn(m // world, n, generator=g).bfloat16().to(device))
            self.row_G.append(torch.randn(m // world, n, generator=g).bfloat16().to(device))
            self.col_M.append(torch.randn(m, n // world, generator=g).bfloat16().to(device))
            self.col_G.append(torch.randn(m, n // world, generator=g).bfloat16().to(device))
        self.total_params = sum(m * n for (m, n) in self.shapes)

    # ---- the schemes: each runs one full optimizer precondition step ----
    def rmnp_local(self):
        for W, M, G in zip(self.row_W, self.row_M, self.row_G):
            M.mul_(MU).add_(G)
            u = F.normalize(M.float(), p=2, dim=-1)  # within-row sums run in fp32 (the part 08 principle)
            scale = max(1.0, (M.shape[0] * self.world) / M.shape[1]) ** 0.5
            W.add_(u.bfloat16(), alpha=-3e-4 * scale)

    def rmnp_colcut(self):
        # local partial sums of squares, concat all matrices into one [sum_m] vector, one all-reduce
        partials = []
        for M, G in zip(self.col_M, self.col_G):
            M.mul_(MU).add_(G)
            partials.append(M.float().pow(2).sum(dim=-1))
        flat = torch.cat(partials)                      # [sum_m] about 410k fp32 values
        dist.all_reduce(flat)                           # the only communication
        idx = 0
        for M in self.col_M:
            m = M.shape[0]
            norms = flat[idx:idx + m].sqrt().clamp_min(1e-7)
            M.div_(norms.bfloat16().unsqueeze(-1))      # reused in place as the update
            idx += m

    def muon_sc(self):
        for i, (W, M, G) in enumerate(zip(self.row_W, self.row_M, self.row_G)):
            M.mul_(MU).add_(G)
            m, n = self.shapes[i]
            full = torch.empty(m, n, dtype=torch.bfloat16, device=self.device)
            dist.all_gather_into_tensor(full, M.contiguous())   # O(mn) communication
            u = newtonschulz5(full)                             # every GPU computes redundantly
            rows = slice(self.rank * (m // self.world), (self.rank + 1) * (m // self.world))
            scale = max(1.0, m / n) ** 0.5
            W.add_(u[rows], alpha=-3e-4 * scale)

    def muon_rr(self):
        for i, (W, M, G) in enumerate(zip(self.row_W, self.row_M, self.row_G)):
            M.mul_(MU).add_(G)
            m, n = self.shapes[i]
            owner = i % self.world
            full = torch.empty(m, n, dtype=torch.bfloat16, device=self.device)
            dist.all_gather_into_tensor(full, M.contiguous())   # O(mn)
            if self.rank == owner:
                full = newtonschulz5(full).contiguous()         # only the owner computes
            dist.broadcast(full, src=owner)                     # pay another O(mn)
            rows = slice(self.rank * (m // self.world), (self.rank + 1) * (m // self.world))
            scale = max(1.0, m / n) ** 0.5
            W.add_(full[rows], alpha=-3e-4 * scale)

    # ---- per-step communication volume per scheme (per GPU, bytes, ring/collective accounting, bf16=2B) ----
    def bytes_per_step(self, scheme):
        N = self.world
        if scheme == "rmnp_local":
            return 0
        if scheme == "rmnp_colcut":
            total_m = sum(m for (m, _) in self.shapes)
            return int(2 * (N - 1) / N * total_m * 4)           # fp32 vector all-reduce
        if scheme == "muon_sc":
            return int(sum((N - 1) / N * m * n * 2 for (m, n) in self.shapes))
        if scheme == "muon_rr":
            return int(sum(((N - 1) / N + 1) * m * n * 2 for (m, n) in self.shapes))


def time_fn(fn, device):
    for _ in range(WARMUP):
        fn()
    dist.barrier()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(STEPS):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / STEPS
    t = torch.tensor([ms], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)                    # take the slowest GPU
    return t.item()


def exactness_check(device, world, rank):
    """Row norms of sharded RMNP match full-matrix RMNP exactly (not an approximation)."""
    m, n = 512, 384
    g = torch.Generator(device="cpu").manual_seed(7)
    full = torch.randn(m, n, generator=g).to(device)
    ref = F.normalize(full, p=2, dim=-1)
    shard = full[rank * m // world:(rank + 1) * m // world]
    mine = F.normalize(shard, p=2, dim=-1)
    err = (mine - ref[rank * m // world:(rank + 1) * m // world]).abs().max().item()
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    err = exactness_check(device, world, rank)
    if rank == 0:
        print(f"[exactness] sharded vs full RMNP max |diff| = {err:.2e}", flush=True)

    bench = Bench(device, world, rank)
    if rank == 0:
        print(f"GPT-2 Large matrix set: {len(bench.shapes)} matrices, "
              f"{bench.total_params/1e6:.0f}M params", flush=True)

    rows = []
    for scheme in ["rmnp_local", "rmnp_colcut", "muon_sc", "muon_rr"]:
        ms = time_fn(getattr(bench, scheme), device)
        by = bench.bytes_per_step(scheme)
        rows.append([scheme, round(ms, 2), by, round(by / 2**20, 1)])
        if rank == 0:
            print(f"{scheme}: {ms:.2f} ms/step, comm {by/2**20:.1f} MiB/step/rank", flush=True)
    del bench
    torch.cuda.empty_cache()

    # ---- square-matrix size sweep: per-matrix precondition time ----
    sweep = []
    for s in SWEEP_SIZES:
        shard = torch.randn(s // world, s, device=device).bfloat16()
        full = torch.empty(s, s, dtype=torch.bfloat16, device=device)

        def rmnp_one():
            F.normalize(shard.float(), p=2, dim=-1)

        def muon_one():
            dist.all_gather_into_tensor(full, shard.contiguous())
            newtonschulz5(full)

        t_r = time_fn(rmnp_one, device)
        t_m = time_fn(muon_one, device)
        gather_bytes = int((world - 1) / world * s * s * 2)
        sweep.append([s, round(t_r, 3), round(t_m, 3), gather_bytes])
        if rank == 0:
            print(f"[{s}x{s}] rmnp {t_r:.3f} ms | muon gather+NS {t_m:.3f} ms", flush=True)
        del shard, full
        torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "dist_opt.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scheme", "ms_per_step", "bytes_per_step", "mib_per_step"])
            w.writerows(rows)
        with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["size", "rmnp_ms", "muon_ms", "gather_bytes"])
            w.writerows(sweep)
        with open(os.path.join(args.out, "exactness.txt"), "w") as f:
            f.write(f"sharded vs full RMNP max abs diff: {err:.3e}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

</details>

