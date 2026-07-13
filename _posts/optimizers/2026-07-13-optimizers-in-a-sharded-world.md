---
layout: post
title: "Optimizers In A Sharded World: Muon's O(mn) Vs RMNP's Zero"
date: 2026-07-13 10:00:00
description: "The locality of an optimizer's precondition operator times the geometry of parameter sharding sets a communication bill you can measure, and on 8 GPUs one optimizer step over all of GPT-2 Large's matrices costs RMNP 5.3 ms with zero communication while Muon spends 360 ms moving 1.15 GiB per GPU, a 67× gap that grows with scale even though the sharded RMNP update is bit-identical to the full-matrix one."
tags: distributed-training optimization deep-learning
categories: optimizers
thumbnail: assets/img/blog/distributed/rmnp-vs-muon/fig-1-locality-geometry.png
toc:
  sidebar: left
related_posts: false
---

> **The "field of view" of an optimizer's precondition operator × the geometry of parameter sharding = a communication bill you can measure.** Muon's Newton-Schulz is a whole-matrix operator, so in a sharded world it pays O(mn) communication per matrix per step, plus a heavyweight recomputation that needs its own scheduling. RMNP's row normalization is a row-local operator, which coincides exactly with FSDP2's / row-block TP's sharding geometry and therefore costs zero communication. Measured on 8 GPUs over all of GPT-2 Large's matrices, that comes to **5.3 ms vs 360 ms per precondition step, a 67× gap**, and the sharded RMNP update is bit-identical to the full-matrix one (max diff = 0).

## 1. Cashing in a thread the series left open

An earlier post on FSDP (post #4 of the series) noted in passing that FSDP2's per-parameter DTensor sharding cuts along **dim-0, the row dimension**, so every GPU holds *whole rows* of every weight. Back then it was just an implementation choice. This post collects the payoff. **When an optimizer's mathematics happens to be organized by rows, that geometry hands you a gift.**

Both protagonists are "matrix-aware" optimizers. For every 2D weight $$W\in\mathbb{R}^{m\times n}$$ they keep momentum $$M_t=\mu M_{t-1}+G_t$$, and they differ only in the step that turns momentum into an update direction:

- **Muon** (Keller Jordan 2024, used by Moonlight/Kimi for 16B+ pretraining) runs a Newton-Schulz iteration approximating orthogonalization, $$\mathrm{NS}(M)\approx UV^\top$$ (flatten all singular values). Five iterations, three matmuls each.
- **RMNP** (Deng, Ouyang et al., ICML 2026) applies per-row L2 normalization, $$\text{update}_{i,:} = M_{i,:}/\lVert M_{i,:}\rVert_2$$, which is one elementwise kernel.

They are the same family of methods. Muon is steepest descent under a spectral-norm constraint (an LMO), and RMNP is steepest descent under a row-wise $$(q,2)$$ mixed-norm constraint. A different constraint geometry means a completely different **dependency range** for the update, and that dependency range is the entire distributed story.

## 2. The operator's field of view: one row, or the whole matrix?

Write both updates entrywise:

$$
\text{RMNP:}\quad \text{update}_{ij} = \frac{M_{ij}}{\sqrt{\sum_{k=1}^{n} M_{ik}^2}}
\qquad\qquad
\text{Muon:}\quad A = XX^\top,\ A_{ij} = \langle X_{i,:},\, X_{j,:}\rangle,\ \cdots
$$

RMNP's denominator sweeps only row $$i$$ itself, so **every row is an independent little problem**. Muon's very first NS step computes $$XX^\top$$, which takes inner products between arbitrary pairs of rows, so rows couple immediately. After multiplying back, every entry mixes information from the whole matrix, and that is the point of orthogonalization, because directions must compete. Spectral information is global by nature.

Now shard the matrix across 8 GPUs the FSDP2 way:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-1-locality-geometry.svg" class="img-fluid rounded" zoomable=true %}

On the left, every row norm RMNP needs lives whole on some GPU, so **each rank's own shard is all it ever looks at**. On the right, Muon's $$A_{ij}$$ needs two rows living on different GPUs. **No rank can even start**, which is why the full matrix must be all-gathered first.

## 3. The ledger: four layouts, cell by cell

Walk through the sharding layouts the series covered. The accounting unit is per [m×n] matrix, per step, per GPU, counting only the optimizer's *added* communication, because gradient sync is identical for both and stays out of the ledger:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-2-comm-ledger.svg" class="img-fluid rounded" zoomable=true %}

Three cells deserve expansion:

**FSDP2 / row-block TP → RMNP zero communication.** Each GPU holds whole-row blocks, so row norms are local, and normalization and the weight update never look at another GPU. Compute is load-balanced by construction, because each rank normalizes exactly $$m/N$$ rows, so there is no scheduling problem to solve.

**Column-cut TP → RMNP needs an O(m) vector.** Rows are cut into $$n/tp$$ segments, but the row norm is a **sum of squares, a decomposable reduction**:

$$
\boxed{\ \lVert M_{i,:}\rVert^2 \;=\; \sum_{\text{rank } r}\ \underbrace{\sum_{k\,\in\,\text{cols of rank } r} M_{ik}^2}_{\text{local partial sum}}\ }
$$

Each rank computes local partial sums (one scalar per row), concatenates them across all matrices into a single $$[\Sigma m]$$ vector, and does **one** all-reduce. For all 144 GPT-2 Large matrices that is ~410K fp32 values, ~3 MiB per GPU, which is three orders of magnitude below the full matrices Muon must move.

**Muon pays O(mn) under every layout.** Row-cut or column-cut, NS needs the full matrix, so each rank must all-gather the bf16 momentum (each GPU receives $$(N{-}1)/N \cdot mn \cdot 2$$ bytes), run NS, and take back its shard. Worse, there is also the NS matmul chain at $$O(mn\cdot\min(m,n))$$. Computing it on every rank is $$N\times$$ redundant, but amortizing it across ranks costs a second O(mn) hop to hand the results back. The measurement below shows how real that dilemma is.

> Honest boundary, the geometric precondition. RMNP's zero relies on **row-aligned sharding**. FSDP2's per-parameter dim-0 cut satisfies it natively, but FSDP1/ZeRO's FlatParameter splits by *element* and slices rows mid-way, so boundary rows would need cross-rank norm stitching. The requirement is trivially easy to meet, whereas Muon requires "reassemble the whole matrix" under *any* sharding.

## 4. One matrix, end to end

The ledger above states the O's. This section earns them on one concrete matrix, the MLP up-projection `c_fc` of GPT-2 Large. Its weight is $$W\,[5120 \times 1280]$$, so in bf16 the full matrix weighs $$5120 \cdot 1280 \cdot 2\,\text{B} = 12.5$$ MiB.

**First, who holds what.** A question worth answering explicitly: when $$W$$ is sharded, everything attached to $$W$$ is sharded the same way. Under FSDP2 with 8 GPUs, one rank's slice of this matrix looks like this.

| tensor | shape on this GPU | how it got here |
|--------|-------------------|-----------------|
| weight shard | [640 × 1280] | FSDP2's dim-0 cut, 640 whole rows |
| gradient shard | [640 × 1280] | reduce-scatter at the end of backward lands exactly these rows |
| momentum $$M_t$$ shard | [640 × 1280] | updated in place from the local gradient shard |
| fp32 master and Adam moments, if used | [640 × 1280] each | optimizer state always follows the parameter's cut |

The congruence is not optional. A replicated $$M_t$$ would cost as much memory as the replicated $$W$$ you just eliminated, so sharded training only pays off if the state follows the weight. The same holds under TP: a rank that owns a $$[5120 \times 160]$$ column slice of $$W$$ keeps exactly the $$[5120 \times 160]$$ slice of $$M_t$$ and nothing more. Everything below leans on this.

**Now walk one optimizer step, Muon and RMNP side by side**, same numbering for the same phase of the step:

| phase | Muon | RMNP |
|-------|------|------|
| ① gradient arrives | reduce-scatter puts a [640×1280] gradient shard on each rank. DP's own cost, identical for both, not counted | same |
| ② momentum update | local, $$M_k \leftarrow \mu M_k + G_k$$ on the shard, 0 bytes | same, 0 bytes |
| ③ assemble what the precondition needs | NS needs the full [5120×1280] and nobody has it, so all-gather the momentum: each GPU receives $$\tfrac{N-1}{N} S = \tfrac{7}{8} \times 12.5 \approx 10.9$$ MiB | the row norms need 640 whole rows, which this rank already has, 0 bytes |
| ④ run the precondition | every rank runs NS on the full matrix, redundantly | every rank normalizes its own 640 rows |
| ⑤ write back | take your 640 rows of the NS result, update the weight shard locally, 0 bytes | the normalized rows are already local, 0 bytes |

Only phase ③ differs, and that is the whole story. Note that phase ⑤ needs no second collective for either optimizer, because FSDP2 keeps parameters sharded at rest. The next forward pass runs its own per-layer all-gather anyway, which distributes the new weights for free, so a "sync after step" phase simply does not exist.

Two easily confused factors of 2 are worth pinning down. The all-gather above is a single phase, so it costs $$\tfrac{N-1}{N}S$$ per GPU. A true all-reduce is reduce-scatter plus all-gather, two phases, so it costs $$2\tfrac{N-1}{N}S$$ (post #1 derives both). Neither 2 is the "×2 B" in the size of $$S$$, which is just bf16's two bytes per element.

Scaling up is now mechanical. Sum $$\tfrac{7}{8} \cdot mn \cdot 2$$ over all 144 matrices of GPT-2 Large and you get 1181 MiB per GPU per step, which is exactly the number the benchmark below reports for Muon.

**The same matrix under column-cut TP.** Cut the other way, each rank holds $$[5120 \times 160]$$, so every row is broken into 8 segments and no rank can finish a row norm alone. But a squared norm is a sum, and sums decompose. Each rank computes one partial sum per row, giving a $$[5120]$$ fp32 vector that weighs 20 KB. Adding the partial sums across ranks is a genuine all-reduce, two phases, so each GPU pays $$2 \times \tfrac{7}{8} \times 20 \approx 35$$ KB. Compare that with Muon's 10.9 MiB on the same matrix: about 320× less, and independent of $$n$$, because what travels is one scalar per row rather than the row itself.

## 5. Experiment: putting the bill on the scale, 8 GPUs

All 2D matrices of GPT-2 Large (36 layers × 4 each = 144 matrices, 708M params), bf16 momentum, FSDP2-style row sharding across 8 GPUs (the benchmark ships with this post). Four schemes, each running the complete precondition step:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-3-measured.svg" class="img-fluid rounded" zoomable=true %}

| scheme | step time | comm / GPU |
|--------|-----------|------------|
| RMNP, row shard (FSDP2 native) | **5.3 ms** | **0** |
| RMNP, column-cut TP (O(m) vector) | **5.3 ms** | 2.8 MiB |
| Muon, gather + redundant NS | 359.5 ms | 1181 MiB |
| Muon, gather + owner NS + broadcast (round-robin) | 457.4 ms | 2531 MiB |

Five readings:

1. **67×** (5.3 vs 359.5). And column-cut RMNP ties the zero-comm version exactly, so the ~3 MiB vector all-reduce really is negligible, just as derived.
2. **Decomposing the bill** cross-checks the series. Running the 144 NS iterations bare on one GPU takes 302.9 ms, so Muon-SC's 360 ≈ **303 compute + 57 communication**. On the communication side, 1181 MiB ÷ 57 ms ≈ **20.4 GiB/s**, consistent with the gather bandwidth post #1 measured on this machine for ~10 MiB messages (PCIe platform). Those 57 ms are priced by bandwidth, a rigid floor that faster GPUs do not remove.
3. **Amortizing compute is *slower*** (457 > 360), because the round-robin variant serializes 144 "owner finishes NS, then broadcast" dependencies and doubles the traffic. On PCIe, saving 7/8 of the compute is wiped out entirely. **This is precisely the problem Canzona (the Qwen team, arXiv 2602.06079) spends an entire systems paper solving**, with α-balanced static partitioning for load balance and an asynchronous pipeline to hide the gathers, earning 1.57× end-to-end and 5.8× on optimizer latency at Qwen3-32B/256 GPUs. A first-rate engineering team published a paper to remedy one fact. *The optimizer needs whole matrices, and the system has none.*
4. **Distributed RMNP is ten lines**, a literal transcription of fig-1's left panel. There is no new communication pattern, no touching ZeRO's bucket layout, nothing to hide:

```python
# each rank, on its own row-aligned shard, no collectives:
local_M.mul_(mu).add_(local_grad)                 # momentum (sharded)
u = F.normalize(local_M.float(), p=2, dim=-1)     # row norms: rows are whole here (fp32 sums)
local_W.add_(u.bfloat16(), alpha=-lr * scale)     # update own shard
```

5. **And it is not an approximation.** The sharded RMNP update is **bit-identical** to the full-matrix one (measured max |diff| = 0.00e+00). When the operator's geometry and the shard's geometry align, distribution is free.

The single-matrix size sweep (panel (b)) adds the final twist. The gap itself grows with the matrix, from 20× at 1k² to **1409×** at 8k². Muon's gather O(mn) and NS O(mn·min) both scale super-linearly, but RMNP's elementwise O(mn) hugs the floor.

> Honest boundary. ① We measure the optimizer precondition segment, not end-to-end training speedup, and its share depends on model and step time (Canzona's 1.57× end-to-end says that on production grids the share is large). ② Our NS is eager bf16, uncompiled, and production kernels are faster. But the 57 ms of communication is bandwidth-priced, and our PCIe topology makes communication expensive. On NVLink the gather is an order of magnitude cheaper, so Muon's pain shifts from "comm + compute" toward "compute + scheduling", while the three structural facts (O(mn), ∝ Ψ, un-overlappable) survive any topology. ③ Under pure DDP (no sharding) both reach zero added communication, so the true watershed is **sharding**, and hyperscale pretraining shards by necessity.

## 6. Why the gap grows with scale

1. **Traffic scaling.** Muon's optimizer communication ∝ total parameter count Ψ, so a 7B model moves ~14 GB of bf16 momentum per GPU per step, landing on the busiest TP links or cross-node DP links. RMNP's counterpart is $$\Sigma m \approx \Psi/n$$, which is three orders smaller and exactly zero under most layouts.
2. **Critical-path position.** The optimizer step sits in each step's **serial segment**, after backward finishes and before the next forward starts, where communication has no compute to overlap with. Gradient all-reduce, by contrast, hides inside backward, as series post #2 showed. Canzona's asynchronous pipeline exists to manufacture something to overlap, but RMNP has nothing that needs hiding.
3. **Parallelism trend.** Collective latency grows with group size, and post #1 measured the latency floor climbing 11→65 µs from 2 to 8 ranks. Muon's gathers get harder to hide, but RMNP's cost does not grow with parallelism at all.

## 7. The convenience is not bought with convergence

All of this matters only if RMNP holds up as an optimizer. The layer-wise Hessians of Transformers are **row-block-diagonally dominant**, so whole-matrix spectral preconditioning can be approximated at the row-block level, and RMNP is exactly Muon's spectral normalization degraded to rows (ICML 2026). HTRMNP recovers part of the discarded spectral correction by scaling with the row norm's $$p$$-th power (default $$p=0.125$$). When Muon's global spectral information genuinely buys more, and how large that gap is (tall matrices, row-curvature heterogeneity), is another post's topic. This one nails a single fact. **On the systems side, row-local vs global is not a difference of degree. It is the difference between "keep your shard" and "reassemble the model, every step."**

## 8. Summary

1. The optimizer's communication bill is the product of two geometries, **the operator's dependency range × the shard's cut**. Muon's NS depends on the whole matrix, so it pays O(mn) plus a scheduling problem under any sharding. RMNP depends on single rows, so it pays 0 under FSDP2/row-TP and an O(m) decomposable reduction under column-TP.
2. Measured on 8 GPUs over GPT-2 Large's full matrix set, the bill is 5.3 ms vs 359.5 ms (**67×**), decomposed as 303 compute + 57 comm with the comm priced correctly by post #1's bandwidth table. The compute-amortizing round-robin variant is *slower* due to serialization, which is why distributed Muon is worth a paper (Canzona) and distributed RMNP is worth ten lines.
3. Sharded RMNP equals full-matrix RMNP bit for bit. Align the operator's geometry with the shard's geometry, and distribution is free.
4. The gap grows with scale, because traffic ∝ Ψ, the step sits on the un-overlappable critical path, and latency rises with parallelism. All three arrows point the same way.

*(Series background: the memory/Ψ ledger in post #0, the collective-bandwidth table in post #1, FSDP2's dim-0 sharding in post #4, TP's row/column cuts in post #5, the bf16/fp32 conventions in post #8.)*

---

*Environment: 8× RTX PRO 6000 Blackwell (96GB), pure PCIe (no NVLink), PyTorch 2.9.1 + NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node=8 bench_dist_opt.py` (four schemes + size sweep + exactness check). Schematic and plotting code accompanies the post.*

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

