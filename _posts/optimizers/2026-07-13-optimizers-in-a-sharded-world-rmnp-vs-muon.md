---
layout: post
title: "Optimizers in a Sharded World: Muon's O(mn) vs RMNP's Zero"
date: 2026-07-13 23:50:00
description: "The locality of an optimizer's precondition operator × the geometry of parameter sharding = a communication bill you can measure. On 8 GPUs, one optimizer step over all of GPT-2 Large's matrices costs RMNP 5.3 ms with zero communication and Muon 360 ms moving 1.15 GiB per GPU — a 67× gap that grows with scale, and the sharded RMNP update is bit-identical to the full-matrix one."
tags: distributed-training optimization deep-learning
categories: optimizers
thumbnail: assets/img/blog/distributed/rmnp-vs-muon/fig-1-locality-geometry.png
toc:
  sidebar: left
related_posts: false
---

> A standalone post (outside the numbering of **Distributed Training, Illustrated**, but freely drawing on its ledgers). One post, one idea: **the "field of view" of an optimizer's precondition operator × the geometry of parameter sharding = a communication bill you can measure.** Muon's Newton-Schulz is a whole-matrix operator — in a sharded world it pays O(mn) communication per matrix per step, plus a heavyweight recomputation that needs its own scheduling. RMNP's row normalization is a row-local operator — it coincides exactly with FSDP2's / row-block TP's sharding geometry: zero communication. Measured on 8 GPUs over all of GPT-2 Large's matrices: **5.3 ms vs 360 ms per precondition step, a 67× gap** — and the sharded RMNP update is bit-identical to the full-matrix one (max diff = 0).

## 1. Cashing in a thread the series left open

Post #4 of the series planted a sentence when introducing FSDP2: its per-parameter DTensor sharding cuts along **dim-0 — the row dimension** — so every GPU holds *whole rows* of every weight. Back then it was just an implementation choice. This post collects the payoff: **when an optimizer's mathematics happens to be organized by rows, that geometry hands you a gift.**

Both protagonists are "matrix-aware" optimizers. For every 2D weight $$W\in\mathbb{R}^{m\times n}$$ they keep momentum $$M_t=\mu M_{t-1}+G_t$$; they differ only in the step that turns momentum into an update direction:

- **Muon** (Keller Jordan 2024; used by Moonlight/Kimi for 16B+ pretraining): Newton-Schulz iteration approximating orthogonalization, $$\mathrm{NS}(M)\approx UV^\top$$ (flatten all singular values). Five iterations, three matmuls each;
- **RMNP** (Deng, Ouyang et al., ICML 2026): per-row L2 normalization, $$\text{update}_{i,:} = M_{i,:}/\lVert M_{i,:}\rVert_2$$ — one elementwise kernel.

They are the same family of methods: Muon is steepest descent under a spectral-norm constraint (an LMO), RMNP under a row-wise $$(q,2)$$ mixed-norm constraint. Different constraint geometry ⇒ a completely different **dependency range** for the update. That dependency range is the entire distributed story.

## 2. The operator's field of view: one row, or the whole matrix?

Write both updates entrywise:

$$
\text{RMNP:}\quad \text{update}_{ij} = \frac{M_{ij}}{\sqrt{\sum_{k=1}^{n} M_{ik}^2}}
\qquad\qquad
\text{Muon:}\quad A = XX^\top,\ A_{ij} = \langle X_{i,:},\, X_{j,:}\rangle,\ \cdots
$$

RMNP's denominator sweeps only row $$i$$ itself — **every row is an independent little problem**. Muon's very first NS step computes $$XX^\top$$: inner products between arbitrary pairs of rows, so rows couple immediately; after multiplying back, every entry mixes information from the whole matrix (which is the point of orthogonalization — directions must compete). Spectral information is global by nature.

Now shard the matrix across 8 GPUs the FSDP2 way:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-1-locality-geometry.svg" class="img-fluid rounded" zoomable=true %}

On the left, every row norm RMNP needs lives whole on some GPU — **each rank's own shard is all it ever looks at**. On the right, Muon's $$A_{ij}$$ needs two rows living on different GPUs — **no rank can even start**, so the full matrix must be all-gathered first.

## 3. The ledger: four layouts, cell by cell

Walk through the sharding layouts the series covered (accounting unit: per [m×n] matrix, per step, per GPU; only the optimizer's *added* communication — gradient sync is identical for both and excluded):

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-2-comm-ledger.svg" class="img-fluid rounded" zoomable=true %}

Three cells deserve expansion:

**FSDP2 / row-block TP → RMNP zero communication.** Each GPU holds whole-row blocks; row norms are local; normalization and the weight update never look at another GPU. Compute is load-balanced by construction — each rank normalizes exactly $$m/N$$ rows; there is no scheduling problem to solve.

**Column-cut TP → RMNP needs an O(m) vector.** Rows are cut into $$n/tp$$ segments, but the row norm is a **sum of squares — a decomposable reduction**:

$$
\boxed{\ \lVert M_{i,:}\rVert^2 \;=\; \sum_{\text{rank } r}\ \underbrace{\sum_{k\,\in\,\text{cols of rank } r} M_{ik}^2}_{\text{local partial sum}}\ }
$$

Each rank computes local partial sums (one scalar per row), concatenates them across all matrices into a single $$[\Sigma m]$$ vector, and does **one** all-reduce — for all 144 GPT-2 Large matrices that is ~410K fp32 values, ~3 MiB per GPU: three orders of magnitude below the full matrices Muon must move.

**Muon pays O(mn) under every layout.** Row-cut or column-cut, NS needs the full matrix: all-gather the bf16 momentum (each GPU receives $$(N{-}1)/N \cdot mn \cdot 2$$ bytes), run NS, take back your shard. Worse, there is also the NS matmul chain, $$O(mn\cdot\min(m,n))$$: computing it on every rank is $$N\times$$ redundant; amortizing it across ranks costs a second O(mn) hop to hand the results back. The measurement below shows how real that dilemma is.

> Honest boundary (the geometric precondition): RMNP's zero relies on **row-aligned sharding**. FSDP2's per-parameter dim-0 cut satisfies it natively; FSDP1/ZeRO's FlatParameter splits by *element*, slicing rows mid-way — boundary rows would need cross-rank norm stitching. The requirement is trivially easy to meet — whereas Muon requires "reassemble the whole matrix" under *any* sharding.

## 4. Experiment: putting the bill on the scale, 8 GPUs

All 2D matrices of GPT-2 Large (36 layers × 4 each = 144 matrices, 708M params), bf16 momentum, FSDP2-style row sharding across 8 GPUs (the benchmark ships with this post). Four schemes, each running the complete precondition step:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/rmnp-vs-muon/fig-3-measured.svg" class="img-fluid rounded" zoomable=true %}

| scheme | step time | comm / GPU |
|--------|-----------|------------|
| RMNP, row shard (FSDP2 native) | **5.3 ms** | **0** |
| RMNP, column-cut TP (O(m) vector) | **5.3 ms** | 2.8 MiB |
| Muon, gather + redundant NS | 359.5 ms | 1181 MiB |
| Muon, gather + owner NS + broadcast (round-robin) | 457.4 ms | 2531 MiB |

Five readings:

1. **67×** (5.3 vs 359.5). And column-cut RMNP ties the zero-comm version exactly — the ~3 MiB vector all-reduce really is, as derived, negligible;
2. **Decomposing the bill** (cross-checking the series): running the 144 NS iterations bare on one GPU takes 302.9 ms, so Muon-SC's 360 ≈ **303 compute + 57 communication**. On the communication side: 1181 MiB ÷ 57 ms ≈ **20.4 GiB/s**, consistent with the gather bandwidth post #1 measured on this machine for ~10 MiB messages (PCIe platform). Those 57 ms are priced by bandwidth — a rigid floor that faster GPUs do not remove;
3. **Amortizing compute is *slower*** (457 > 360): the round-robin variant serializes 144 "owner finishes NS, then broadcast" dependencies and doubles the traffic; on PCIe, saving 7/8 of the compute is wiped out entirely. **This is precisely the problem Canzona (the Qwen team, arXiv 2602.06079) spends an entire systems paper solving**: α-balanced static partitioning for load balance, an asynchronous pipeline to hide the gathers — earning 1.57× end-to-end and 5.8× on optimizer latency at Qwen3-32B/256 GPUs. A first-rate engineering team publishing a paper to remedy one fact: *the optimizer needs whole matrices, and the system has none*;
4. **Distributed RMNP is ten lines** — a literal transcription of fig-1's left panel; no new communication pattern, no touching ZeRO's bucket layout, nothing to hide:

```python
# each rank, on its own row-aligned shard, no collectives:
local_M.mul_(mu).add_(local_grad)                 # momentum (sharded)
u = F.normalize(local_M.float(), p=2, dim=-1)     # row norms: rows are whole here (fp32 sums)
local_W.add_(u.bfloat16(), alpha=-lr * scale)     # update own shard
```

5. **And it is not an approximation**: the sharded RMNP update is **bit-identical** to the full-matrix one (measured max |diff| = 0.00e+00). When the operator's geometry and the shard's geometry align, distribution is free.

The single-matrix size sweep (panel (b)) adds the final twist: the gap itself grows with the matrix — 20× at 1k², **1409×** at 8k². Muon's gather O(mn) and NS O(mn·min) both scale super-linearly; RMNP's elementwise O(mn) hugs the floor.

> Honest boundary: ① we measure the optimizer precondition segment, not end-to-end training speedup — its share depends on model and step time (Canzona's 1.57× end-to-end says that on production grids the share is large); ② our NS is eager bf16, uncompiled — production kernels are faster, but the 57 ms of communication is bandwidth-priced, and our PCIe topology makes communication expensive: on NVLink the gather is an order of magnitude cheaper and Muon's pain shifts from "comm + compute" toward "compute + scheduling" — while the three structural facts (O(mn), ∝ Ψ, un-overlappable) survive any topology; ③ under pure DDP (no sharding) both reach zero added communication — the true watershed is **sharding**, and hyperscale pretraining shards by necessity.

## 5. Why the gap grows with scale

1. **Traffic scaling**: Muon's optimizer communication ∝ total parameter count Ψ. A 7B model moves ~14 GB of bf16 momentum per GPU per step, landing on the busiest TP links or cross-node DP links; RMNP's counterpart is $$\Sigma m \approx \Psi/n$$ — three orders smaller, and exactly zero under most layouts;
2. **Critical-path position**: the optimizer step sits in each step's **serial segment** — backward finished, next forward not started — where communication has no compute to overlap with (contrast: gradient all-reduce hides inside backward, series post #2). Canzona's asynchronous pipeline exists to manufacture something to overlap; RMNP has nothing that needs hiding;
3. **Parallelism trend**: collective latency grows with group size (post #1: the latency floor climbs 11→65 µs from 2 to 8 ranks). Muon's gathers get harder to hide; RMNP's cost does not grow with parallelism at all.

## 6. The convenience is not bought with convergence

All of this matters only if RMNP holds up as an optimizer. The layer-wise Hessians of Transformers are **row-block-diagonally dominant**, so whole-matrix spectral preconditioning can be approximated at the row-block level — RMNP is exactly Muon's spectral normalization degraded to rows (ICML 2026); HTRMNP recovers part of the discarded spectral correction by scaling with the row norm's $$p$$-th power (default $$p=0.125$$). When Muon's global spectral information genuinely buys more, and how large that gap is (tall matrices, row-curvature heterogeneity), is another post's topic. This one nails a single fact: **on the systems side, row-local vs global is not a difference of degree — it is the difference between "keep your shard" and "reassemble the model, every step."**

## 7. Summary

1. The optimizer's communication bill is the product of two geometries: **the operator's dependency range × the shard's cut**. Muon's NS depends on the whole matrix → O(mn) plus a scheduling problem under any sharding; RMNP depends on single rows → 0 under FSDP2/row-TP, an O(m) decomposable reduction under column-TP;
2. Measured on 8 GPUs, GPT-2 Large's full matrix set: 5.3 ms vs 359.5 ms (**67×**), decomposed as 303 compute + 57 comm with the comm priced correctly by post #1's bandwidth table; the compute-amortizing round-robin variant is *slower* due to serialization — distributed Muon is worth a paper (Canzona), distributed RMNP is worth ten lines;
3. Sharded RMNP equals full-matrix RMNP bit for bit: align the operator's geometry with the shard's geometry, and distribution is free;
4. The gap grows with scale: traffic ∝ Ψ, sitting on the un-overlappable critical path, with latency rising in parallelism — all three arrows point the same way.

*(Series background: the memory/Ψ ledger in post #0, the collective-bandwidth table in post #1, FSDP2's dim-0 sharding in post #4, TP's row/column cuts in post #5, the bf16/fp32 conventions in post #8.)*

---

*Environment: 8× RTX PRO 6000 Blackwell (96GB), pure PCIe (no NVLink), PyTorch 2.9.1 + NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node=8 bench_dist_opt.py` (four schemes + size sweep + exactness check); schematic and plotting code accompanies the post.*
