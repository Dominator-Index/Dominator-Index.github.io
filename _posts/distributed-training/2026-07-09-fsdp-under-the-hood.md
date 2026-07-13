---
layout: post
title: "FSDP Under The Hood: ZeRO-3 The PyTorch Way"
date: 2026-07-09 10:00:00
description: "FlatParameter vs per-parameter DTensor sharded along dim-0, the gather/prefetch/reshard timeline, and the reshard_after_forward knob — measured: 11.8 → 1.3 GiB resident, with every gap priced by post #1's bandwidth table."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/04/fig-2-prefetch-timeline.png
toc:
  sidebar: left
related_posts: false
---

> Part 4 of **An Overview of Distributed Learning**. One post, one idea: **FSDP's sharding geometry and its fetch-and-return choreography** — FlatParameter (FSDP1) vs per-parameter DTensor sharded along dim-0 (FSDP2), and `reshard_after_forward`, the one switch that toggles between ZeRO-2 and ZeRO-3 semantics. Experiments: GPT-2 Large, FSDP2 vs DDP, 8 GPUs.

## 1. From ZeRO-3 to FSDP: the same idea, grown into the framework

Where [post #3](/blog/2026/distributed-training-illustrated-3-zero/) ended: under ZeRO-3, parameters no longer reside anywhere in full — each layer is all-gathered when the forward pass reaches it and discarded right after. That turns the optimizer from a training-loop accessory into a **runtime system**: it must hook module boundaries, schedule fetches, and manage communication streams. DeepSpeed builds that system outside PyTorch with hooks and coordinators; PyTorch later built it natively as **FSDP** — twice. The difference between the two generations is exactly the kind of "looks like an implementation detail, actually changes the story" that this series cares about.

## 2. Sharding geometry: FSDP1 flattens, FSDP2 cuts rows

The two generations have nearly identical memory/communication behavior (both are ZeRO-3). **The real difference is what one shard *is*:**

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/04/fig-1-cut-geometry.svg" class="img-fluid rounded" zoomable=true %}

- **FSDP1 (FlatParameter)**: all parameters of a wrap unit are **flattened and concatenated** into one giant 1-D buffer, split equally **by element count** across $$N$$ ranks. A shard is a byte range — it crosses parameter boundaries and slices matrix rows mid-way. The upside is simplicity (one buffer, one all-gather); the price is that a shard loses all mathematical structure: it is not a matrix, not rows — just bytes. Per-parameter states (freezing, mixed dtypes, per-param optimizer settings) become awkward.
- **FSDP2 (per-parameter DTensor)**: every parameter is sharded **independently**, along **dim-0 (the row dimension)**, as a `DTensor(placements=[Shard(0)])` that remembers its global shape. **Each rank holds complete rows** — a shard is itself a small $$[\frac{m}{N} \times n]$$ matrix.

FSDP2 pays zero extra communication for this (the all-gather moves the same bytes) and collects a long list of wins: per-parameter freeze/quantize/mixed-dtype, optimizer state aligned with DTensor (checkpoints addressable by logical tensor), composability with tensor parallelism's DTensor layouts, and deterministic memory release (no more FlatParameter `recordStream` haunting).

> **A seed for post #9**: "each rank holds complete rows" looks like an engineering convenience today, but it means **any row-wise decomposable computation (say, a row norm) can run on the sharded state with no communication at all.** Remember this sentence.

## 3. The choreography: gather on demand, prefetch ahead, reshard behind

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/04/fig-2-prefetch-timeline.svg" class="img-fluid rounded" zoomable=true %}

One FSDP2 step (sharding unit = one transformer block):

1. **Forward**: while block $$k$$ computes, block $$k{+}1$$'s parameter all-gather is already **prefetching** on a separate NCCL stream — only the first AG is exposed (the same idea as DDP's bucket overlap in post #2 and stage-3 prefetch in post #3, making its third appearance).
2. **Reshard** (the red ticks): the moment a block is done, its gathered full parameters are freed — memory returns to the sharded state. This is ZeRO-3 semantics.
3. **Backward**: the parameters were resharded, so **each block must be all-gathered again**; finished gradients reduce-scatter in buckets (fp32 reduction — precision semantics in post #8).

`reshard_after_forward` is thus an explicit **memory-for-communication** knob:

$$
\text{True (ZeRO-3)}:\ \text{low memory floor, backward pays an extra } \tfrac{N-1}{N}\cdot 2\Psi_{\text{bf16}} \text{ of AG}
\qquad
\text{False (ZeRO-2)}:\ \text{the reverse}
$$

We measure both sides of the trade in §5 and reconcile them against post #1's bandwidth table.

## 4. Reading along in real source, and usage

**FSDP2** — entry point `torch.distributed.fsdp.fully_shard`, implementation under `torch/distributed/fsdp/_fully_shard/`: parameter sharding in `_fsdp_param.py`, prefetch scheduling in `_fsdp_param_group.py`.

**FSDP1** — the `FullyShardedDataParallel` wrapper class. FlatParameter lives in `_flat_param.py`, two-thousand-plus lines — the complexity is itself the argument.

Our experiment's usage (full script ships with the post):

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,   # compute in bf16
                          reduce_dtype=torch.float32)   # reduce grads in fp32
for block in model.transformer.h:                       # one shard unit per block
    fully_shard(block, mp_policy=mp, reshard_after_forward=True)
fully_shard(model, mp_policy=mp, reshard_after_forward=True)
# then the usual loss.backward(); opt.step() — sharding, fetching and comms all hide in module hooks
```

Note the API shift from FSDP1: no wrapper class — `fully_shard` transforms the module **in place** and parameters become DTensors; `MixedPrecisionPolicy` replaces autocast (fp32 sharded master parameters + bf16 gathered compute parameters).

## 5. Experiment: FSDP2 vs DDP (GPT-2 Large, 8 GPUs)

**Setup**: Ψ = 0.774B, micro-batch 4×1024. DDP runs classic mixed precision (fp32 params + bf16 autocast); FSDP2 runs fp32 sharded masters + bf16 compute params — identical optimizer-precision semantics, so the comparison is fair.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/04/fig-3-fsdp-vs-ddp.svg" class="img-fluid rounded" zoomable=true %}

**Four accounts, all reconciled:**

1. **DDP resides at 11.76 GiB ≈ 16Ψ fp32 bytes** (params 4Ψ + gradient buckets 4Ψ + Adam 8Ψ = 11.5 GiB; `gradient_as_bucket_view` keeps bucket memory resident). FSDP2 resides at **1.27 GiB ≈ (4Ψ+8Ψ)/8 = 1.08 GiB** plus small buffers — **a 9.3× cut, exactly the "divide the whole ledger by N" promise**.
2. **DDP's exposed communication**: 477 − 153 (single-GPU compute) = 324 ms ≈ one all-reduce of 3.1 GB fp32 gradients (post #1: algbw 10.2 GB/s → 304 ms, 6% off). Reducing in fp32 is brutally expensive on this PCIe box — the main reason FSDP2 (338 ms) *beats* DDP: it all-gathers bf16 parameters and streams gradient reduce-scatters at bucket granularity.
3. **`reshard_after_forward=False` is 64.7 ms faster** ≈ the skipped backward re-all-gather of one bf16 model (1.55 GB / 21.8 GB/s ≈ 71 ms, 9% off); the price is +1.32 GiB of peak ≈ one bf16 model (1.44 GiB). **Both ends of the knob are priced, to within 10%, by post #1's bandwidth table**.
4. Compared with post #3: FSDP2's resident 1.27 GiB is **half of DeepSpeed stage 3's 2.53 GiB** — native per-param DTensor bookkeeping carries far less buffer overhead than an external implementation. Not a knock on DeepSpeed; a structural advantage of living inside the framework.

> Honest boundary: the DDP column uses fp32 parameters (classic mixed precision); with pure-bf16-parameter training its memory and communication both halve — the gap narrows, the conclusion stands. And this machine has no NVLink; on NVLink clusters the throughput differences compress.

## 6. Summary

1. FSDP = native ZeRO-3; the generational difference is **shard geometry**: FSDP1 flattens and splits by elements (shard = byte range, rows severed), FSDP2 shards each param along dim-0 (**shard = complete rows, a matrix in its own right**).
2. The runtime beats three times per unit: gather on demand, prefetch ahead, reshard behind; `reshard_after_forward` toggles ZeRO-2/3 semantics, and both prices are computable from post #1's table (measured <10% error).
3. Measured: resident memory 11.76 → 1.27 GiB (9.3×, reconciling 16Ψ → 16Ψ/8); FSDP2 carries half the buffer overhead of DeepSpeed stage 3; on machines where fp32 reduction is expensive, FSDP2 is even faster than DDP.
4. Row-complete shards are a detail today and the protagonist of post #9.

**Next: Tensor Parallelism — Megatron's two cuts.** So far the model itself still runs whole on every GPU (only *storage* was sharded). TP is the first scheme to cut **the computation of a single layer**: how Column and Row cuts pair into a "zero communication in the middle" combo, the conjugate operators $$f/g$$, and the bill of four activation all-reduces per layer per step.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node=8 bench_fsdp.py --mode {ddp,fsdp2} [--no-reshard]`; plotting and schematic code accompanies the series.*
