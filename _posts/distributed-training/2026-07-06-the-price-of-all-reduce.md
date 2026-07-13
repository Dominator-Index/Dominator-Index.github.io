---
layout: post
title: "The Price Of All-Reduce"
date: 2026-07-06 10:00:00
description: "The six collective primitives, the ring all-reduce derivation of 2(N−1)/N·S, and what all of it actually measures on a real 8-GPU PCIe box."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/01/fig-2-ring-allreduce.png
toc:
  sidebar: left
featured: true
related_posts: false
---

> The six collective communication primitives are the vocabulary of all distributed training, and ring all-reduce costs each GPU exactly $$2\frac{N-1}{N}S$$ bytes. Every number in this post was measured on our own 8-GPU machine, and the benchmark and plotting code ships with the post.

Every distributed training scheme, whether DDP, ZeRO, FSDP, TP or PP, decomposes at the bottom into different arrangements of the same six building blocks, the collective communication primitives. Every communication bill in the rest of this series will be denominated in these blocks. So the series starts by making the blocks themselves precise: **what each primitive does, how many bytes each GPU pays for it, and how fast those bytes actually move on real hardware.**

First, the accounting convention for the entire series (stated once, used forever):

> **Communication volume = bytes a given GPU cumulatively sends, per operation.** $$S$$ is the total byte size of the logical tensor, $$N$$ the process-group size. We count only sends because modern interconnects (NVLink/PCIe/IB) are full-duplex, so send and receive don't compete for bandwidth, and in well-designed collectives every GPU's receive volume equals its send volume anyway (you'll see why below).

## 1. The six primitives, one figure

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/01/fig-1-six-primitives.svg" class="img-fluid rounded" zoomable=true %}

Read the figure in three groups, by participation topology:

| Primitive | Semantics ($$N$$ GPUs) | Per-GPU send volume | Role in training |
|-----------|------------------------|---------------------|------------------|
| **broadcast** | root's full tensor copied to everyone | ≈ $$S$$ per GPU (ring implementation) | weight sync at init |
| **scatter** | root cuts the tensor into $$N$$ shards, shard $$k$$ → rank $$k$$ | root sends $$\frac{N-1}{N}S$$, others 0 | distributing data/state |
| **gather** | inverse scatter: all shards collected at root | each sends $$\frac{S}{N}$$ | collecting results |
| **all-gather** | each rank holds one shard; afterwards **everyone holds the full concatenation** | $$\frac{N-1}{N}S$$ | ZeRO-3/FSDP parameter fetch |
| **reduce-scatter** | each rank holds a full tensor; shard $$k$$ of the **sum** lands on rank $$k$$ | $$\frac{N-1}{N}S$$ | ZeRO gradient sync |
| **all-reduce** | element-wise sum of everyone's tensor, **result on everyone** | $$2\frac{N-1}{N}S$$ | DDP gradient sync |

Two structural facts jump out:

1. **all-reduce = reduce-scatter + all-gather.** Compare the last row of the figure. First let "the sum of shard $$k$$" land on rank $$k$$ (reduce-scatter), then circulate the finished shards to everyone (all-gather), and that *is* all-reduce. The volumes agree too: $$\frac{N-1}{N}S + \frac{N-1}{N}S = 2\frac{N-1}{N}S$$. This is not a coincidence but how NCCL actually implements it, and ZeRO's core idea (post #3) is precisely to split the all-reduce open in the middle and insert local computation between the two halves.
2. **The `all-` primitives have no root**, so all ranks are symmetric. The rooted ones (broadcast/scatter/gather) have a distinguished GPU whose link becomes the bottleneck, an asymmetry that will show up undisguised in the experiments (§4).

In PyTorch they map one-to-one onto `torch.distributed`:

```python
import torch.distributed as dist

dist.broadcast(t, src=0)                    # root's t overwrites everyone's
dist.scatter(out, chunks, src=0)            # root sends chunks[k] to rank k
dist.gather(t, outs, dst=0)                 # everyone's t collected at root
dist.all_gather_into_tensor(out, t)         # out = concat of everyone's t, for everyone
dist.reduce_scatter_tensor(out, t)          # out = this rank's shard of the group sum
dist.all_reduce(t)                          # t = element-wise group sum, in place
```

Collectives are **synchronization points**. Every rank in the group must issue the same call (a missing rank hangs everyone), with matching shapes and dtypes.

## 2. Ring all-reduce: two phases, $$N-1$$ steps each

Why not do it naively? The centralized scheme (everyone sends to rank 0, which sums and sends back) makes the root receive and send $$(N-1)S$$ bytes each, so its bandwidth demand explodes linearly in $$N$$ while the other $$N-1$$ GPUs' links sit idle. The information-theoretic floor is that each GPU must send ≈ $$S$$ and receive ≈ $$S$$, because its data must enter the sum and the sum must reach it. A good algorithm should **push every GPU close to that floor while keeping all links busy simultaneously.**

Ring all-reduce does exactly this. Arrange the $$N$$ GPUs in a logical ring, cut the tensor into $$N$$ shards, run $$N-1$$ steps of reduce-scatter and then $$N-1$$ steps of all-gather:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/01/fig-2-ring-allreduce.svg" class="img-fluid rounded" zoomable=true %}

How to read it:

- **Phase 1 (RS steps 1–3)**: each step, every GPU *sends one shard, receives one shard, and adds what it received into its own copy*. Partial sums travel clockwise, absorbing one term per hop, and after $$N-1$$ steps each GPU owns **one fully-summed shard** (Σ).
- **Phase 2 (AG steps 1–3)**: the finished shards travel around the ring once more, pure forwarding, no arithmetic, and everyone collects all Σ shards.
- At every step, every GPU is **sending (to its successor) and receiving (from its predecessor) at the same time**, so both directions of the full-duplex link are saturated and no GPU sits idle. This is why counting only sends is legitimate. Receive equals send, and the two don't contend.

**The bill**: each GPU sends $$2(N-1)$$ times, one $$\frac{S}{N}$$-byte shard each:

$$
\boxed{\;\text{per-GPU send} \;=\; 2(N-1)\cdot\frac{S}{N} \;=\; 2\,\frac{N-1}{N}\,S \;\xrightarrow{\;N\to\infty\;}\; 2S\;}
$$

**This bound is the mathematical foundation of data-parallel scalability.** Per-GPU communication does not grow with $$N$$ (capped at $$2S$$), so adding GPUs adds no per-GPU burden. The cost hides in the other term of the time model:

$$
T_{\text{ring}} \;\approx\; 2(N-1)\left(\frac{S/N}{B} + \alpha\right)
\;=\; \underbrace{2\,\frac{N-1}{N}\cdot\frac{S}{B}}_{\text{bandwidth term, } \approx 2S/B}
\;+\; \underbrace{2(N-1)\,\alpha}_{\text{latency term, linear in } N}
$$

where $$B$$ is the one-directional link bandwidth and $$\alpha$$ the fixed per-step overhead (kernel launch + one hop). Two regimes follow:

- **Large $$S$$**: bandwidth-dominated, $$T \approx 2S/B$$, independent of $$N$$, so big tensors scale freely.
- **Small $$S$$**: latency-dominated, $$T \approx 2(N-1)\alpha$$, independent of $$S$$, so small tensors pay pure latency, and that latency grows linearly with $$N$$.

The boundary between the regimes is directly visible in the measurements (§4.3), and it is the entire reason DDP buckets small gradients into big ones (post #2).

## 3. Two bandwidth conventions: algbw vs busbw

Benchmarking collectives has a classic unit trap, so let's nail it down first (the nccl-tests convention, used in all plots below):

- **Algorithm bandwidth** $$\text{algbw} = S/t$$: the user's view, as in "my $$S$$-byte tensor took $$t$$ seconds to synchronize." It includes the algorithm's redundant data movement, so it is **not comparable across primitives**.
- **Bus bandwidth** $$\text{busbw} = \text{algbw} \times \text{correction}$$: the hardware's view, normalized to "traffic actually carried per link," comparable across primitives and machines, and directly checkable against the link's spec sheet.

| Primitive | Correction factor | Source |
|-----------|-------------------|--------|
| all-reduce | $$2\frac{N-1}{N}$$ | the derivation above |
| all-gather / reduce-scatter | $$\frac{N-1}{N}$$ | single-phase ring |
| broadcast | $$1$$ | each GPU relays ≈ $$S$$ |
| scatter / gather | $$\frac{N-1}{N}$$ | root-side traffic |

One immediately testable corollary is that **at equal $$S$$, all-reduce's algbw should be about half of all-gather's**, because it does twice the work. We check this below.

## 4. Experiments: six primitives on eight GPUs

**Setup** (the series' standard environment):

| | |
|---|---|
| Hardware | 8× NVIDIA RTX PRO 6000 Blackwell (96 GB), **pure PCIe, no NVLink** |
| Topology | GPUs paired under PCIe switches (0-1/2-3/4-5/6-7), dual NUMA, UPI across sockets |
| Software | PyTorch 2.9.1 + cu128, NCCL 2.27.5, bf16 tensors |
| Method | CUDA-event timing, ≥5 warm-up iterations, mean of 10–50 runs; `torchrun --standalone` |
| Sweep | $$S$$ from 4 KiB to 1 GiB, $$N \in \{2, 4, 8\}$$ |

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/01/fig-3-topology.svg" class="img-fluid rounded" zoomable=true %}

This machine has no NVLink, just a commoner topology, and that's a feature because the bottlenecks are unusually easy to see.

### 4.1 Bus bandwidth of the six primitives

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/01/fig-4-primitives-busbw.svg" class="img-fluid rounded" zoomable=true %}

Three layers of information:

1. **The four ring collectives (all-reduce / all-gather / reduce-scatter / broadcast) crowd onto the same ~16–19 GB/s plateau.** This is what the busbw convention is for. The algorithms differ, but normalized to link traffic they all hit the same wall, the ring's bottleneck link. On our topology the ring must cross the UPI socket link (the red link in the topology figure), and every link on the ring is busy in both directions at once, so the effective one-directional share is this ~18 GB/s.
2. **scatter/gather reach 44–50 GB/s** because they are not rings. The root does one-way point-to-point distribution/collection to $$N-1$$ distinct peers, exposing the practical one-directional bandwidth of a single PCIe Gen5 x16 link (~50+ GB/s). Same machine, same wires, **2.5× more usable bandwidth just by changing the communication pattern**, so communication performance is a function of algorithm × topology, not a hardware constant.
3. **"AR = RS + AG", empirically**: at $$S = 1$$ GiB, $$t_{\text{AR}} = 105.3$$ ms while $$t_{\text{RS}} + t_{\text{AG}} = 59.2 + 49.3 = 108.5$$ ms, a 3% discrepancy. The textbook identity holds at millisecond precision. And the §3 corollary checks out: algbw(AR) = 10.2 GB/s ≈ half of algbw(AG) = 21.8 GB/s ✓.

### 4.2 What adding GPUs does: all-reduce on 2 / 4 / 8 cards

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/01/fig-5-allreduce-scaling.svg" class="img-fluid rounded" zoomable=true %}

The left panel (algbw, what you feel) shows that larger $$N$$ is slower, and 8 GPUs deliver less than half the per-tensor throughput of 2. In the right panel (busbw) the three curves **collapse onto one**. Divide out the $$2\frac{N-1}{N}$$ factor and what remains is just the capability of the wires (21.5 GB/s at $$N{=}2$$, 17.8 at $$N{=}8$$, where the residual gap is the 8-GPU ring having to cross UPI while the 2-GPU pair shares one PCIe switch). **The formula is not an approximation. It is a change of coordinates on the measured curves.**

### 4.3 The latency floor: are small tensors free? No, they're deadly

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/01/fig-6-latency-floor.svg" class="img-fluid rounded" zoomable=true %}

For $$S \le 64$$ KiB the time curve is perfectly flat. A 4 KiB and a 64 KiB all-reduce cost the same (~60–70 µs at $$N{=}8$$), which is 16× the bytes in identical time, so **you are paying pure latency** $$2(N-1)\alpha$$. And the floor rises with $$N$$, from ~11 µs at $$N{=}2$$ to ~65 µs at $$N{=}8$$. On this machine the bandwidth/latency boundary sits at about 1 MiB.

To feel the weight of that number, consider that a GPT-2's LayerNorm weight is a few KB. If every small parameter tensor were all-reduced individually, a 124M model's ~150 tensors × 65 µs ≈ 10 ms of pure latency per step, which is nearly half the cost of synchronizing the *entire model in one call* (bf16, $$S \approx 250$$ MB, ~25 ms) while moving 0.1% of the bytes. **This is the entire reason DDP gradient bucketing exists**, and it is the protagonist of the next post.

## 5. Reading along in real source

**PyTorch DDP**: the six calls benchmarked here are the very ones production frameworks issue. DDP's gradient sync is one `all_reduce` per gradient bucket, and the bucketing machinery lives in `torch/csrc/distributed/c10d/reducer.cpp`, entered from `torch.nn.parallel.DistributedDataParallel`. ZeRO and FSDP replace it with `reduce_scatter_tensor` + `all_gather_into_tensor` (posts #3–#4).

**NCCL**: channel construction lives in the topology engine (`ncclTopoCompute`), the two-phase pipelined ring in `src/device/all_reduce.h`. NCCL also auto-switches between ring/tree algorithms and LL/LL128/Simple protocols by message size, and the small bump of the broadcast curve at 1 MiB in fig-4 is such a protocol switch showing through.

**MARS (our own repo)**: the Moonlight-style optimizer in [MARS](https://github.com/AGI-Arena/MARS) (`MARS/optimizers/muon.py`) ends each step with one `dist.all_reduce(updates_flat)`. You can now price it exactly. With $$S$$ the total parameter bytes, each GPU pays an extra $$2\frac{N-1}{N}S$$ per step.

## 6. Summary

1. Six primitives are the communication vocabulary of all distributed training. **All-reduce = reduce-scatter + all-gather**, confirmed at 3% measurement error.
2. Ring all-reduce costs each GPU $$2\frac{N-1}{N}S < 2S$$, **independent of $$N$$**, which is the foundation of DP scalability. The price is a latency term $$2(N-1)\alpha$$ linear in $$N$$, so small-tensor collectives are pure latency.
3. Compare performance in busbw. Measured, the four ring collectives hit one shared wall (the bottleneck link) while rooted primitives expose single-link one-way bandwidth, a 2.5× spread on the same hardware.
4. This machine's numbers (pure PCIe, dual NUMA): ring collectives ~18 GB/s, point-to-point one-way ~50 GB/s, latency floor 11→65 µs ($$N$$: 2→8), bandwidth/latency boundary ~1 MiB.

**Next: Data Parallelism, part 1, from DP to DDP.** The gradient-sync bill of $$2\frac{N-1}{N}S$$ is due every single step, but DDP hides it inside the backward pass with two tricks: bucketing (against the latency floor) and compute/communication overlap (against the bandwidth term). We will sweep `bucket_cap_mb` from 1 to 500 and watch both tricks work.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node={2,4,8} bench_collectives.py`. Benchmark, plotting and schematic-generation code accompanies the series.*

*All benchmark scripts, schematic generators, plotting code and raw result CSVs for this post live in [assets/blog/code/01-collective-communication](https://github.com/Dominator-Index/Dominator-Index.github.io/tree/main/assets/blog/code/01-collective-communication).*

---

## Appendix: The Code That Ran

Every number in this post comes from the scripts below, embedded verbatim. Plotting and schematic code plus the raw result CSVs live in the folder linked above.

<details markdown="1">
<summary><code>bench_collectives.py</code></summary>

```python
"""
Collective communication primitive benchmark (Distributed Training Illustrated, post 01).

Benchmarks 6 primitives: broadcast / scatter / gather / all_gather / reduce_scatter / all_reduce
Conventions (matching the post):
  - S = logical message size (bytes of the full tensor)
  - algbw = S / t                          (algorithm bandwidth, the "user view")
  - busbw = algbw x correction factor      (bus bandwidth, the "hardware view", nccl-tests convention)
      all_reduce:      2(N-1)/N
      all_gather:      (N-1)/N
      reduce_scatter:  (N-1)/N
      broadcast:       1
      scatter/gather:  (N-1)/N

Usage:
  torchrun --standalone --nproc_per_node=8 bench_collectives.py --out ../results/collectives_n8.csv
"""

import argparse
import csv
import os

import torch
import torch.distributed as dist

SIZES = [4 * 2**10, 64 * 2**10, 2**20, 16 * 2**20, 256 * 2**20, 2**30]  # 4KiB..1GiB
DTYPE = torch.bfloat16


def bus_factor(op, n):
    return {
        "all_reduce": 2 * (n - 1) / n,
        "all_gather": (n - 1) / n,
        "reduce_scatter": (n - 1) / n,
        "broadcast": 1.0,
        "scatter": (n - 1) / n,
        "gather": (n - 1) / n,
    }[op]


def make_op(op, size_bytes, rank, world, device):
    """Return (fn, actually_allocated_ok). size_bytes is the logical message size S."""
    numel = size_bytes // DTYPE.itemsize
    # Ensure divisibility by world size (needed by scatter/gather/AG/RS)
    numel = (numel // world) * world
    if numel == 0:
        return None

    if op == "all_reduce":
        t = torch.randn(numel, dtype=DTYPE, device=device)
        return lambda: dist.all_reduce(t)

    if op == "broadcast":
        t = torch.randn(numel, dtype=DTYPE, device=device)
        return lambda: dist.broadcast(t, src=0)

    if op == "all_gather":
        out = torch.empty(numel, dtype=DTYPE, device=device)
        inp = torch.randn(numel // world, dtype=DTYPE, device=device)
        return lambda: dist.all_gather_into_tensor(out, inp)

    if op == "reduce_scatter":
        inp = torch.randn(numel, dtype=DTYPE, device=device)
        out = torch.empty(numel // world, dtype=DTYPE, device=device)
        return lambda: dist.reduce_scatter_tensor(out, inp)

    if op == "scatter":
        out = torch.empty(numel // world, dtype=DTYPE, device=device)
        if rank == 0:
            chunks = list(torch.randn(numel, dtype=DTYPE, device=device).chunk(world))
            return lambda: dist.scatter(out, chunks, src=0)
        return lambda: dist.scatter(out, None, src=0)

    if op == "gather":
        inp = torch.randn(numel // world, dtype=DTYPE, device=device)
        if rank == 0:
            outs = list(torch.empty(numel, dtype=DTYPE, device=device).chunk(world))
            return lambda: dist.gather(inp, outs, dst=0)
        return lambda: dist.gather(inp, None, dst=0)

    raise ValueError(op)


def bench(fn, iters, device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    ops = ["broadcast", "scatter", "gather", "all_gather", "reduce_scatter", "all_reduce"]
    rows = []
    for op in ops:
        for size in SIZES:
            fn = make_op(op, size, rank, world, device)
            if fn is None:
                continue
            warmup = 20 if size < 256 * 2**20 else 5
            iters = 50 if size < 256 * 2**20 else 10
            for _ in range(warmup):
                fn()
            t_ms = bench(fn, iters, device)
            algbw = size / (t_ms / 1e3) / 1e9  # GB/s
            busbw = algbw * bus_factor(op, world)
            if rank == 0:
                rows.append([op, world, size, round(t_ms, 4), round(algbw, 2), round(busbw, 2)])
                print(f"{op:15s} N={world} S={size/2**20:9.3f}MiB  t={t_ms:9.3f}ms  algbw={algbw:7.2f}GB/s  busbw={busbw:7.2f}GB/s", flush=True)
            del fn
            torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op", "world_size", "bytes", "time_ms", "algbw_GBps", "busbw_GBps"])
            w.writerows(rows)
        print(f"wrote {args.out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

</details>

