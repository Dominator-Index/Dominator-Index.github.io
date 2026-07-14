---
layout: post
title: "From DP To DDP: Buckets, Overlap, And A 25 MB Sweet Spot"
date: 2026-07-07 10:00:00
description: "Why data parallelism is mathematically equivalent to large-batch training, and how DDP reduces communication overhead through gradient bucketing and overlap, measured with a full bucket_cap_mb sweep on 8 GPUs."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/02/fig-2-bucket-overlap.png
toc:
  sidebar: left
related_posts: false
---

> Data parallelism computes the same averaged gradient as large-batch training. Its main cost is gradient synchronization. [Post #1](/blog/2026/the-price-of-all-reduce/) derived the cost of one all-reduce. This post shows how PyTorch DDP reduces its visible overhead through bucketing and overlap. We measure nanoGPT-124M on 8 GPUs across a full `bucket_cap_mb` sweep.

## 1. The invariant: DP never approximates anything

Data parallelism (DP) replicates the model on $N$ GPUs and splits the batch across them. Each GPU runs its own forward and backward passes, after which an all-reduce **averages the gradients** so that every replica applies the same update.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-1-dp-one-step.svg" class="img-fluid rounded" zoomable=true %}

Its correctness rests on one line of arithmetic. Split the global batch $$\mathcal{B}$$ into $$\mathcal{B}_0,\dots,\mathcal{B}_{N-1}$$. Since the loss is a sample average,

$$
\nabla L_{\mathcal{B}}(\theta) \;=\; \frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}} \nabla \ell(x;\theta)
\;=\; \frac{1}{N}\sum_{k=0}^{N-1}\underbrace{\frac{1}{|\mathcal{B}_k|}\sum_{x\in\mathcal{B}_k} \nabla \ell(x;\theta)}_{\text{local gradient } g_k \text{ on rank } k}
$$

**In exact arithmetic, the all-reduced average equals the gradient computed on the full batch.** All replicas start from the same $\theta_0$, receive the same averaged gradient, and apply the same update at every step. They therefore remain synchronized. DP does not change the optimization problem. It distributes a large-batch computation. The engineering challenge is to schedule the all-reduce without making the rest of the step wait.

We can estimate this cost using the formula from post #1. If the gradients occupy $S$ bytes, each GPU sends $2\frac{N-1}{N}S$ bytes per step. nanoGPT-124M uses fp32 gradients, so $S = 4\Psi \approx 498$ MB. At the all-reduce algorithm bandwidth measured on 8 GPUs, about 10.1 GB/s, an all-reduce with no overlap should take approximately **49 ms**. Section 4 compares this estimate with the measured overhead.

## 2. Why naive DP is slow, and how DDP fixes it

A naive implementation calls `all_reduce` on every `.grad` after `backward()`. This creates two bottlenecks:

1. **Repeated latency:** GPT-2 has roughly 150 parameter tensors, ranging from a few kilobytes to a few megabytes. Reducing each tensor separately pays the ~65 µs latency floor at $N{=}8$ about 150 times, adding roughly 10 ms.
2. **No overlap:** if communication starts only after backward finishes, the full 49 ms all-reduce remains on the critical path.

`torch.nn.parallel.DistributedDataParallel` (DDP) addresses both problems:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-2-bucket-overlap.svg" class="img-fluid rounded" zoomable=true %}

- **Bucketing:** DDP groups gradients into `bucket_cap_mb`-sized buckets, ordered roughly by when they become ready during backward. It performs one all-reduce per bucket, turning about 150 small collectives into roughly 20 larger ones and reducing repeated latency.
- **Overlap:** backward produces gradients from the last layer toward the first, so some gradients become ready before backward completes. DDP registers an autograd hook for each parameter. As soon as every gradient in a bucket is ready, DDP launches its all-reduce asynchronously on a **separate NCCL stream**. Communication for later layers can then run while earlier layers are still computing. Ideally, only the final communication tail remains after backward.

Bucket size controls a trade-off. **Small buckets launch early but pay more collective latency. Large buckets reduce latency but become ready too late to overlap effectively.** Section 4 measures the best balance.

## 3. Reading along in real source

**Bucket assignment** is implemented by `_compute_bucket_assignment_by_size` in `torch.distributed`. The first bucket defaults to 1 MB so that the earliest gradients can be reduced quickly. Later buckets use the default `bucket_cap_mb` value of 25 MB.

**Hooks and async reduction** live in `torch/csrc/distributed/c10d/reducer.cpp`: `Reducer::autograd_hook` → `mark_variable_ready` → bucket full → `all_reduce_bucket` on the comm stream.

**`gradient_as_bucket_view=True`** makes `.grad` a view into bucket memory, saving one copy and one gradient's worth of memory.

**`no_sync()`** is implemented in `torch/nn/parallel/distributed.py`. During gradient accumulation, it disables communication for the first gas−1 microbatches (Section 5).

Our benchmark is the standard usage, and the whole script ships with the post:

```python
model = DDP(model, device_ids=[local_rank],
            bucket_cap_mb=args.bucket_mb, gradient_as_bucket_view=True)
for micro in range(gas):
    ctx = model.no_sync() if micro < gas - 1 else nullcontext()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(X, Y)
        (loss / gas).backward()     # under no_sync: local accumulation only
opt.step(); opt.zero_grad(set_to_none=True)
```

## 4. Experiment 1: the `bucket_cap_mb` U-curve

**Setup**: nanoGPT-124M (fp32 params, bf16 autocast, fp32 grads, $$S \approx 498$$ MB), micro-batch 12×1024 tokens, 8-GPU DDP, 10 warm-up + 30 timed steps. Single-GPU baseline: 122.2 ms/step. Environment as always: 8× RTX PRO 6000 Blackwell, pure PCIe, NCCL 2.27.5.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-3-bucket-sweep.svg" class="img-fluid rounded" zoomable=true %}

The sweep shows three results:

1. **Step time follows the U-shaped curve predicted in Section 2.** A 1 MB bucket takes 152.7 ms because it launches too many collectives. The **25 MB setting is best at 147.1 ms**, matching PyTorch's default. A 500 MB bucket delays communication until almost the entire model is ready and takes 172.0 ms.
2. **The single-bucket result matches the cost estimate.** With almost no overlap, it adds 49.8 ms over the single-GPU baseline. Section 1 predicted approximately 49 ms from post #1's independent bandwidth measurements, a difference of 2%.
3. **Overlap saves 24.9 ms.** The best bucket takes 147.1 ms, compared with 172.0 ms for the single bucket, so DDP hides about half of the 49.8 ms communication cost. Communication is expensive on this PCIe machine, at about 41% of the 122.2 ms compute baseline. NVLink systems have a smaller communication cost and can usually hide a larger fraction of it.

The resulting scaling efficiency is $\frac{668.2}{8 \times 100.6} = 83\%$. The remaining 17% gap corresponds to roughly 25 ms of exposed communication.

## 5. Experiment 2: reducing communication with gradient accumulation and `no_sync`

Gradient accumulation combines gas microbatches into one optimizer step without changing the averaged gradient:

$$
g \;=\; \frac{1}{\text{gas}}\sum_{m=1}^{\text{gas}} g^{(m)}
\quad\Longleftrightarrow\quad
\text{backward } \tfrac{\ell^{(m)}}{\text{gas}} \text{ per microbatch, accumulate into .grad}
$$

Because summation commutes with all-reduce, **accumulating locally and synchronizing once** produces the same result as synchronizing after every microbatch. It requires one collective instead of gas collectives. DDP normally launches communication on every `backward()`, so `no_sync()` must explicitly disable the redundant synchronizations:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-4-throughput.svg" class="img-fluid rounded" zoomable=true %}

- With gas=4 but without `no_sync`, throughput is 674.8 ktok/s because three of the four synchronizations are unnecessary.
- With gas=4 and `no_sync`, throughput rises to **754.1 ktok/s with 94% scaling efficiency**, at 521.5 ms per step. The 61.3 ms reduction is approximately $3 \times 20$ ms, consistent with skipping three partially overlapped all-reduces.
- `no_sync` spreads one synchronization cost across four times as many tokens, reducing the communication-to-compute ratio from 41% to about 10%. Larger accumulation factors move throughput closer to the ideal scaling line, which is one reason gradient accumulation is common in large-model training.

> **Boundary of the result.** Each skipped synchronization saves about 20 ms rather than the full 25 ms of exposed communication because the all-reduces were already partially overlapped. In addition, `no_sync` requires local accumulation in `.grad`, which reduces some of the memory savings from `gradient_as_bucket_view`.

## 6. DP does not reduce model-state memory

DDP reduces the visible **time** spent on communication but does not reduce model-state **memory**. Every GPU still stores the full 16Ψ ledger from [post #0](/blog/2026/why-a-single-gpu-is-never-enough/): parameters, gradients, fp32 master parameters and Adam states. With 8 GPUs, the same optimizer state is stored eight times.

**Next: Data Parallelism, part 2, ZeRO's three-stage ledger.** Post #1 showed that all-reduce consists of reduce-scatter followed by all-gather. After reduce-scatter, rank $k$ owns the complete reduced gradient for shard $k$. ZeRO lets that rank **store and update only** the matching optimizer-state shard, then all-gathers the updated parameters. Communication remains nearly unchanged while model-state memory is divided by $N$. The next post measures per-GPU memory across ZeRO stages 1, 2 and 3.

## 7. Summary

1. In exact arithmetic, DP's averaged gradient equals the full-batch gradient, so all replicas remain synchronized. The main systems question is how to schedule communication.
2. DDP uses two techniques. **Bucketing** turns about 150 small collectives into roughly 20 larger ones, and **overlap** launches ready buckets during backward on a separate NCCL stream.
3. Bucket size produces a U-shaped performance curve, with the 25 MB default at the minimum. The single-bucket overhead of 49.8 ms matches post #1's bandwidth estimate within 2%, and the best setting hides half of that cost for 83% scaling efficiency.
4. Gradient accumulation should use `no_sync` so that one synchronization is shared across gas microbatches. With gas=4, scaling efficiency reaches 94%.
5. DP and DDP do not shard model state, so every rank still stores the 16Ψ ledger. ZeRO addresses this remaining redundancy.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `bench_ddp.py` (single-GPU baseline + bucket sweep + no_sync ablation). Benchmark, plotting and schematic code accompanies the series.*

*All benchmark scripts, schematic generators, plotting code and raw result CSVs for this post live in [assets/blog/code/02-data-parallel-ddp](https://github.com/Dominator-Index/Dominator-Index.github.io/tree/main/assets/blog/code/02-data-parallel-ddp).*

---

## Appendix: The Code That Ran

Every number in this post comes from the scripts below, embedded verbatim. Plotting and schematic code plus the raw result CSVs live in the folder linked above.

<details markdown="1">
<summary><code>bench_ddp.py</code></summary>

```python
"""
DDP experiments (Distributed Training Illustrated, post 02): nanoGPT-124M throughput.

Measures three things:
  1. Single GPU vs DDP (how big the comm overhead is, how much overlap buys back)
  2. bucket_cap_mb sweep (bucket size vs throughput: latency floor vs overlap window tradeoff)
  3. Effect of no_sync under gradient accumulation (skips gas-1 all-reduces)

Usage:
  # single-GPU baseline
  python bench_ddp.py --mode single --out ../results/ddp.csv
  # DDP + bucket sweep
  torchrun --standalone --nproc_per_node=8 bench_ddp.py --mode ddp --bucket-mb 25 --out ../results/ddp.csv
  # gradient accumulation with/without no_sync
  torchrun --standalone --nproc_per_node=8 bench_ddp.py --mode ddp --gas 4 --no-sync {0,1} --out ../results/ddp.csv
"""

import argparse
import csv
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, "/jumbo/yaoqingyang/ouyangzhuoli/MARS/MARS")
from model import GPT, GPTConfig  # noqa: E402

MBS, BLOCK = 12, 1024
WARMUP, STEPS = 10, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "ddp"], required=True)
    ap.add_argument("--bucket-mb", type=float, default=25.0)
    ap.add_argument("--gas", type=int, default=1)
    ap.add_argument("--no-sync", type=int, default=1)  # whether to use no_sync during accumulation
    ap.add_argument("--out", default="../results/ddp.csv")
    args = ap.parse_args()

    is_ddp = args.mode == "ddp"
    if is_ddp:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
    else:
        rank, world, device = 0, 1, torch.device("cuda", 0)
        torch.cuda.set_device(device)

    torch.manual_seed(1337 + rank)
    model = GPT(GPTConfig(dropout=0.0)).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[device.index], bucket_cap_mb=args.bucket_mb,
                    gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    X = torch.randint(0, 50304, (MBS, BLOCK), device=device)
    Y = torch.randint(0, 50304, (MBS, BLOCK), device=device)

    def one_step():
        for micro in range(args.gas):
            # no_sync: skip the all-reduce for the first gas-1 microbatches
            skip = is_ddp and args.no_sync and micro < args.gas - 1
            ctx = model.no_sync() if skip else nullcontext()
            with ctx, autocast:
                _, loss = model(X, Y)
                (loss / args.gas).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        one_step()
    torch.cuda.synchronize()
    if is_ddp:
        dist.barrier()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        one_step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / STEPS

    tokens_per_step = MBS * BLOCK * args.gas * world
    tps = tokens_per_step / dt
    if rank == 0:
        row = [args.mode, world, args.bucket_mb if is_ddp else "", args.gas,
               args.no_sync if (is_ddp and args.gas > 1) else "",
               round(dt * 1e3, 2), round(tps / 1e3, 1)]
        newfile = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["mode", "world", "bucket_cap_mb", "gas", "no_sync", "step_ms", "ktok_per_s"])
            w.writerow(row)
        print("ROW:", row, flush=True)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

</details>

