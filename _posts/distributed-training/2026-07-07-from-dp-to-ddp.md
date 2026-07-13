---
layout: post
title: "From DP To DDP: Buckets, Overlap, And A 25 MB Sweet Spot"
date: 2026-07-07 10:00:00
description: "DP's exactness invariant, and how DDP hides the gradient all-reduce inside the backward pass through bucketing against the latency floor and overlap against the bandwidth bill, with a full bucket_cap_mb sweep on 8 GPUs."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/02/fig-2-bucket-overlap.png
toc:
  sidebar: left
related_posts: false
---

> Data parallelism never approximates anything, and DDP buries its communication bill inside the backward pass. [Post #1](/blog/2026/distributed-training-illustrated-1-collective-communication/) priced one all-reduce, so this post watches PyTorch DDP pay that price and haggle it down, with nanoGPT-124M on 8 GPUs and a full `bucket_cap_mb` sweep.

## 1. The invariant: DP never approximates anything

Data parallelism (DP) is the most intuitive parallelism: replicate the model $$N$$ times, split the batch $$N$$ ways, let each GPU run forward/backward independently, then **average the gradients with one all-reduce** and apply identical updates.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-1-dp-one-step.svg" class="img-fluid rounded" zoomable=true %}

Its correctness rests on one line of arithmetic. Split the global batch $$\mathcal{B}$$ into $$\mathcal{B}_0,\dots,\mathcal{B}_{N-1}$$. Since the loss is a sample average,

$$
\nabla L_{\mathcal{B}}(\theta) \;=\; \frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}} \nabla \ell(x;\theta)
\;=\; \frac{1}{N}\sum_{k=0}^{N-1}\underbrace{\frac{1}{|\mathcal{B}_k|}\sum_{x\in\mathcal{B}_k} \nabla \ell(x;\theta)}_{\text{local gradient } g_k \text{ on rank } k}
$$

**The all-reduced average equals, bit for bit, the gradient one giant GPU would compute on the full batch.** So DP's semantic invariant is that replicas start from the same $$\theta_0$$, receive the same averaged gradient every step, apply the same update, **and stay bit-identical forever**. DP approximates nothing. It merely spreads out a big-batch computation. All that remains is an engineering question: *when and how to run that all-reduce so it doesn't block the pipeline.*

Let's also pre-compute the bill with the formula from post #1. With gradient bytes $$S$$, each GPU pays $$2\frac{N-1}{N}S$$ per step. Our experimental model, nanoGPT-124M, keeps fp32 gradients, so $$S = 4\Psi \approx 498$$ MB. At the all-reduce algorithm bandwidth we measured last time (8 GPUs, ~10.1 GB/s), **one bare, un-hidden all-reduce ≈ 49 ms**. Remember that number, because §4 will reconcile against it.

## 2. Two ways naive DP dies, two tricks that save DDP

The naive implementation (loop over parameters after `backward()`, `all_reduce` each `.grad`) dies of two ailments, both foreshadowed in post #1:

1. **The latency floor**: GPT-2 has ~150 parameter tensors, mostly KBs to a few MBs. Per-tensor all-reduce hits the ~65 µs floor ($$N{=}8$$) every time, for ~10 ms of pure latency.
2. **Serial exposure**: waiting for the whole backward before communicating leaves the entire 49 ms naked on the critical path.

`torch.nn.parallel.DistributedDataParallel` answers with two tricks, one figure:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-2-bucket-overlap.svg" class="img-fluid rounded" zoomable=true %}

- **Bucketing**: parameters are grouped (in reverse order of arrival during backward) into buckets of `bucket_cap_mb`, and one bucket = one all-reduce. "150 small collectives" becomes "~20 large ones", so the latency cost divides by 7.
- **Overlap**: backward naturally computes gradients **last layer first**, so later layers' gradients are ready early. DDP registers an autograd hook per parameter, and the instant a bucket's gradients are all ready, its all-reduce launches asynchronously on a **separate NCCL stream**, running concurrently with the still-ongoing backward of earlier layers. Ideally, by the time backward finishes, most communication is already done or in flight, and only the last bucket's tail is exposed.

The trade-off is drawn in the figure too. **Too small a bucket falls back onto the per-collective latency floor, and too big a bucket leaves nothing to overlap with.** The sweet spot must be measured, which is what §4 does.

## 3. Reading along in real source

**Bucket assignment** happens in `_compute_bucket_assignment_by_size` in `torch.distributed`. The first bucket defaults to 1 MB, deliberately tiny so the last layer's gradients depart as early as possible. The rest default to `bucket_cap_mb` = 25 MB.

**Hooks and async reduction** live in `torch/csrc/distributed/c10d/reducer.cpp`: `Reducer::autograd_hook` → `mark_variable_ready` → bucket full → `all_reduce_bucket` on the comm stream.

**`gradient_as_bucket_view=True`** makes `.grad` a view into bucket memory, saving one copy and one gradient's worth of memory.

**`no_sync()`** lives in `torch/nn/parallel/distributed.py` and disables the hooks' communication for the first gas−1 microbatches of gradient accumulation (§5).

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

Three readings:

1. **The U-curve** predicted in §2: 1 MB (too many buckets, latency) 152.7 ms → **25 MB optimal, 147.1 ms** (exactly PyTorch's default, which was not picked out of a hat) → 500 MB (whole model in one bucket) 172.0 ms.
2. **Reconciliation**: the single-bucket run (zero overlap) costs 49.8 ms over the single-GPU floor, matching the "bare all-reduce ≈ 49 ms" priced in §1 from post #1's bandwidth measurements to 2%. **Two posts, two independent experiments, one number**.
3. **Overlap's net gain**: best bucket (147.1 ms) vs zero overlap (172.0 ms) = 24.9 ms saved, which is **exactly half of the 49.8 ms bill hidden**. On this PCIe box, where communication is expensive relative to compute (49.8/122.2 ≈ 41%), hiding half is a solid result. On NVLink machines the bill itself is smaller and hides more completely.

Scaling efficiency: $$\frac{668.2}{8 \times 100.6} = 83\%$$. The missing 17% is precisely the ~25 ms of exposed communication.

## 5. Experiment 2: gradient accumulation and `no_sync` to pay fewer bills

Gradient accumulation (gas microbatches per optimizer step) is mathematically a free batch-size multiplier:

$$
g \;=\; \frac{1}{\text{gas}}\sum_{m=1}^{\text{gas}} g^{(m)}
\quad\Longleftrightarrow\quad
\text{backward } \tfrac{\ell^{(m)}}{\text{gas}} \text{ per microbatch, accumulate into .grad}
$$

Summation commutes with all-reduce, so **accumulating locally and synchronizing once** gives the identical result as synchronizing every microbatch but pays 1 bill instead of gas. DDP's hooks fire on every `backward()`, so the redundant syncs must be turned off explicitly with `no_sync()`:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-4-throughput.svg" class="img-fluid rounded" zoomable=true %}

- gas=4 without `no_sync`: 674.8 ktok/s, since three of the four syncs are pure waste.
- gas=4 with `no_sync`: **754.1 ktok/s (94% scaling efficiency)**, 521.5 ms/step. The 61.3 ms saved ≈ 3 × 20 ms (the three skipped, already-partially-overlapped all-reduces).
- Intuition: `no_sync` amortizes the bill over 4× the tokens, dropping the comm/compute ratio from 41% to ~10%. Larger gas approaches the ideal line, which is one reason large-model training always ships with gradient accumulation.

> Honest boundary: each skipped sync saves ~20 ms, not the full 25 ms of exposed time, because the skipped all-reduces were themselves partially hidden by overlap. And under `no_sync`, `.grad` must accumulate locally, which claws back some of `gradient_as_bucket_view`'s memory savings.

## 6. DP's ceiling: it saves not one byte of memory

DDP solves communication's **time** problem but does nothing for **memory**. Every GPU still carries the full 16Ψ ledger from [post #0](/blog/2026/distributed-training-illustrated-0-the-5d-map/): parameters, gradients, fp32 master weights, Adam state, all of it. Across 8 GPUs, the same optimizer state is stored 8 times.

**Next: Data Parallelism, part 2, ZeRO's three-stage ledger.** Recall from post #1 that all-reduce = reduce-scatter + all-gather. ZeRO splits it open in the middle. After the reduce-scatter, each rank holds the complete sum of 1/N of the gradients, so let rank $$k$$ **update and store only** shard $$k$$ of the optimizer state and then all-gather the fresh parameters. Communication volume is nearly unchanged, but memory divides by $$N$$. We will measure per-GPU memory across stages 1/2/3.

## 7. Summary

1. DP's invariant is that the averaged gradient equals the big-batch gradient bit-for-bit, so replicas stay identical. Correctness is free, and the only question is where to put the communication.
2. DDP has two tricks. **Bucketing** fights the latency floor (150 small collectives → ~20 large), and **overlap** hides communication inside backward (autograd hooks + a separate NCCL stream).
3. Measured, bucket size is a U-curve with the 25 MB default sitting at the bottom. The single-bucket run's +49.8 ms reconciles with post #1's bandwidth data to 2%, and the best bucket hides half the bill for 83% scaling efficiency.
4. Gradient accumulation demands `no_sync`. The bill then amortizes by gas, giving 94% efficiency.
5. DP/DDP saves zero memory (16Ψ × N redundancy), and that is ZeRO's stage.

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

