---
layout: post
title: "FSDP Under The Hood: ZeRO-3 The PyTorch Way"
date: 2026-07-09 10:00:00
description: "How FSDP1 and FSDP2 shard parameters, how gather, prefetch and reshard fit into one step, and how reshard_after_forward reduces resident memory from 11.8 GiB to 1.3 GiB on 8 GPUs."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/04/fig-2-prefetch-timeline.png
toc:
  sidebar: left
related_posts: false
---

> FSDP1 flattens parameters before sharding, while FSDP2 shards each parameter independently along dim-0. This difference determines what structure remains in each shard. The `reshard_after_forward` option then controls whether full parameters are kept for backward or gathered again. We compare FSDP2 with DDP on GPT-2 Large across 8 GPUs.

## 1. From ZeRO-3 to FSDP: integrating sharding into PyTorch

[Post #3](/blog/2026/zero-three-ledgers/) described ZeRO-3, where no rank stores complete parameters between operations. Each layer's parameters are all-gathered when needed and released after use. Supporting this execution model requires hooks at module boundaries, scheduled parameter fetches and dedicated communication streams. DeepSpeed adds these mechanisms around PyTorch modules. PyTorch later implemented them natively as **FSDP**, first with flattened parameters and then with per-parameter tensors.

## 2. Sharding geometry: FSDP1 flattens, FSDP2 cuts rows

Both generations implement ZeRO-3-style memory and communication behavior. **They differ in what each shard contains:**

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/04/fig-1-cut-geometry.svg" class="img-fluid rounded" zoomable=true %}

- **FSDP1 (FlatParameter):** all parameters in a wrapping unit are **flattened and concatenated** into one 1-D buffer, then divided equally by element count across $N$ ranks. A shard can cross parameter boundaries and split matrix rows. This representation needs only one buffer and one all-gather, but it no longer preserves individual matrix structure. Per-parameter operations such as freezing, mixed dtypes and optimizer settings become harder to manage.
- **FSDP2 (per-parameter DTensor):** each parameter is sharded **independently** along **dim-0**, the row dimension. A `DTensor(placements=[Shard(0)])` retains the global shape and placement metadata. **Each rank holds complete rows**, so its shard remains a valid $[\frac{m}{N} \times n]$ matrix.

This structure does not add communication because all-gather moves the same number of bytes. It also supports per-parameter freezing, quantization and mixed dtypes. Optimizer states and checkpoints stay aligned with logical tensors. The structure composes with tensor-parallel DTensor layouts and makes memory release more predictable.

> **Connection to post #9.** Because each rank holds complete rows, any row-local operation, such as computing a row norm, can run directly on the shard without communication.

## 3. FSDP execution: gather, prefetch and reshard

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/04/fig-2-prefetch-timeline.svg" class="img-fluid rounded" zoomable=true %}

For one FSDP2 step, with each transformer block as a sharding unit:

1. **Forward:** while block $k$ computes, FSDP **prefetches** the parameters for block $k{+}1$ with an all-gather on a separate NCCL stream. Ideally, only the first all-gather is fully exposed. This uses the same overlap principle as DDP gradient buckets in post #2 and ZeRO-3 prefetch in post #3.
2. **Reshard:** after a block finishes, FSDP releases its gathered full parameters and returns to the sharded representation. The red ticks in the figure mark these releases. This is ZeRO-3 behavior.
3. **Backward:** because the parameters were released after forward, **each block must be all-gathered again**. Completed gradients are reduced and sharded with bucketed reduce-scatter. Post #8 discusses why the reduction uses fp32.

`reshard_after_forward` therefore controls a direct **trade-off between memory and communication:**

$$
\text{True (ZeRO-3)}:\ \text{low memory floor, backward pays an extra } \tfrac{N-1}{N}\cdot 2\Psi_{\text{bf16}} \text{ of AG}
\qquad
\text{False (ZeRO-2)}:\ \text{the reverse}
$$

Section 5 measures both settings and compares their time and memory differences with the bandwidth results from post #1.

## 4. Reading along in real source, and usage

**FSDP2** is exposed through `torch.distributed.fsdp.fully_shard`. Its implementation is under `torch/distributed/fsdp/_fully_shard/`, with parameter sharding in `_fsdp_param.py` and prefetch scheduling in `_fsdp_param_group.py`.

**FSDP1** uses the `FullyShardedDataParallel` wrapper class. Its FlatParameter implementation is contained in the more than two-thousand-line `_flat_param.py`, reflecting the complexity of managing flattened parameters.

Our experiment's usage (full script ships with the post):

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,   # compute in bf16
                          reduce_dtype=torch.float32)   # reduce grads in fp32
for block in model.transformer.h:                       # one shard unit per block
    fully_shard(block, mp_policy=mp, reshard_after_forward=True)
fully_shard(model, mp_policy=mp, reshard_after_forward=True)
# Call loss.backward() and opt.step(). Module hooks manage sharding, fetching, and communication
```

Unlike FSDP1, FSDP2 does not wrap the module in a new class. `fully_shard` transforms it **in place**, and its parameters become DTensors. `MixedPrecisionPolicy` specifies fp32 sharded master parameters and bf16 gathered parameters for computation, replacing the autocast setup used in this experiment.

## 5. Experiment: FSDP2 vs DDP (GPT-2 Large, 8 GPUs)

**Setup:** Ψ = 0.774B with a 4×1024 microbatch. DDP uses fp32 parameters with bf16 autocast, while FSDP2 uses fp32 sharded master parameters and bf16 gathered parameters for computation. Both retain the same optimizer-state precision.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/04/fig-3-fsdp-vs-ddp.svg" class="img-fluid rounded" zoomable=true %}

The four main measurements are:

1. **DDP uses 11.76 GiB of resident memory, close to the 16Ψ ledger:** 4Ψ for fp32 parameters, 4Ψ for gradient buckets and 8Ψ for Adam states, totaling 11.5 GiB before small overheads. `gradient_as_bucket_view` keeps the bucket memory resident. FSDP2 uses **1.27 GiB**, close to $(4\Psi+8\Psi)/8 = 1.08$ GiB plus buffers. This is a **9.3× reduction**, consistent with sharding the full model-state ledger across 8 GPUs.
2. **DDP exposes about 324 ms of communication:** 477 ms total minus the 153 ms single-GPU compute baseline. Post #1 predicts 304 ms for an all-reduce of 3.1 GB of fp32 gradients at 10.2 GB/s, a difference of 6%. The high cost of fp32 reduction on this PCIe machine is the main reason FSDP2, at 338 ms, is faster than DDP. FSDP2 all-gathers bf16 parameters and streams gradient reduce-scatters by bucket.
3. **Setting `reshard_after_forward=False` saves 64.7 ms** by avoiding the backward all-gather. The expected time for gathering one 1.55 GB bf16 model at 21.8 GB/s is about 71 ms, a difference of 9%. Peak memory increases by 1.32 GiB, close to the 1.44 GiB size of one bf16 model. Both effects match the estimates from post #1 within 10%.
4. FSDP2 uses 1.27 GiB of resident memory, about **half of DeepSpeed stage 3's 2.53 GiB** in post #3. Native per-parameter DTensors require less additional buffering and bookkeeping than the measured external implementation.

> **Boundary of the comparison.** DDP uses fp32 parameters with classic mixed precision. Pure-bf16 parameter training would roughly halve its parameter memory and gradient communication, reducing the gap. This machine also has no NVLink, so throughput differences would be smaller on NVLink systems.

## 6. Summary

1. FSDP provides native ZeRO-3-style sharding. FSDP1 flattens parameters and divides them by element count, so a shard may split matrix rows. FSDP2 shards each parameter along dim-0, so **each shard contains complete rows and remains a matrix**.
2. FSDP gathers parameters on demand, prefetches the next unit and reshards completed units. `reshard_after_forward` selects whether parameters remain gathered for backward. The measured time and memory differences match estimates from post #1 within 10%.
3. Resident memory falls from 11.76 to 1.27 GiB, a 9.3× reduction consistent with sharding a 16Ψ ledger across 8 GPUs. FSDP2 uses about half the resident memory measured for DeepSpeed stage 3 and is faster than DDP on this PCIe machine, where fp32 reduction is expensive.
4. The complete-row structure of FSDP2 shards enables the communication-free row-local optimizer operations studied in post #9.

**Next comes Tensor Parallelism and Megatron's two matrix cuts.** The methods covered so far shard storage while each layer still performs its full computation. TP instead divides **the computation of a single layer**. The next post explains how column and row cuts compose without communication between them, introduces the conjugate operators $f/g$, and derives the cost of four activation all-reduces per layer and training step.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `torchrun --standalone --nproc_per_node=8 bench_fsdp.py --mode {ddp,fsdp2} [--no-reshard]`. Plotting and schematic code accompanies the series.*

*All benchmark scripts, schematic generators, plotting code and raw result CSVs for this post live in [assets/blog/code/04-fsdp](https://github.com/Dominator-Index/Dominator-Index.github.io/tree/main/assets/blog/code/04-fsdp).*

---

## Appendix: The Code That Ran

Every number in this post comes from the scripts below, embedded verbatim. Plotting and schematic code plus the raw result CSVs live in the folder linked above.

<details markdown="1">
<summary><code>bench_fsdp.py</code></summary>

```python
"""
FSDP experiments (Distributed Training Illustrated, post 04): FSDP2 (fully_shard) vs DDP, GPT-2 Large.

Measures three things:
  1. Per-GPU memory of FSDP2 vs DDP (compare with post 03's deepspeed stage 3 / stage 0)
  2. reshard_after_forward ablation (True = ZeRO-3 semantics / False = ZeRO-2 semantics: skip resharding after forward, saving one all-gather in backward)
  3. Throughput comparison

Usage:
  torchrun --standalone --nproc_per_node=8 bench_fsdp.py --mode {ddp,fsdp2} [--no-reshard] --out ../results/fsdp.csv
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, "/jumbo/yaoqingyang/ouyangzhuoli/MARS/MARS")
from model import GPT, GPTConfig, Block  # noqa: E402

MBS, BLOCK = 4, 1024
WARMUP, STEPS = 5, 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ddp", "fsdp2"], required=True)
    ap.add_argument("--no-reshard", action="store_true")
    ap.add_argument("--out", default="../results/fsdp.csv")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    torch.manual_seed(1337)
    model = GPT(GPTConfig(n_layer=36, n_head=20, n_embd=1280, dropout=0.0)).to(device)

    if args.mode == "ddp":
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        # FSDP2: fully_shard each Block (unit = transformer block), bf16 compute + fp32 reduce
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        reshard = not args.no_reshard
        for block in model.transformer.h:
            fully_shard(block, mp_policy=mp, reshard_after_forward=reshard)
        fully_shard(model, mp_policy=mp, reshard_after_forward=reshard)
        import contextlib
        autocast = contextlib.nullcontext()  # MixedPrecisionPolicy already handles precision

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    X = torch.randint(0, 50304, (MBS, BLOCK), device=device)
    Y = torch.randint(0, 50304, (MBS, BLOCK), device=device)

    def one_step():
        with autocast:
            _, loss = model(X, Y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        one_step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / STEPS
    peak = torch.cuda.max_memory_allocated() / 2**30
    resident = torch.cuda.memory_allocated() / 2**30

    tps = MBS * BLOCK * world / dt / 1e3
    if rank == 0:
        label = args.mode + ("-noreshard" if (args.mode == "fsdp2" and args.no_reshard) else "")
        row = [label, world, round(resident, 2), round(peak, 2), round(dt * 1e3, 1), round(tps, 1)]
        newfile = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["mode", "world", "resident_GiB", "peak_GiB", "step_ms", "ktok_per_s"])
            w.writerow(row)
        print("ROW:", row, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

</details>

