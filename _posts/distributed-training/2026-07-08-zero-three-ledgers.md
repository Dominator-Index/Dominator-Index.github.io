---
layout: post
title: "ZeRO: Three Ledgers, Zero Redundancy"
date: 2026-07-08 10:00:00
description: "How ZeRO places local updates between reduce-scatter and all-gather, divides optimizer-state memory by N without extra communication, and differs from the theoretical ledger in real DeepSpeed measurements."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/03/fig-1-split-allreduce.png
toc:
  sidebar: left
related_posts: false
---

> ZeRO separates all-reduce into reduce-scatter and all-gather, then uses the intermediate sharded state to remove replication from the 16Ψ memory ledger. We measure per-GPU memory for GPT-2 Large (770M) across DeepSpeed stages 0, 1, 2 and 3 on 8 GPUs. Two results differ from the theoretical ledger and reveal how the implementation actually manages memory.

## 1. The problem: DP replicates what it shouldn't

[Post #2](/blog/2026/from-dp-to-ddp/) showed that every DDP rank stores the full ledger from [post #0](/blog/2026/why-a-single-gpu-is-never-enough/): 2Ψ bytes of bf16 parameters, 2Ψ bytes of bf16 gradients, and 12Ψ bytes of fp32 master parameters and Adam moments. The total is 16Ψ bytes per GPU. With eight GPUs, the same fp32 optimizer state is stored eight times.

The largest replicated component is the 12Ψ optimizer state, which is used only during `optimizer.step()`. Adam updates each parameter independently: the update for parameter $i$ depends only on its gradient and its own $m$ and $v$ states. The work can therefore be partitioned by parameter. If rank $k$ updates parameter shard $k$, it needs only the corresponding optimizer-state shard.

The remaining question is how rank $k$ obtains the **fully reduced gradient for shard $k$**. The reduce-scatter operation from post #1 provides exactly this output.

## 2. ZeRO's key move: split the all-reduce, work in the middle

Post #1 showed that **all-reduce = reduce-scatter + all-gather**. After reduce-scatter, rank $k$ holds the complete reduced gradient for shard $k$. ZeRO-1 uses that intermediate result to update the matching parameter and optimizer-state shard locally before the all-gather.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/03/fig-1-split-allreduce.svg" class="img-fluid rounded" zoomable=true %}

$$
\underbrace{\text{all-reduce}(g)\;\to\;\text{everyone updates all }\theta}_{\text{DDP: every rank needs all }12\Psi\text{ of state}}
\;\Longrightarrow
\underbrace{\text{reduce-scatter}(g)\;\to\;\text{local update of shard }k\;\to\;\text{all-gather}(\theta')}_{\text{ZeRO-1: each rank needs }12\Psi/N}
$$

The communication volume is unchanged. Reduce-scatter sends $\frac{N-1}{N}S$ bytes per GPU, and all-gather sends the same amount, for a total of $2\frac{N-1}{N}S$. This is exactly the cost of the all-reduce they replace. Optimizer-state memory falls from 12Ψ to 12Ψ/N without adding communication, which is the basis of the name *Zero Redundancy Optimizer*.

> The semantic difference is that all-gather now carries **updated parameters** $\theta'$ rather than gradients. Operations such as gradient clipping still need a global gradient norm. This requires a small additional reduction, but only a constant number of scalars per rank because a squared norm is additive across shards.

The three ZeRO stages apply sharding cumulatively to optimizer states, gradients and parameters:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/03/fig-2-three-stage-ledger.svg" class="img-fluid rounded" zoomable=true %}

| Stage | Shards | Per-GPU holds | Added communication |
|-------|--------|---------------|---------------------|
| 1 | optimizer state (12Ψ) | $$2+2+\frac{12}{N}$$ | 0 (the identity above) |
| 2 | + gradients (2Ψ) | $$2+\frac{2}{N}+\frac{12}{N}$$ | 0 (grads wanted a reduce-scatter anyway) |
| 3 | + parameters (2Ψ) | $$\frac{16}{N}$$ | params all-gathered per layer in fwd & bwd: ≈ $$+\frac{N-1}{N}\cdot 2\Psi$$ per step |

Stage 3 changes the execution model because no rank stores complete parameters between operations. When the forward or backward pass reaches a layer, its parameter shards are all-gathered, used, and then released. Model-state memory is divided by $N$, while communication increases by roughly 50%. In the units from post #1, the cost is $2\frac{N-1}{N}S_g + 2\frac{N-1}{N}S_\theta \approx 3\frac{N-1}{N}S$.

## 3. Reading along in real source

**DeepSpeed** implements stages 1 and 2 in `deepspeed/runtime/zero/stage_1_and_2.py`. `DeepSpeedZeroOptimizer` performs bucketed reduce-scatter and routes shards to their owning ranks through `average_tensor`. Stage 3 uses `stage3.py` together with `partitioned_param_coordinator.py` for on-demand all-gather and prefetch. Its configuration is defined in `zero/config.py`.

**nanotron** offers a minimal ZeRO-1 reference: `ZeroDistributedOptimizer` in `src/nanotron/optim/zero.py`, with param groups split by dp rank.

**PyTorch provides native counterparts.** `torch.distributed.optim.ZeroRedundancyOptimizer` is similar to ZeRO-1, while FSDP implements ZeRO-3-style parameter sharding and is the subject of the next post.

**Our benchmark** keeps the model and workload fixed while changing only `zero_optimization.stage`. It records resident and peak memory separately. The full script is included with the post.

## 4. Experiment: the ladder, measured (GPT-2 Large, 8 GPUs)

**Setup:** Ψ = 0.774B, bf16 training, torch AdamW with fp32 states, a 4×1024 microbatch, DeepSpeed 0.19.2 and `overlap_comm=True`. We report **resident memory**, measured by `memory_allocated` after the step, and **peak memory**, measured by `max_memory_allocated` during the step. Peak memory also includes activations and communication buffers.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/03/fig-3-memory-ladder.svg" class="img-fluid rounded" zoomable=true %}

**Results that match the theoretical ledger:**

1. **Memory falls by 7.79 GiB from stage 0 to stage 1.** The theoretical reduction is $12\Psi \times \frac{7}{8} = 7.57$ GiB, a difference of 3%.
2. Stage 0 uses 10.47 GiB of resident memory, close to $14\Psi$: 2Ψ for bf16 parameters and 12Ψ for fp32 optimizer state. Stage 1 uses 2.68 GiB, close to $3.5\Psi$. Both are within 0.2 GiB of a ledger that treats gradients as transient.

**Results that differ from the simplified ledger:**

3. **Stages 1 and 2 have identical measurements:** 2.68 GiB resident and 15.01 GiB peak. The theoretical table predicts that stage 2 should save another $2\Psi\times\frac{7}{8}\approx 1.3$ GiB by sharding gradients. In this implementation, however, DeepSpeed already streams gradients through bucketed reduce-scatter and frees each bucket after use. Full gradients are therefore transient rather than resident, so stage 1 already obtains most of the memory reduction attributed to stage 2.
4. **Stage 3 uses 2.53 GiB of resident memory, while the basic ledger predicts $1.75\Psi \approx 1.35$ GiB.** The additional ~1.2 GiB comes from all-gather workspaces and parameter-shard bookkeeping. Setting `stage3_param_persistence_threshold` to 0 does not change the measurement. For this 770M model, framework overhead nearly removes stage 3's memory advantage over stage 2. Sharding the 2Ψ parameter term becomes more valuable as model size grows, which motivates the common practice of using ZeRO-2 until parameter replication no longer fits.
5. Peak memory remains between 15 and 22 GiB because **ZeRO does not shard activations**, which use about 12 GiB here without checkpointing. Activation memory can instead be reduced with gradient checkpointing or sequence sharding (post #6).

### Step time

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/03/fig-4-step-time.svg" class="img-fluid rounded" zoomable=true %}

Stage 3 is the fastest configuration in this experiment, at 276 ms per step compared with 360 ms for stage 0. Exposed communication takes about 190–207 ms in stages 0, 1 and 2, but only 123 ms in stage 3. Its **per-layer prefetch** overlaps parameter all-gathers with layer computation. This is the reverse of DDP's overlap pattern: DDP overlaps outgoing gradients, while ZeRO-3 overlaps incoming parameters. These timings come from a 770M model on a PCIe-only machine with a high communication-to-compute ratio, so they should not be extrapolated directly to NVLink clusters. The general result is that sharding need not reduce throughput when communication is overlapped effectively.

## 5. Summary

1. ZeRO uses the sequence reduce-scatter → **local update** → all-gather. It preserves the communication volume of all-reduce while reducing optimizer-state memory from 12Ψ to 12Ψ/N.
2. The stages are cumulative. Stage 1 shards optimizer states without extra communication, stage 2 also shards gradients, and stage 3 shards parameters at roughly 50% additional communication. Stage 3 divides the full model-state ledger by $N$.
3. The measured stage-1 reduction matches theory within 3%. Stages 1 and 2 use the same memory because DeepSpeed streams gradient buckets, while stage 3 adds about 1.2 GiB of framework buffers. These differences show which tensors are resident and which are transient in the implementation.
4. Activations are a separate ledger, untouched by ZeRO.
5. Per-layer prefetch makes stage 3 faster in this experiment. PyTorch provides a native implementation of the same execution model through FSDP.

**Next comes FSDP, PyTorch's native implementation of ZeRO-3-style sharding.** The next post compares FlatParameter in FSDP1 with per-parameter DTensor sharding along dim-0 in FSDP2. It also explains the prefetch and reshard timeline and why row-aligned sharding matters for the final post in this series.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, DeepSpeed 0.19.2, NCCL 2.27.5. Reproduce: `deepspeed --num_gpus=8 bench_zero.py --stage {0,1,2,3}`. The single-GPU baseline and plotting code accompany the series.*

*All benchmark scripts, schematic generators, plotting code and raw result CSVs for this post live in [assets/blog/code/03-zero](https://github.com/Dominator-Index/Dominator-Index.github.io/tree/main/assets/blog/code/03-zero).*

---

## Appendix: The Code That Ran

Every number in this post comes from the scripts below, embedded verbatim. Plotting and schematic code plus the raw result CSVs live in the folder linked above.

<details markdown="1">
<summary><code>bench_zero.py</code></summary>

```python
"""
ZeRO experiments (Distributed Training Illustrated, post 03): deepspeed stage 0/1/2/3 per-GPU memory + throughput.

Model: GPT-2 Large (770M, Psi=0.77e9), bf16 training + fp32 optimizer state.
Theoretical ledger (per GPU, 8 GPUs, bytes):
  stage 0: 2Psi (bf16 params) + 2Psi (bf16 grads) + 12Psi (fp32 master+m+v)  = 16Psi ~ 12.3 GB
  stage 1: 2Psi + 2Psi + 12Psi/8                                             ~  4.6 GB
  stage 2: 2Psi + 2Psi/8 + 12Psi/8                                           ~  2.9 GB
  stage 3: 2Psi/8 + 2Psi/8 + 12Psi/8                                         ~  1.5 GB
(activations etc. come on top, we reconcile against post-step resident memory and record peak separately)

Usage:
  deepspeed --num_gpus=8 bench_zero.py --stage {0,1,2,3} --out ../results/zero.csv
"""

import argparse
import csv
import os
import sys
import time

import torch
import deepspeed

sys.path.insert(0, "/jumbo/yaoqingyang/ouyangzhuoli/MARS/MARS")
from model import GPT, GPTConfig  # noqa: E402

MBS, BLOCK = 4, 1024
WARMUP, STEPS = 5, 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--persist0", action="store_true")
    ap.add_argument("--out", default="../results/zero.csv")
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()

    ds_config = {
        "train_micro_batch_size_per_gpu": MBS,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": args.stage,
            "overlap_comm": True,
            **({"stage3_param_persistence_threshold": 0} if args.persist0 else {}),
        },
        "wall_clock_breakdown": False,
    }

    torch.manual_seed(1337)
    model = GPT(GPTConfig(n_layer=36, n_head=20, n_embd=1280, dropout=0.0))
    n_params = sum(p.numel() for p in model.parameters())
    # Pass a torch AdamW instance to avoid deepspeed FusedAdam's JIT build (local compiler is incompatible)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_config)
    rank = engine.global_rank
    world = engine.world_size
    device = engine.device

    X = torch.randint(0, 50304, (MBS, BLOCK), device=device)
    Y = torch.randint(0, 50304, (MBS, BLOCK), device=device)

    def one_step():
        _, loss = engine(X, Y)
        engine.backward(loss)
        engine.step()

    for _ in range(WARMUP):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        one_step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / STEPS
    peak = torch.cuda.max_memory_allocated() / 2**30
    resident = torch.cuda.memory_allocated() / 2**30  # resident after the step (params + grad buffers + optimizer state)

    tps = MBS * BLOCK * world / dt / 1e3
    if rank == 0:
        row = [args.stage, world, n_params, round(resident, 2), round(peak, 2),
               round(dt * 1e3, 1), round(tps, 1)]
        newfile = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["stage", "world", "n_params", "resident_GiB", "peak_GiB", "step_ms", "ktok_per_s"])
            w.writerow(row)
        print("ROW:", row, flush=True)


if __name__ == "__main__":
    main()
```

</details>

<details markdown="1">
<summary><code>single_large_baseline.py</code></summary>

```python
import sys, time, torch
sys.path.insert(0, "/jumbo/yaoqingyang/ouyangzhuoli/MARS/MARS")
from model import GPT, GPTConfig
torch.manual_seed(1337)
m = GPT(GPTConfig(n_layer=36, n_head=20, n_embd=1280, dropout=0.0)).cuda().bfloat16()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True)
X = torch.randint(0, 50304, (4, 1024), device="cuda"); Y = torch.randint(0, 50304, (4, 1024), device="cuda")
def step():
    _, loss = m(X, Y); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
for _ in range(5): step()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(15): step()
torch.cuda.synchronize()
print(f"single-GPU GPT-Large mbs4 bf16: {(time.perf_counter()-t0)/15*1e3:.1f} ms/step")
```

</details>

