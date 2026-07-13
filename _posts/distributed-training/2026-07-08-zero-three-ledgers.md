---
layout: post
title: "ZeRO: Three Ledgers, Zero Redundancy"
date: 2026-07-08 10:00:00
description: "ZeRO splits the all-reduce open and inserts a local update in the middle, dividing optimizer state by N at zero extra communication, measured on GPT-2 Large across DeepSpeed stages 0-3 including two places where the paper's ledger and the implementation honestly disagree."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/03/fig-1-split-allreduce.png
toc:
  sidebar: left
related_posts: false
---

> ZeRO splits the all-reduce open so the redundant entries of the 16Ψ ledger get sharded across the DP group. The experiments run GPT-2 Large (770M) through DeepSpeed stages 0/1/2/3 on 8 GPUs with per-GPU memory measured, and they turn up two honest findings where theory and implementation disagree, instructively.

## 1. The problem: DP replicates what it shouldn't

[Post #2](/blog/2026/distributed-training-illustrated-2-from-dp-to-ddp/) ended with every DDP rank carrying the full ledger from [post #0](/blog/2026/distributed-training-illustrated-0-the-5d-map/): bf16 params 2Ψ, bf16 grads 2Ψ, fp32 master weights + Adam moments 12Ψ, totalling 16Ψ bytes. With eight GPUs, that means the same fp32 optimizer state is stored eight times.

The key observation is that **the biggest redundancy (12Ψ of optimizer state) is touched only during `optimizer.step()`, and the update is element-wise**. Parameter $$i$$'s Adam update depends only on parameter $$i$$'s gradient, $$m$$, and $$v$$. Element-wise means **the update work can be partitioned any way we like**, so let rank $$k$$ update only shard $$k$$ of the parameters, and it only ever needs shard $$k$$ of the state.

One question remains. How does rank $$k$$ get the **full gradient sum for shard $$k$$**? Post #1 already answered it.

## 2. ZeRO's key move: split the all-reduce, work in the middle

Post #1 showed that **all-reduce = reduce-scatter + all-gather**, and at the exact moment the reduce-scatter finishes, rank $$k$$ holds the complete sum of gradient shard $$k$$, which is precisely the input that "rank $$k$$ updates shard $$k$$" requires. That is ZeRO-1's entire wisdom. **Don't finish the all-reduce. Stop halfway and do the work.**

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/03/fig-1-split-allreduce.svg" class="img-fluid rounded" zoomable=true %}

$$
\underbrace{\text{all-reduce}(g)\;\to\;\text{everyone updates all }\theta}_{\text{DDP: every rank needs all }12\Psi\text{ of state}}
\;\Longrightarrow\.
\underbrace{\text{reduce-scatter}(g)\;\to\;\text{local update of shard }k\;\to\;\text{all-gather}(\theta')}_{\text{ZeRO-1: each rank needs }12\Psi/N}
$$

Reconcile with post #1's price list. Reduce-scatter $$\frac{N-1}{N}S$$ + all-gather $$\frac{N-1}{N}S$$ = $$2\frac{N-1}{N}S$$, which is **identical to the all-reduce it replaces**. Not one extra byte of communication, and 12Ψ becomes 12Ψ/N. Hence the name *zero redundancy*, because what is removed is pure redundancy, not something bought with bandwidth.

> The one semantic difference is that the all-gather now carries **updated parameters** $$\theta'$$ instead of gradients. Operations needing a global gradient norm (gradient clipping) require one extra tiny collective, but a norm is a sum of squares, a decomposable reduction that costs O(1) scalars per rank (post #1's old friend).

The three stages apply the same move to three objects, cumulatively:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/03/fig-2-three-stage-ledger.svg" class="img-fluid rounded" zoomable=true %}

| Stage | Shards | Per-GPU holds | Added communication |
|-------|--------|---------------|---------------------|
| 1 | optimizer state (12Ψ) | $$2+2+\frac{12}{N}$$ | 0 (the identity above) |
| 2 | + gradients (2Ψ) | $$2+\frac{2}{N}+\frac{12}{N}$$ | 0 (grads wanted a reduce-scatter anyway) |
| 3 | + parameters (2Ψ) | $$\frac{16}{N}$$ | params all-gathered per layer in fwd & bwd: ≈ $$+\frac{N-1}{N}\cdot 2\Psi$$ per step |

Stage 3 is the qualitative jump because parameters no longer live anywhere in full. As the forward pass reaches each layer, that layer's shards are all-gathered into a full matrix, used, and discarded (again in backward). Memory divides by $$N$$ outright, but communication grows ~50% (in post #1's units: $$2\frac{N-1}{N}S_g + 2\frac{N-1}{N}S_\theta \approx 3\frac{N-1}{N}S$$).

## 3. Reading along in real source

**DeepSpeed** keeps stages 1/2 in `deepspeed/runtime/zero/stage_1_and_2.py` (`DeepSpeedZeroOptimizer`: bucketed reduce-scatter, partition-ownership routing in `average_tensor`). Stage 3 is `stage3.py` plus `partitioned_param_coordinator.py` (all-gather-on-demand with prefetch), and the config surface is `zero/config.py`.

**nanotron** offers a minimal ZeRO-1 reference: `ZeroDistributedOptimizer` in `src/nanotron/optim/zero.py`, with param groups split by dp rank.

**PyTorch-native counterparts** exist too. ZeRO-1 ≈ `torch.distributed.optim.ZeroRedundancyOptimizer`, and ZeRO-3 ≈ FSDP, the next post's protagonist.

**Our benchmark** runs the same model while only `zero_optimization.stage` changes, with resident and peak memory recorded separately. The full script ships with the post.

## 4. Experiment: the ladder, measured (GPT-2 Large, 8 GPUs)

**Setup**: Ψ = 0.774B, bf16 training, torch AdamW (fp32 state), micro-batch 4×1024, DeepSpeed 0.19.2, `overlap_comm=True`. Two gauges: **resident** (`memory_allocated` after the step, so params + optimizer state) and **peak** (`max_memory_allocated` during the step, which adds activations and comm buffers).

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/03/fig-3-memory-ladder.svg" class="img-fluid rounded" zoomable=true %}

**The accounts that reconcile:**

1. **The stage-0 → 1 drop is 7.79 GiB**, and theory predicts $$12\Psi \times \frac{7}{8} = 7.57$$ GiB, a 3% error. ZeRO's core claim holds precisely.
2. Stage 0 resident is 10.47 GiB ≈ $$14\Psi$$ (2Ψ bf16 params + 12Ψ fp32 state) and stage 1 resident is 2.68 ≈ $$3.5\Psi$$. Both match the grads-are-transient ledger to within 0.2 GiB.

**The accounts that don't (more instructive):**

3. **Stage 1 and stage 2 measure identically** (resident 2.68 = 2.68, peak 15.01 = 15.01). The paper's table says stage 2 should save another $$2\Psi\times\frac{7}{8}\approx 1.3$$ GiB. Where did it go? **DeepSpeed's gradients are streamed anyway.** Bucketed reduce-scatter (post #2's old friend) frees each bucket as it passes through, so gradients never exist in full residence. The paper's ledger books gradients as a resident line item, but an efficient implementation books them as transient flow. Stage 2's savings were already collected by stage 1's plumbing.
4. **Stage 3 resides at 2.53 GiB where the bare ledger says $$1.75\Psi \approx 1.35$$ GiB.** The extra ~1.2 GiB is all-gather working buffers and bookkeeping for the parameter shards (measured invariant to `stage3_param_persistence_threshold`, since setting it to 0 changes nothing). On a 770M model, framework overhead nearly cancels stage 3's gain over stage 2, so **the 2Ψ parameter term only becomes worth sharding at 7B+ scale**, which is why "ZeRO-2 now, ZeRO-3 when the model outgrows it" is the standard playbook.
5. Peaks sit at 15–22 GiB because **activations (~12 GiB here, no checkpointing) are outside ZeRO's jurisdiction.** To cut them you need gradient checkpointing, or you shard the sequence dimension (post #6).

### And speed?

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/03/fig-4-step-time.svg" class="img-fluid rounded" zoomable=true %}

Sharding is not slower. Stage 3 is actually the *fastest* (276 ms vs stage 0's 360 ms). Exposed communication runs ~190–207 ms for stages 0/1/2 but only 123 ms for stage 3, whose **per-layer prefetch** hides the parameter all-gathers inside layer-by-layer compute. That is the same idea as DDP's bucket overlap with the direction reversed, because DDP hides departing gradients while ZeRO-3 hides arriving parameters. One caveat. This is a 770M model on a PCIe-only box with a high comm/compute ratio, so the numbers don't extrapolate to NVLink clusters. But the qualitative conclusion does travel. *Sharding is not slower, and overlap decides everything.*

## 5. Summary

1. ZeRO's heart is the identity all-reduce = reduce-scatter + **local update** + all-gather. Communication is unchanged, and optimizer state drops from 12Ψ to 12Ψ/N.
2. The three stages are cumulative. Stage 1 shards state for free, stage 2 shards grads for free, and stage 3 shards params at +50% comm for memory fully /N.
3. The measured stage-1 drop reconciles with theory to 3%, stage 1 = stage 2 because good implementations stream gradients anyway, and stage 3 carries ~1.2 GiB of framework buffers. The gaps between paper ledger and implementation are themselves the best guide to how the implementation works.
4. Activations are a separate ledger, untouched by ZeRO.
5. Stage 3's prefetch makes sharding *faster* here, but at that point it is effectively a different system, and PyTorch rebuilt it natively as FSDP.

**Next comes FSDP, which is how PyTorch implements ZeRO-3.** It covers FlatParameter (FSDP1) vs per-parameter DTensor sharded along dim-0 (FSDP2), the prefetch/reshard timeline, and why "row-aligned sharding", a seemingly innocuous implementation detail, becomes the foundation of this series' final post.

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

