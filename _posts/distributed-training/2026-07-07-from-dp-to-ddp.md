---
layout: post
title: "From DP To DDP: Buckets, Overlap, And A 25 MB Sweet Spot"
date: 2026-07-07 10:00:00
description: "DP's exactness invariant, and how DDP hides the gradient all-reduce inside the backward pass: bucketing against the latency floor, overlap against the bandwidth bill — with a full bucket_cap_mb sweep on 8 GPUs."
tags: distributed-training deep-learning
categories: distributed-training
thumbnail: assets/img/blog/distributed/02/fig-2-bucket-overlap.png
toc:
  sidebar: left
related_posts: false
---

> Part 2 of **An Overview of Distributed Learning**. One post, one idea: **data parallelism's invariant, and how DDP buries the communication bill inside the backward pass.** [Post #1](/blog/2026/distributed-training-illustrated-1-collective-communication/) priced one all-reduce; this post watches PyTorch DDP pay that price — and haggle it down. Experiments: nanoGPT-124M on 8 GPUs, with a full `bucket_cap_mb` sweep.

## 1. The invariant: DP never approximates anything

Data parallelism (DP) is the most intuitive parallelism: replicate the model $$N$$ times, split the batch $$N$$ ways, let each GPU run forward/backward independently, then **average the gradients with one all-reduce** and apply identical updates.

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-1-dp-one-step.svg" class="img-fluid rounded" zoomable=true %}

Its correctness rests on one line of arithmetic. Split the global batch $$\mathcal{B}$$ into $$\mathcal{B}_0,\dots,\mathcal{B}_{N-1}$$; since the loss is a sample average,

$$
\nabla L_{\mathcal{B}}(\theta) \;=\; \frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}} \nabla \ell(x;\theta)
\;=\; \frac{1}{N}\sum_{k=0}^{N-1}\underbrace{\frac{1}{|\mathcal{B}_k|}\sum_{x\in\mathcal{B}_k} \nabla \ell(x;\theta)}_{\text{local gradient } g_k \text{ on rank } k}
$$

**The all-reduced average equals, bit for bit, the gradient one giant GPU would compute on the full batch.** So DP's semantic invariant is: replicas start from the same $$\theta_0$$, receive the same averaged gradient every step, apply the same update — **and stay bit-identical forever**. DP approximates nothing; it merely spreads out a big-batch computation. All that remains is an engineering question: *when and how to run that all-reduce so it doesn't block the pipeline.*

Let's also pre-compute the bill (formula from post #1): gradient bytes $$S$$, each GPU pays $$2\frac{N-1}{N}S$$ per step. Our experimental model, nanoGPT-124M, keeps fp32 gradients: $$S = 4\Psi \approx 498$$ MB. At the all-reduce algorithm bandwidth we measured last time (8 GPUs, ~10.1 GB/s), **one bare, un-hidden all-reduce ≈ 49 ms**. Remember that number — §4 will reconcile against it.

## 2. Two ways naive DP dies, two tricks that save DDP

The naive implementation (loop over parameters after `backward()`, `all_reduce` each `.grad`) dies of two ailments, both foreshadowed in post #1:

1. **The latency floor**: GPT-2 has ~150 parameter tensors, mostly KBs to a few MBs. Per-tensor all-reduce hits the ~65 µs floor ($$N{=}8$$) every time — ~10 ms of pure latency;
2. **Serial exposure**: waiting for the whole backward before communicating leaves the entire 49 ms naked on the critical path.

`torch.nn.parallel.DistributedDataParallel` answers with two tricks, one figure:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-2-bucket-overlap.svg" class="img-fluid rounded" zoomable=true %}

- **Bucketing**: parameters are grouped (in reverse order of arrival during backward) into buckets of `bucket_cap_mb`; one bucket = one all-reduce. "150 small collectives" becomes "~20 large ones" — the latency cost divides by 7;
- **Overlap**: backward naturally computes gradients **last layer first**, so later layers' gradients are ready early. DDP registers an autograd hook per parameter; the instant a bucket's gradients are all ready, its all-reduce launches asynchronously on a **separate NCCL stream**, running concurrently with the still-ongoing backward of earlier layers. Ideally, by the time backward finishes, most communication is already done or in flight — only the last bucket's tail is exposed.

The trade-off is drawn in the figure too: **too small a bucket falls back onto the per-collective latency floor; too big a bucket leaves nothing to overlap with.** The sweet spot must be measured — that is §4.

## 3. Reading along in real source

- Bucket assignment: `_compute_bucket_assignment_by_size` in `torch.distributed` (first bucket defaults to 1 MB, the rest to `bucket_cap_mb` = 25 MB) — the first bucket is deliberately tiny so the last layer's gradients depart as early as possible;
- Hooks and async reduction: `torch/csrc/distributed/c10d/reducer.cpp` (`Reducer::autograd_hook` → `mark_variable_ready` → bucket-full → `all_reduce_bucket` on the comm stream);
- `gradient_as_bucket_view=True`: makes `.grad` a view into bucket memory — saves one copy and one gradient's worth of memory;
- The `no_sync()` context: `torch/nn/parallel/distributed.py` — disables the hooks' communication for the first gas−1 microbatches of gradient accumulation (§5);
- Our benchmark is the standard usage, whole script ships with the post:

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

1. **The U-curve** predicted in §2: 1 MB (too many buckets, latency) 152.7 ms → **25 MB optimal, 147.1 ms** (exactly PyTorch's default — it was not picked out of a hat) → 500 MB (whole model in one bucket) 172.0 ms;
2. **Reconciliation**: the single-bucket run = zero overlap costs 49.8 ms over the single-GPU floor — matching the "bare all-reduce ≈ 49 ms" priced in §1 from post #1's bandwidth measurements, to 2%. **Two posts, two independent experiments, one number**;
3. **Overlap's net gain**: best bucket (147.1 ms) vs zero overlap (172.0 ms) = 24.9 ms saved — **exactly half of the 49.8 ms bill hidden**. On this PCIe box, where communication is expensive relative to compute (49.8/122.2 ≈ 41%), hiding half is a solid result; on NVLink machines the bill itself is smaller and hides more completely.

Scaling efficiency: $$\frac{668.2}{8 \times 100.6} = 83\%$$. The missing 17% is precisely the ~25 ms of exposed communication.

## 5. Experiment 2: gradient accumulation and `no_sync` — pay fewer bills

Gradient accumulation (gas microbatches per optimizer step) is mathematically a free batch-size multiplier:

$$
g \;=\; \frac{1}{\text{gas}}\sum_{m=1}^{\text{gas}} g^{(m)}
\quad\Longleftrightarrow\quad
\text{backward } \tfrac{\ell^{(m)}}{\text{gas}} \text{ per microbatch, accumulate into .grad}
$$

Summation commutes with all-reduce: **accumulating locally and synchronizing once** gives the identical result as synchronizing every microbatch — but pays 1 bill instead of gas. DDP's hooks fire on every `backward()`, so the redundant syncs must be turned off explicitly with `no_sync()`:

{% include figure.liquid loading="eager" path="assets/img/blog/distributed/02/fig-4-throughput.svg" class="img-fluid rounded" zoomable=true %}

- gas=4 without `no_sync`: 674.8 ktok/s — three of the four syncs are pure waste;
- gas=4 with `no_sync`: **754.1 ktok/s (94% scaling efficiency)**, 521.5 ms/step — 61.3 ms saved ≈ 3 × 20 ms (the three skipped, already-partially-overlapped all-reduces);
- Intuition: `no_sync` amortizes the bill over 4× the tokens, dropping the comm/compute ratio from 41% to ~10%. Larger gas approaches the ideal line — one reason large-model training always ships with gradient accumulation.

> Honest boundary: each skipped sync saves ~20 ms, not the full 25 ms of exposed time, because the skipped all-reduces were themselves partially hidden by overlap; and under `no_sync`, `.grad` must accumulate locally, which claws back some of `gradient_as_bucket_view`'s memory savings.

## 6. DP's ceiling: it saves not one byte of memory

DDP solves communication's **time** problem but does nothing for **memory**: every GPU still carries the full 16Ψ ledger from [post #0](/blog/2026/distributed-training-illustrated-0-the-5d-map/) — parameters, gradients, fp32 master weights, Adam state, all of it. Across 8 GPUs, the same optimizer state is stored 8 times.

**Next: Data Parallelism, part 2 — ZeRO's three-stage ledger.** Recall from post #1 that all-reduce = reduce-scatter + all-gather. ZeRO splits it open in the middle: after the reduce-scatter, each rank holds the complete sum of 1/N of the gradients — so let rank $$k$$ **update and store only** shard $$k$$ of the optimizer state, then all-gather the fresh parameters. Communication volume nearly unchanged; memory divided by $$N$$. We will measure per-GPU memory across stages 1/2/3.

## 7. Summary

1. DP's invariant: the averaged gradient equals the big-batch gradient bit-for-bit; replicas stay identical — correctness is free, and the only question is where to put the communication;
2. DDP's two tricks: **bucketing** against the latency floor (150 small collectives → ~20 large), **overlap** hides communication inside backward (autograd hooks + a separate NCCL stream);
3. Measured: bucket size is a U-curve with the 25 MB default sitting at the bottom; the single-bucket run's +49.8 ms reconciles with post #1's bandwidth data to 2%; the best bucket hides half the bill — 83% scaling efficiency;
4. Gradient accumulation demands `no_sync`: the bill amortizes by gas, 94% efficiency;
5. DP/DDP saves zero memory (16Ψ × N redundancy) — that is ZeRO's stage.

---

*Environment: 8× RTX PRO 6000 Blackwell, PyTorch 2.9.1, NCCL 2.27.5. Reproduce: `bench_ddp.py` (single-GPU baseline + bucket sweep + no_sync ablation); benchmark, plotting and schematic code accompanies the series.*
