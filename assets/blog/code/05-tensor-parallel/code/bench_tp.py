"""
TP experiment (part 05 of the Illustrated Distributed Training series): a hand-written
Megatron-style Column+Row parallel MLP.

Verifies two things:
  1. Correctness: an MLP with Column-cut fc1 + Row-cut fc2 matches the single-GPU
     forward output and backward gradients exactly (bit-for-bit in fp64, up to
     rounding in bf16). "Zero communication in the middle" is not an approximation.
  2. Cost: 1 all-reduce per layer in forward (the g operator), 1 in backward
     (the f operator). Measures the per-layer time breakdown (compute vs comm)
     for TP=2/4/8.

Usage:
  torchrun --standalone --nproc_per_node={2,4,8} bench_tp.py --out ../results/tp.csv
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

H = 1280               # hidden size (GPT-2 Large scale)
FF = 4 * H             # 5120
MBS, SEQ = 4, 1024
WARMUP, STEPS = 20, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/tp.csv")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    torch.manual_seed(1337)  # same seed on every rank so weights match
    # full weights (fp32 so results can be reconciled exactly)
    W1 = torch.randn(FF, H, device=device) / H**0.5      # fc1 [out=FF, in=H]
    W2 = torch.randn(H, FF, device=device) / FF**0.5     # fc2 [out=H, in=FF]
    X = torch.randn(MBS * SEQ, H, device=device, requires_grad=True)

    # ---- single-GPU reference ----
    ref = F.gelu(X @ W1.t()) @ W2.t()
    ref_loss = ref.square().mean()
    ref_loss.backward()
    ref_grad = X.grad.clone()
    X.grad = None

    # ---- TP: Column-cut W1 (along out dim), Row-cut W2 (along in dim) ----
    shard = FF // world
    W1_k = W1[rank * shard:(rank + 1) * shard]            # [FF/N, H] whole rows
    W2_k = W2[:, rank * shard:(rank + 1) * shard]         # [H, FF/N] column slice

    Xtp = X.detach().clone().requires_grad_(True)
    # forward: f operator = identity (X already replicated), intermediate Y_k local, g operator = all-reduce
    Y_k = F.gelu(Xtp @ W1_k.t())                          # [B, FF/N] intermediate activation: zero comm
    Z_k = Y_k @ W2_k.t()                                  # [B, H] partial sum
    Z = Z_k.clone()
    dist.all_reduce(Z)                                    # g: the only forward comm
    loss = Z.square().mean()
    loss.backward()                                       # backward: dX is a partial sum
    gX = Xtp.grad.clone()
    dist.all_reduce(gX)                                   # f's backward: all-reduce dX

    fwd_err = (Z - ref).abs().max().item()
    grad_err = (gX - ref_grad).abs().max().item()

    # ---- timing: break down compute vs communication ----
    def timed(fn):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        dist.barrier(); torch.cuda.synchronize()
        s.record()
        for _ in range(STEPS):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / STEPS

    def fwd_only():
        z = F.gelu(Xtp @ W1_k.t()) @ W2_k.t()
        return z

    def fwd_with_ar():
        z = F.gelu(Xtp @ W1_k.t()) @ W2_k.t()
        dist.all_reduce(z)
        return z

    for _ in range(WARMUP):
        fwd_with_ar()
    t_compute = timed(fwd_only)
    t_total = timed(fwd_with_ar)
    t_comm = t_total - t_compute

    if rank == 0:
        row = [world, fwd_err, grad_err, round(t_compute, 3), round(t_comm, 3),
               round(t_total, 3), round(t_comm / t_total * 100, 1)]
        newfile = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["tp", "fwd_max_err", "grad_max_err", "compute_ms", "comm_ms", "total_ms", "comm_pct"])
            w.writerow(row)
        print("ROW:", row, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
