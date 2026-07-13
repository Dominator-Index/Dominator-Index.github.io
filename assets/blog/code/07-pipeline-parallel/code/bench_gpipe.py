"""
Hand-written GPipe (part 07 of the Illustrated Distributed Training series): measure
the bubble fraction vs microbatch count.

p stages (one per GPU), total batch fixed, sweep microbatch count m from 1 to 32:
  theory: T(m) = (m + p - 1) * t_slot   (t_slot = single-stage time for one microbatch)
  bubble fraction = (p-1) / (m + p - 1)
m=1 is naive model parallelism (only 1 GPU working at a time). Larger m thins the bubble.

Usage:
  torchrun --standalone --nproc_per_node=4 bench_gpipe.py --out ../results/gpipe.csv
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

D = 4096                # hidden width
LAYERS_PER_STAGE = 6    # layers per stage
TOTAL_ROWS = 8192       # total batch (rows), fixed
M_LIST = [1, 2, 4, 8, 16, 32]
WARMUP, STEPS = 3, 10


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU())
            for _ in range(LAYERS_PER_STAGE)])

    def forward(self, x):
        return self.layers(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/gpipe.csv")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])       # = p
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    p = world

    torch.manual_seed(1337 + rank)
    stage = Stage().to(device).bfloat16()
    opt = torch.optim.SGD(stage.parameters(), lr=1e-4)  # cheap step: we only measure schedule geometry

    def gpipe_step(m):
        rows = TOTAL_ROWS // m
        ins, outs, reqs = [], [], []
        # ---- forward: m microbatches flow through one after another ----
        for i in range(m):
            if rank == 0:
                x = torch.randn(rows, D, device=device, dtype=torch.bfloat16)
                x.requires_grad_(True)
            else:
                x = torch.empty(rows, D, device=device, dtype=torch.bfloat16)
                dist.recv(x, src=rank - 1)
                x.requires_grad_(True)
            out = stage(x)
            if rank < p - 1:
                reqs.append(dist.isend(out.detach(), dst=rank + 1))
            ins.append(x)
            outs.append(out)
        for r_ in reqs:
            r_.wait()
        # ---- backward: reverse order ----
        for i in reversed(range(m)):
            if rank == p - 1:
                loss = outs[i].float().square().mean() / m
                loss.backward()
            else:
                g = torch.empty_like(outs[i])
                dist.recv(g, src=rank + 1)
                outs[i].backward(g)
            if rank > 0:
                dist.send(ins[i].grad, dst=rank - 1)
        opt.step()
        opt.zero_grad(set_to_none=True)

    # ---- t_slot: fwd+bwd time of one microbatch on this stage (no communication) ----
    def slot_time(m):
        rows = TOTAL_ROWS // m
        x = torch.randn(rows, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            out = stage(x)
            out.float().square().mean().backward()
            x.grad = None
            stage.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / STEPS * 1e3

    rows_out = []
    for m in M_LIST:
        for _ in range(WARMUP):
            gpipe_step(m)
        dist.barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            gpipe_step(m)
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) / STEPS * 1e3
        t_slot = slot_time(m)
        if rank == 0:
            bubble_theory = (p - 1) / (m + p - 1)
            t_ideal = m * t_slot          # ideal time with no bubble and no communication
            bubble_meas = 1 - t_ideal / step_ms
            rows_out.append([p, m, round(t_slot, 2), round(step_ms, 1),
                             round(bubble_theory, 3), round(bubble_meas, 3)])
            print("ROW:", rows_out[-1], flush=True)

    if rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["p", "m", "slot_ms", "step_ms", "bubble_theory", "bubble_measured"])
            w.writerows(rows_out)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
