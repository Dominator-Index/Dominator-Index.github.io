"""
手写 GPipe(《图解分布式训练》第 07 篇):实测气泡比例 vs microbatch 数。

p 个 stage(每卡一个),总 batch 固定,microbatch 数 m 从 1 扫到 32:
  理论:T(m) = (m + p - 1) · t_slot   (t_slot = 一个 microbatch 的单 stage 时间)
  气泡比例 = (p-1) / (m + p - 1)
m=1 就是朴素模型并行(只有 1 张卡在干活);m 越大气泡越薄。

用法:
  torchrun --standalone --nproc_per_node=4 bench_gpipe.py --out ../results/gpipe.csv
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

D = 4096                # 隐宽
LAYERS_PER_STAGE = 6    # 每个 stage 的层数
TOTAL_ROWS = 8192       # 总 batch(行数),固定
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
    opt = torch.optim.SGD(stage.parameters(), lr=1e-4)  # 便宜的 step:只测调度几何

    def gpipe_step(m):
        rows = TOTAL_ROWS // m
        ins, outs, reqs = [], [], []
        # ---- 前向:m 个 microbatch 依次流过 ----
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
        # ---- 反向:倒序 ----
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

    # ---- t_slot:一个 microbatch 在本 stage 的 fwd+bwd 时间(无通信)----
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
            t_ideal = m * t_slot          # 无气泡、无通信的理想时间
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
