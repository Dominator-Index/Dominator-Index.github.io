"""
DDP 实验(《图解分布式训练》第 02 篇):nanoGPT-124M 吞吐。

测三件事:
  1. 单卡 vs DDP(通信开销有多大,重叠回收了多少)
  2. bucket_cap_mb 扫描(分桶大小 vs 吞吐:延迟地板与重叠窗口的折中)
  3. 梯度累积时 no_sync 的效果(省掉 gas-1 次 all-reduce)

用法:
  # 单卡基线
  python bench_ddp.py --mode single --out ../results/ddp.csv
  # DDP + 桶扫描
  torchrun --standalone --nproc_per_node=8 bench_ddp.py --mode ddp --bucket-mb 25 --out ../results/ddp.csv
  # 梯度累积 ±no_sync
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
    ap.add_argument("--no-sync", type=int, default=1)  # 累积时是否用 no_sync
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
            # no_sync: 前 gas-1 个 microbatch 跳过 all-reduce
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
