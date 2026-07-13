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
