"""
ZeRO 实验(《图解分布式训练》第 03 篇):deepspeed stage 0/1/2/3 每卡显存 + 吞吐。

模型:GPT-2 Large(770M,Ψ=0.77e9),bf16 训练 + fp32 优化器状态。
理论账本(每卡,8 卡,字节):
  stage 0: 2Ψ(bf16参数) + 2Ψ(bf16梯度) + 12Ψ(fp32 master+m+v)      = 16Ψ ≈ 12.3 GB
  stage 1: 2Ψ + 2Ψ + 12Ψ/8                                          ≈  4.6 GB
  stage 2: 2Ψ + 2Ψ/8 + 12Ψ/8                                        ≈  2.9 GB
  stage 3: 2Ψ/8 + 2Ψ/8 + 12Ψ/8                                      ≈  1.5 GB
(另加激活等;用 post-step 常驻显存对账,峰值另记)

用法:
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
    # 传 torch AdamW 实例,避开 deepspeed FusedAdam 的 JIT 编译(本机编译器不兼容)
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
    resident = torch.cuda.memory_allocated() / 2**30  # step 之后的常驻(参数+梯度buf+优化器状态)

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
