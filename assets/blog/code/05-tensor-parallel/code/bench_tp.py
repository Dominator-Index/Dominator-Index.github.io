"""
TP 实验(《图解分布式训练》第 05 篇):手写 Megatron 式 Column+Row 并行 MLP。

验证两件事:
  1. 正确性:Column 切(fc1)+ Row 切(fc2)的 MLP,前向输出与反向梯度
     和单卡完全一致(fp64 下逐位,bf16 下到舍入误差)——"中间零通信"不是近似
  2. 代价:每层前向 1 次 all-reduce(g 算子),反向 1 次(f 算子);
     实测 TP=2/4/8 的每层时间分解(计算 vs 通信)

用法:
  torchrun --standalone --nproc_per_node={2,4,8} bench_tp.py --out ../results/tp.csv
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

H = 1280               # hidden(GPT-2 Large 口径)
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

    torch.manual_seed(1337)  # 所有 rank 同种子 → 权重一致
    # 完整权重(fp32 保证可对账)
    W1 = torch.randn(FF, H, device=device) / H**0.5      # fc1 [out=FF, in=H]
    W2 = torch.randn(H, FF, device=device) / FF**0.5     # fc2 [out=H, in=FF]
    X = torch.randn(MBS * SEQ, H, device=device, requires_grad=True)

    # ---- 单卡参考 ----
    ref = F.gelu(X @ W1.t()) @ W2.t()
    ref_loss = ref.square().mean()
    ref_loss.backward()
    ref_grad = X.grad.clone()
    X.grad = None

    # ---- TP:Column 切 W1(沿 out 维),Row 切 W2(沿 in 维) ----
    shard = FF // world
    W1_k = W1[rank * shard:(rank + 1) * shard]            # [FF/N, H] 完整的行
    W2_k = W2[:, rank * shard:(rank + 1) * shard]         # [H, FF/N] 列切

    Xtp = X.detach().clone().requires_grad_(True)
    # 前向:f 算子 = identity(X 已复制);中间 Y_k 本地;g 算子 = all-reduce
    Y_k = F.gelu(Xtp @ W1_k.t())                          # [B, FF/N] 中间激活:零通信
    Z_k = Y_k @ W2_k.t()                                  # [B, H] 部分和
    Z = Z_k.clone()
    dist.all_reduce(Z)                                    # g:前向唯一一次通信
    loss = Z.square().mean()
    loss.backward()                                       # 反向:dX 是部分和
    gX = Xtp.grad.clone()
    dist.all_reduce(gX)                                   # f 的反向:all-reduce dX

    fwd_err = (Z - ref).abs().max().item()
    grad_err = (gX - ref_grad).abs().max().item()

    # ---- 计时:分解计算与通信 ----
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
