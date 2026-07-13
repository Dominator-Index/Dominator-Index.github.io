"""
GEMM 吞吐 vs 精度(《图解分布式训练》第 08 篇):同一个 8192^3 矩阵乘,
只换 dtype,实测 TFLOPS —— "为什么 99% 的计算要走低精度"的那笔账。

fp32   经典 fp32(关 TF32)
tf32   fp32 输入、Tensor Core TF32 路径(尾数截到 10 位)
bf16   bf16 输入、fp32 累加(Tensor Core)
fp16   fp16 输入、fp32 累加(Tensor Core)

用法:python bench_gemm.py --out ../results/gemm.csv
"""

import argparse
import csv
import os

import torch

N = 8192
WARMUP, STEPS = 10, 50
DEVICE = torch.device("cuda", 0)


def bench(dtype, allow_tf32):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    a = torch.randn(N, N, device=DEVICE, dtype=dtype)
    b = torch.randn(N, N, device=DEVICE, dtype=dtype)
    for _ in range(WARMUP):
        a @ b
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(STEPS):
        a @ b
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / STEPS
    tflops = 2 * N**3 / (ms / 1e3) / 1e12
    return ms, tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/gemm.csv")
    args = ap.parse_args()

    rows = []
    for name, dtype, tf32 in [("fp32", torch.float32, False),
                              ("tf32", torch.float32, True),
                              ("bf16", torch.bfloat16, False),
                              ("fp16", torch.float16, False)]:
        ms, tflops = bench(dtype, tf32)
        rows.append([name, round(ms, 3), round(tflops, 1)])
        print(f"{name}: {ms:.2f} ms, {tflops:.1f} TFLOPS", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dtype", "ms", "tflops"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
