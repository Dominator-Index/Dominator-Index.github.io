"""
集合通信原语基准测试(《图解分布式训练》第 01 篇)。

测 6 种原语:broadcast / scatter / gather / all_gather / reduce_scatter / all_reduce
口径(与正文一致):
  - S = 逻辑消息大小(完整张量的字节数)
  - algbw = S / t                          (算法带宽,"用户视角")
  - busbw = algbw × 校正因子               (总线带宽,"硬件视角",nccl-tests 口径)
      all_reduce:      2(N-1)/N
      all_gather:      (N-1)/N
      reduce_scatter:  (N-1)/N
      broadcast:       1
      scatter/gather:  (N-1)/N

用法:
  torchrun --standalone --nproc_per_node=8 bench_collectives.py --out ../results/collectives_n8.csv
"""

import argparse
import csv
import os

import torch
import torch.distributed as dist

SIZES = [4 * 2**10, 64 * 2**10, 2**20, 16 * 2**20, 256 * 2**20, 2**30]  # 4KiB..1GiB
DTYPE = torch.bfloat16


def bus_factor(op, n):
    return {
        "all_reduce": 2 * (n - 1) / n,
        "all_gather": (n - 1) / n,
        "reduce_scatter": (n - 1) / n,
        "broadcast": 1.0,
        "scatter": (n - 1) / n,
        "gather": (n - 1) / n,
    }[op]


def make_op(op, size_bytes, rank, world, device):
    """返回 (fn, actually_allocated_ok)。size_bytes 是逻辑消息大小 S。"""
    numel = size_bytes // DTYPE.itemsize
    # 保证能被 world 整除(scatter/gather/AG/RS 需要)
    numel = (numel // world) * world
    if numel == 0:
        return None

    if op == "all_reduce":
        t = torch.randn(numel, dtype=DTYPE, device=device)
        return lambda: dist.all_reduce(t)

    if op == "broadcast":
        t = torch.randn(numel, dtype=DTYPE, device=device)
        return lambda: dist.broadcast(t, src=0)

    if op == "all_gather":
        out = torch.empty(numel, dtype=DTYPE, device=device)
        inp = torch.randn(numel // world, dtype=DTYPE, device=device)
        return lambda: dist.all_gather_into_tensor(out, inp)

    if op == "reduce_scatter":
        inp = torch.randn(numel, dtype=DTYPE, device=device)
        out = torch.empty(numel // world, dtype=DTYPE, device=device)
        return lambda: dist.reduce_scatter_tensor(out, inp)

    if op == "scatter":
        out = torch.empty(numel // world, dtype=DTYPE, device=device)
        if rank == 0:
            chunks = list(torch.randn(numel, dtype=DTYPE, device=device).chunk(world))
            return lambda: dist.scatter(out, chunks, src=0)
        return lambda: dist.scatter(out, None, src=0)

    if op == "gather":
        inp = torch.randn(numel // world, dtype=DTYPE, device=device)
        if rank == 0:
            outs = list(torch.empty(numel, dtype=DTYPE, device=device).chunk(world))
            return lambda: dist.gather(inp, outs, dst=0)
        return lambda: dist.gather(inp, None, dst=0)

    raise ValueError(op)


def bench(fn, iters, device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    ops = ["broadcast", "scatter", "gather", "all_gather", "reduce_scatter", "all_reduce"]
    rows = []
    for op in ops:
        for size in SIZES:
            fn = make_op(op, size, rank, world, device)
            if fn is None:
                continue
            warmup = 20 if size < 256 * 2**20 else 5
            iters = 50 if size < 256 * 2**20 else 10
            for _ in range(warmup):
                fn()
            t_ms = bench(fn, iters, device)
            algbw = size / (t_ms / 1e3) / 1e9  # GB/s
            busbw = algbw * bus_factor(op, world)
            if rank == 0:
                rows.append([op, world, size, round(t_ms, 4), round(algbw, 2), round(busbw, 2)])
                print(f"{op:15s} N={world} S={size/2**20:9.3f}MiB  t={t_ms:9.3f}ms  algbw={algbw:7.2f}GB/s  busbw={busbw:7.2f}GB/s", flush=True)
            del fn
            torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op", "world_size", "bytes", "time_ms", "algbw_GBps", "busbw_GBps"])
            w.writerows(rows)
        print(f"wrote {args.out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
