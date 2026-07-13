"""
Toy Ring Attention implementation (part 06 of the Illustrated Distributed Training series).

Each rank holds 1/N of the sequence (its own Q_k, K_k, V_k). KV blocks travel
around the ring for N-1 hops while each rank accumulates locally with online
softmax. Mathematically this is exactly equivalent to full attention, not an
approximation.

Verification: compare the output against single-GPU full attention (fp32).
Timing: breakdown of compute (blockwise attention) vs comm (KV ring exchange).

Usage:
  torchrun --standalone --nproc_per_node={2,4,8} bench_ring_attention.py --out ../results/ringattn.csv
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist

B, NH, S_TOTAL, D = 1, 8, 8192, 64   # batch, heads, total sequence length, head dim
WARMUP, STEPS = 10, 30


def blockwise_update(O, m, l, Q, K, V):
    """Online softmax: absorb one new KV block. O: [*, sq, d], m/l: [*, sq, 1]"""
    S = Q @ K.transpose(-2, -1) / D**0.5            # [*, sq, skv]
    m_blk = S.max(dim=-1, keepdim=True).values
    m_new = torch.maximum(m, m_blk)
    scale = torch.exp(m - m_new)                     # rescale the old running stats
    p = torch.exp(S - m_new)                         # unnormalized weights of the new block
    l_new = l * scale + p.sum(dim=-1, keepdim=True)
    O_new = O * scale + p @ V
    return O_new, m_new, l_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/ringattn.csv")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    s_local = S_TOTAL // world
    torch.manual_seed(1337)  # same seed: every rank generates the same full QKV, then takes its own slice
    Qf = torch.randn(B, NH, S_TOTAL, D, device=device)
    Kf = torch.randn(B, NH, S_TOTAL, D, device=device)
    Vf = torch.randn(B, NH, S_TOTAL, D, device=device)
    sl = slice(rank * s_local, (rank + 1) * s_local)
    Q, K, V = Qf[:, :, sl].contiguous(), Kf[:, :, sl].contiguous(), Vf[:, :, sl].contiguous()

    # ---- single-GPU reference (non-causal, full softmax attention) ----
    ref = torch.softmax(Qf @ Kf.transpose(-2, -1) / D**0.5, dim=-1) @ Vf
    ref_local = ref[:, :, sl]

    nxt, prv = (rank + 1) % world, (rank - 1) % world

    def ring_attention():
        O = torch.zeros_like(Q)
        m = torch.full((B, NH, s_local, 1), -float("inf"), device=device)
        l = torch.zeros(B, NH, s_local, 1, device=device)
        k_cur, v_cur = K.clone(), V.clone()
        for step in range(world):
            if step < world - 1:  # send the current KV out first (a chance to overlap with compute, this toy version is serial)
                k_buf, v_buf = torch.empty_like(k_cur), torch.empty_like(v_cur)
                ops = [dist.P2POp(dist.isend, k_cur, nxt), dist.P2POp(dist.irecv, k_buf, prv),
                       dist.P2POp(dist.isend, v_cur, nxt), dist.P2POp(dist.irecv, v_buf, prv)]
                reqs = dist.batch_isend_irecv(ops)
            O, m, l = blockwise_update(O, m, l, Q, k_cur, v_cur)
            if step < world - 1:
                for r in reqs:
                    r.wait()
                k_cur, v_cur = k_buf, v_buf
        return O / l

    out = ring_attention()
    err = (out - ref_local).abs().max().item()

    # ---- timing breakdown ----
    def timed(fn):
        s_, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
        dist.barrier(); torch.cuda.synchronize()
        s_.record()
        for _ in range(STEPS):
            fn()
        e_.record(); torch.cuda.synchronize()
        return s_.elapsed_time(e_) / STEPS

    def compute_only():  # same number of blockwise_update calls, no communication
        O = torch.zeros_like(Q)
        m = torch.full((B, NH, s_local, 1), -float("inf"), device=device)
        l = torch.zeros(B, NH, s_local, 1, device=device)
        for _ in range(world):
            O, m, l = blockwise_update(O, m, l, Q, K, V)
        return O / l

    for _ in range(WARMUP):
        ring_attention()
    t_total = timed(ring_attention)
    t_comp = timed(compute_only)

    if rank == 0:
        row = [world, S_TOTAL, s_local, err, round(t_comp, 3), round(t_total - t_comp, 3),
               round(t_total, 3)]
        newfile = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["cp", "s_total", "s_local", "max_err", "compute_ms", "comm_ms", "total_ms"])
            w.writerow(row)
        print("ROW:", row, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
