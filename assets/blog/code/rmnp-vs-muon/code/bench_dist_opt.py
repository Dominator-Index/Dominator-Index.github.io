"""
Measured communication ledger of distributed optimizer preconditioning (standalone
RMNP vs Muon post).

Setup: FSDP2-style dim-0 (row) sharding, all 2D matrices of GPT-2 Large (36 layers
x 4 each, hidden 1280), 8 GPUs. Only the precondition segment of the optimizer step
is measured (momentum update + normalization/NS + writing back params). Gradient
synchronization is excluded (identical for both).

Four schemes:
  rmnp_local    RMNP under row sharding: each GPU normalizes its own complete row block, 0 communication
  rmnp_colcut   RMNP under column-cut TP: local partial sums of squares + one [sum_m] vector all-reduce
  muon_sc       Muon synchronous compute: all-gather momentum into the full matrix, every GPU redundantly runs NS, take back its row block
  muon_rr       Muon round-robin: all-gather + only the owner runs NS + broadcast the update
                (2x communication, 1/N compute, the "amortized" route in the Moonlight/Canzona sense)

Also: a square-matrix size sweep (1k/2k/4k/8k) and a numerical check that sharded
RMNP == full-matrix RMNP.

Usage: torchrun --standalone --nproc_per_node=8 bench_dist_opt.py --out ../results
"""

import argparse
import csv
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

# GPT-2 Large: hidden 1280, 36 layers, 4 2D matrices per layer ([out, in])
H, L = 1280, 36
LAYER_SHAPES = [(3 * H, H), (H, H), (4 * H, H), (H, 4 * H)]  # qkv/attn.proj/c_fc/c_proj
SWEEP_SIZES = [1024, 2048, 4096, 8192]
NS_STEPS = 5
MU = 0.95
WARMUP, STEPS = 3, 10


def newtonschulz5(G, steps=NS_STEPS, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Bench:
    """Sharded state for one set of GPT-2 Large matrices (row-sharded and column-sharded copies, bf16)."""

    def __init__(self, device, world, rank):
        self.device, self.world, self.rank = device, world, rank
        self.shapes = LAYER_SHAPES * L
        g = torch.Generator(device="cpu").manual_seed(1337)
        # row sharding (FSDP2): [m/N, n] per GPU, column sharding (TP RowLinear): [m, n/N] per GPU
        self.row_W, self.row_M, self.row_G = [], [], []
        self.col_M, self.col_G = [], []
        for (m, n) in self.shapes:
            self.row_W.append(torch.randn(m // world, n, generator=g).bfloat16().to(device))
            self.row_M.append(torch.randn(m // world, n, generator=g).bfloat16().to(device))
            self.row_G.append(torch.randn(m // world, n, generator=g).bfloat16().to(device))
            self.col_M.append(torch.randn(m, n // world, generator=g).bfloat16().to(device))
            self.col_G.append(torch.randn(m, n // world, generator=g).bfloat16().to(device))
        self.total_params = sum(m * n for (m, n) in self.shapes)

    # ---- the schemes: each runs one full optimizer precondition step ----
    def rmnp_local(self):
        for W, M, G in zip(self.row_W, self.row_M, self.row_G):
            M.mul_(MU).add_(G)
            u = F.normalize(M.float(), p=2, dim=-1)  # within-row sums run in fp32 (the part 08 principle)
            scale = max(1.0, (M.shape[0] * self.world) / M.shape[1]) ** 0.5
            W.add_(u.bfloat16(), alpha=-3e-4 * scale)

    def rmnp_colcut(self):
        # local partial sums of squares, concat all matrices into one [sum_m] vector, one all-reduce
        partials = []
        for M, G in zip(self.col_M, self.col_G):
            M.mul_(MU).add_(G)
            partials.append(M.float().pow(2).sum(dim=-1))
        flat = torch.cat(partials)                      # [sum_m] about 410k fp32 values
        dist.all_reduce(flat)                           # the only communication
        idx = 0
        for M in self.col_M:
            m = M.shape[0]
            norms = flat[idx:idx + m].sqrt().clamp_min(1e-7)
            M.div_(norms.bfloat16().unsqueeze(-1))      # reused in place as the update
            idx += m

    def muon_sc(self):
        for i, (W, M, G) in enumerate(zip(self.row_W, self.row_M, self.row_G)):
            M.mul_(MU).add_(G)
            m, n = self.shapes[i]
            full = torch.empty(m, n, dtype=torch.bfloat16, device=self.device)
            dist.all_gather_into_tensor(full, M.contiguous())   # O(mn) communication
            u = newtonschulz5(full)                             # every GPU computes redundantly
            rows = slice(self.rank * (m // self.world), (self.rank + 1) * (m // self.world))
            scale = max(1.0, m / n) ** 0.5
            W.add_(u[rows], alpha=-3e-4 * scale)

    def muon_rr(self):
        for i, (W, M, G) in enumerate(zip(self.row_W, self.row_M, self.row_G)):
            M.mul_(MU).add_(G)
            m, n = self.shapes[i]
            owner = i % self.world
            full = torch.empty(m, n, dtype=torch.bfloat16, device=self.device)
            dist.all_gather_into_tensor(full, M.contiguous())   # O(mn)
            if self.rank == owner:
                full = newtonschulz5(full).contiguous()         # only the owner computes
            dist.broadcast(full, src=owner)                     # pay another O(mn)
            rows = slice(self.rank * (m // self.world), (self.rank + 1) * (m // self.world))
            scale = max(1.0, m / n) ** 0.5
            W.add_(full[rows], alpha=-3e-4 * scale)

    # ---- per-step communication volume per scheme (per GPU, bytes, ring/collective accounting, bf16=2B) ----
    def bytes_per_step(self, scheme):
        N = self.world
        if scheme == "rmnp_local":
            return 0
        if scheme == "rmnp_colcut":
            total_m = sum(m for (m, _) in self.shapes)
            return int(2 * (N - 1) / N * total_m * 4)           # fp32 vector all-reduce
        if scheme == "muon_sc":
            return int(sum((N - 1) / N * m * n * 2 for (m, n) in self.shapes))
        if scheme == "muon_rr":
            return int(sum(((N - 1) / N + 1) * m * n * 2 for (m, n) in self.shapes))


def time_fn(fn, device):
    for _ in range(WARMUP):
        fn()
    dist.barrier()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(STEPS):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / STEPS
    t = torch.tensor([ms], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)                    # take the slowest GPU
    return t.item()


def exactness_check(device, world, rank):
    """Row norms of sharded RMNP match full-matrix RMNP exactly (not an approximation)."""
    m, n = 512, 384
    g = torch.Generator(device="cpu").manual_seed(7)
    full = torch.randn(m, n, generator=g).to(device)
    ref = F.normalize(full, p=2, dim=-1)
    shard = full[rank * m // world:(rank + 1) * m // world]
    mine = F.normalize(shard, p=2, dim=-1)
    err = (mine - ref[rank * m // world:(rank + 1) * m // world]).abs().max().item()
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    err = exactness_check(device, world, rank)
    if rank == 0:
        print(f"[exactness] sharded vs full RMNP max |diff| = {err:.2e}", flush=True)

    bench = Bench(device, world, rank)
    if rank == 0:
        print(f"GPT-2 Large matrix set: {len(bench.shapes)} matrices, "
              f"{bench.total_params/1e6:.0f}M params", flush=True)

    rows = []
    for scheme in ["rmnp_local", "rmnp_colcut", "muon_sc", "muon_rr"]:
        ms = time_fn(getattr(bench, scheme), device)
        by = bench.bytes_per_step(scheme)
        rows.append([scheme, round(ms, 2), by, round(by / 2**20, 1)])
        if rank == 0:
            print(f"{scheme}: {ms:.2f} ms/step, comm {by/2**20:.1f} MiB/step/rank", flush=True)
    del bench
    torch.cuda.empty_cache()

    # ---- square-matrix size sweep: per-matrix precondition time ----
    sweep = []
    for s in SWEEP_SIZES:
        shard = torch.randn(s // world, s, device=device).bfloat16()
        full = torch.empty(s, s, dtype=torch.bfloat16, device=device)

        def rmnp_one():
            F.normalize(shard.float(), p=2, dim=-1)

        def muon_one():
            dist.all_gather_into_tensor(full, shard.contiguous())
            newtonschulz5(full)

        t_r = time_fn(rmnp_one, device)
        t_m = time_fn(muon_one, device)
        gather_bytes = int((world - 1) / world * s * s * 2)
        sweep.append([s, round(t_r, 3), round(t_m, 3), gather_bytes])
        if rank == 0:
            print(f"[{s}x{s}] rmnp {t_r:.3f} ms | muon gather+NS {t_m:.3f} ms", flush=True)
        del shard, full
        torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "dist_opt.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scheme", "ms_per_step", "bytes_per_step", "mib_per_step"])
            w.writerows(rows)
        with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["size", "rmnp_ms", "muon_ms", "gather_bytes"])
            w.writerows(sweep)
        with open(os.path.join(args.out, "exactness.txt"), "w") as f:
            f.write(f"sharded vs full RMNP max abs diff: {err:.3e}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
