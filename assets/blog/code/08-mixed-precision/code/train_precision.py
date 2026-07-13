"""
三精度对照训练(《图解分布式训练》第 08 篇):同一个模型、同一份数据、同一串随机种子,
只换精度方案,把"混合精度 ≈ fp32 收敛 + bf16 速度、纯 bf16 训坏"测出来。

三种方案:
  fp32    参数/计算全 fp32(基线)
  mixed   fp32 参数(即主副本)+ autocast bf16 计算 —— 事实上的 bf16 混合精度
  bf16    参数/动量/计算全 bf16,无 fp32 主副本 —— "省掉主副本"的天真方案

另测两笔账(都在 mixed 运行里,fp32 参数是可信的对照面):
  1. swallow fraction:每 200 步,把这一步的 fp32 更新量 Δ 真的按 bf16 舍入一遍,
     数一数有多少参数 (w+Δ).bf16 == w.bf16 —— 若参数是 bf16,这些更新会被整个吃掉;
  2. 梯度直方图:第 500 步所有参数梯度的 log10|g| 分布,对照 fp16/bf16 的可表示下界。

模型:char-level GPT(6 层 384 宽,~10M 参数,自包含定义,SDPA 注意力——
三种 dtype 都能跑;仓库 model.py 硬依赖 flash-attn,只收 fp16/bf16),tiny-shakespeare,单卡。
用法:python train_precision.py --outdir ../results
"""

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ln2 = nn.LayerNorm(n_embd, bias=False)
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + self.fc_proj(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, block, n_layer=6, n_head=6, n_embd=384):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight  # 权重绑定
        self.apply(lambda m: torch.nn.init.normal_(m.weight, 0.0, 0.02)
                   if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, idx, targets):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

BLOCK, BATCH = 256, 64
MAX_ITERS, WARMUP_ITERS = 6000, 100
LR, MIN_LR = 1e-3, 1e-4
EVAL_INTERVAL, EVAL_ITERS = 100, 20
SWALLOW_INTERVAL = 200
GRADHIST_STEP = 500
DEVICE = torch.device("cuda", 0)


def load_data():
    text = open(os.path.join(os.path.dirname(__file__), "input.txt")).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data, gen):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)


def lr_at(it):
    if it < WARMUP_ITERS:
        return LR * (it + 1) / WARMUP_ITERS
    t = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, val, autocast_ctx):
    model.eval()
    gen = torch.Generator().manual_seed(4242)  # 三个 run 用同一批验证 batch
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(val, gen)
        with autocast_ctx():
            _, loss = model(x, y)
        losses.append(loss.float().item())
    model.train()
    return sum(losses) / len(losses)


def run(scheme, train, val, vocab, outdir, tag=""):
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = GPT(vocab, BLOCK).to(DEVICE)
    if scheme == "bf16":
        model = model.bfloat16()  # 参数就是 bf16;Adam 动量随参数 dtype,也是 bf16

    if scheme == "mixed":
        def autocast_ctx():
            return torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    gen = torch.Generator().manual_seed(1337)  # 三个 run 吃同一串训练 batch

    curve, swallow_rows, hist_rows = [], [], []
    step_times = []
    train_ema = None  # train loss 的指数滑动平均(0.98),过拟合期比 val 更能看出停滞
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    for it in range(MAX_ITERS):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        x, y = get_batch(train, gen)

        ev0.record()
        with autocast_ctx():
            _, loss = model(x, y)
        loss.backward()

        # ---- swallow 对账(mixed:参数是 fp32,模拟"如果参数是 bf16"会吃掉多少更新)----
        do_swallow = scheme == "mixed" and (it + 1) % SWALLOW_INTERVAL == 0
        if do_swallow:
            prev = [p.detach().clone() for p in model.parameters()]
        # ---- 梯度直方图(mixed,第 500 步)----
        if scheme == "mixed" and it + 1 == GRADHIST_STEP:
            g_all = torch.cat([p.grad.detach().float().abs().flatten()
                               for p in model.parameters() if p.grad is not None])
            g_all = g_all[g_all > 0]
            logg = torch.log10(g_all)
            cnt = torch.histc(logg, bins=56, min=-12, max=2)
            edges = torch.linspace(-12, 2, 57)
            for lo, c in zip(edges[:-1].tolist(), cnt.tolist()):
                hist_rows.append([round(lo, 2), int(c)])

        opt.step()
        opt.zero_grad(set_to_none=True)
        ev1.record()
        torch.cuda.synchronize()
        if it >= 20 and not do_swallow:
            step_times.append(ev0.elapsed_time(ev1))
        li = loss.float().item()
        train_ema = li if train_ema is None else 0.98 * train_ema + 0.02 * li

        if do_swallow:
            tot, sw = 0, 0
            with torch.no_grad():
                for p, w0 in zip(model.parameters(), prev):
                    delta = p.detach() - w0
                    w0b = w0.bfloat16()
                    swallowed = (w0b.float() + delta).bfloat16() == w0b
                    sw += swallowed.sum().item()
                    tot += p.numel()
            swallow_rows.append([it + 1, round(sw / tot, 4)])
            print(f"  [swallow] step {it+1}: {sw/tot:.1%}", flush=True)

        if (it + 1) % EVAL_INTERVAL == 0 or it == 0:
            vl = evaluate(model, val, autocast_ctx)
            curve.append([it + 1, round(vl, 4), round(train_ema, 4)])
            print(f"[{scheme}] step {it+1}: val {vl:.4f} train_ema {train_ema:.4f}", flush=True)

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    step_ms = sum(step_times) / len(step_times)
    final_val = curve[-1][1]

    with open(os.path.join(outdir, f"curve_{scheme}{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "val_loss", "train_ema"]); w.writerows(curve)
    if swallow_rows:
        with open(os.path.join(outdir, f"swallow{tag}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step", "swallow_frac"]); w.writerows(swallow_rows)
    if hist_rows:
        with open(os.path.join(outdir, "gradhist.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["log10_bin_lo", "count"]); w.writerows(hist_rows)
    return [scheme, round(step_ms, 2), round(peak_gib, 3), final_val]


def main():
    global MAX_ITERS, WARMUP_ITERS, LR, MIN_LR, GRADHIST_STEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../results")
    ap.add_argument("--steps", type=int, default=MAX_ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--min-lr", type=float, default=MIN_LR)
    ap.add_argument("--tag", default="")  # 小 LR 机制实验用 --tag _lowlr
    args = ap.parse_args()
    MAX_ITERS, WARMUP_ITERS, LR, MIN_LR = args.steps, args.warmup, args.lr, args.min_lr
    if args.tag:
        GRADHIST_STEP = -1  # 直方图只在主实验记
    os.makedirs(args.outdir, exist_ok=True)

    train, val, vocab = load_data()
    print(f"vocab {vocab}, train {len(train)/1e6:.2f}M chars", flush=True)

    summary = []
    for scheme in ["fp32", "mixed", "bf16"]:
        print(f"=== {scheme} ===", flush=True)
        summary.append(run(scheme, train, val, vocab, args.outdir, args.tag))
    with open(os.path.join(args.outdir, f"summary{args.tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "step_ms", "peak_mem_gib", "final_val_loss"])
        w.writerows(summary)
    for row in summary:
        print("SUMMARY:", row)


if __name__ == "__main__":
    main()
