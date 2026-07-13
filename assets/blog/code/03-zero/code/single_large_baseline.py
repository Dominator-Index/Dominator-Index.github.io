import sys, time, torch
sys.path.insert(0, "/jumbo/yaoqingyang/ouyangzhuoli/MARS/MARS")
from model import GPT, GPTConfig
torch.manual_seed(1337)
m = GPT(GPTConfig(n_layer=36, n_head=20, n_embd=1280, dropout=0.0)).cuda().bfloat16()
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True)
X = torch.randint(0, 50304, (4, 1024), device="cuda"); Y = torch.randint(0, 50304, (4, 1024), device="cuda")
def step():
    _, loss = m(X, Y); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
for _ in range(5): step()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(15): step()
torch.cuda.synchronize()
print(f"single-GPU GPT-Large mbs4 bf16: {(time.perf_counter()-t0)/15*1e3:.1f} ms/step")
