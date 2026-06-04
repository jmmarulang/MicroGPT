import os, random, json, math, time
import torch
import torch.nn.functional as F

random.seed(42); torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type == 'cuda' else ''))

if not os.path.exists('input.txt'):
    import warnings, pandas as pd
    warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
    df = pd.read_parquet("hf://datasets/karpathy/tinystories-gpt4-clean/tinystories_gpt4_clean.parquet")
    with open('input.txt', 'w') as f:
        for s in df['text'].iloc[20000:25000]:
            f.write(json.dumps(s) + '\n')

docs = [json.loads(l) for l in open('input.txt') if l.strip()]
random.shuffle(docs)

uchars = sorted('\n !"$\',-.' + '0123456789:;?' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'abcdefghijklmnopqrstuvwxyz')
BOS = len(uchars); vocab_size = BOS + 1

stoi = {c: i for i, c in enumerate(uchars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(uchars[i] for i in ids)
print(f"docs: {len(docs)}, vocab: {vocab_size}, sample: {docs[0][:80]}...")

# ── Hyperparameters ──────────────────────────────────────────────────────────
n_layer    = 6
n_embd     = 256
block_size = 256
n_head     = 8
head_dim   = n_embd // n_head
batch_size = 64

W = lambda r, c: (torch.randn(r, c, device=device) * 0.02).requires_grad_(True)
sd = {'wte': W(vocab_size, n_embd)}
for i in range(n_layer):
    sd |= {f'l{i}.wq': W(n_embd, n_embd), f'l{i}.wk': W(n_embd, n_embd),
           f'l{i}.wv': W(n_embd, n_embd), f'l{i}.wo': W(n_embd, n_embd),
           f'l{i}.fc1': W(4*n_embd, n_embd), f'l{i}.fc2': W(n_embd, 4*n_embd)}
params = list(sd.values())
print(f"params: {sum(p.numel() for p in params):,}")

def rmsnorm(x):
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()

t = torch.arange(block_size, device=device).float()
f = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
rope_cos, rope_sin = torch.outer(t, f).cos(), torch.outer(t, f).sin()

def apply_rope(x, cos, sin):
    d = x.dtype
    x = x.float().unflatten(-1, (-1, 2))
    x0, x1 = x[..., 0], x[..., 1]
    cos = cos.view(1, -1, 1, cos.shape[-1])
    sin = sin.view(1, -1, 1, sin.shape[-1])
    return torch.stack([x0*cos - x1*sin, x0*sin + x1*cos], -1).flatten(-2).to(d)

def forward(tokens):
    B, T = tokens.shape
    x = rmsnorm(F.embedding(tokens, sd['wte']))
    cos, sin = rope_cos[:T], rope_sin[:T]
    for i in range(n_layer):
        r = x; x = rmsnorm(x)
        q = F.linear(x, sd[f'l{i}.wq']).view(B, T, n_head, head_dim)
        k = F.linear(x, sd[f'l{i}.wk']).view(B, T, n_head, head_dim)
        v = F.linear(x, sd[f'l{i}.wv']).view(B, T, n_head, head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True
        ).transpose(1,2).reshape(B, T, -1)
        x = F.linear(x, sd[f'l{i}.wo']) + r
        r = x; x = rmsnorm(x)
        x = F.silu(F.linear(x, sd[f'l{i}.fc1']))
        x = F.linear(x, sd[f'l{i}.fc2']) + r
    return F.linear(rmsnorm(x), sd['wte'])

forward = torch.compile(forward)
print("model ready")

all_tokens = torch.tensor(
    [t for doc in docs for t in [BOS] + encode(doc)] + [BOS],
    dtype=torch.long, device=device
)

def get_batch():
    s = torch.randint(0, len(all_tokens) - block_size - 1, (batch_size,), device=device)
    idx = s.unsqueeze(1) + torch.arange(block_size + 1, device=device)
    t = all_tokens[idx]
    return t[:, :-1], t[:, 1:]

# ── Optimizer: AdamW ─────────────────────────────────────────────────────────
num_steps = 3500
warmup    = 200
lr        = 1e-3
min_lr    = 1e-4

def get_lr(step):
    if step < warmup:
        return lr * step / warmup
    p = (step - warmup) / (num_steps - warmup)
    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))

opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-10, fused=(device.type == 'cuda'))
scaler = torch.amp.GradScaler('cuda')
start = time.time()

for step in range(num_steps + 1):
    opt.param_groups[0]['lr'] = get_lr(step)
    if step % 200 == 0:
        xb, yb = get_batch()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            logits = forward(xb)
            loss_val = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1)).item()
            preds = logits[:, -1, :vocab_size].argmax(1)
            acc = (preds == yb[:, -1]).float().mean().item()
        print(f"Step {step:4d} | Loss: {loss_val:.4f} | Acc: {acc:.1%} | {time.time()-start:.1f}s")
    if step >= num_steps: break
    opt.zero_grad(set_to_none=True)
    xb, yb = get_batch()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        loss = F.cross_entropy(forward(xb).reshape(-1, vocab_size), yb.reshape(-1))
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update()

print(f"Training time: {time.time() - start:.1f}s")

# ── Generate ─────────────────────────────────────────────────────────────────
def generate(num_chars=200, temperature=0.7):
    tokens = [BOS]
    with torch.no_grad():
        for _ in range(num_chars):
            x = torch.tensor([tokens[-block_size:]], device=device)
            logits = forward(x)[0, -1, :vocab_size]
            next_id = torch.multinomial(F.softmax(logits / temperature, -1), 1).item()
            if next_id == BOS: break
            tokens.append(next_id)
    return decode(tokens[1:])

print(generate())
