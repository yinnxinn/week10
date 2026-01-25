import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# 1. 超参数设置 (Hyperparameters)
# ==========================================
batch_size = 32        # 每次训练样本数
block_size = 64        # 上下文长度（唐诗通常较短，64足够）
max_iters = 50       # 训练迭代次数
eval_interval = 5    # 每隔多少次打印一次效果
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 128           # 向量维度
n_head = 4             # 多头注意力的头数
n_layer = 4            # Transformer 层数
dropout = 0.1

torch.manual_seed(1337)

# ==========================================
# 2. 准备唐诗数据 (Data Preparation)
# ==========================================
# 实际使用时建议读取一个包含几万首唐诗的 .txt 文件
# 这里为了演示，我们提供几首著名的诗作为训练集
raw_text = """
床前明月光，疑是地上霜。举头望明月，低头思故乡。
春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
功盖三分国，名成八阵图。江流石不转，遗恨失吞吴。
红豆生南国，春来发几枝。愿君多采撷，此物最相思。
城阙辅三秦，风烟望五津。与君离别意，同是宦游人。
海内存知己，天涯若比邻。无为在歧路，儿女共沾巾。
""" * 100 # 简单倍增数据模拟大规模训练

# 字符级词典
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 转化为 Tensor
data = torch.tensor(encode(raw_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 获取 Batch 数据
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ==========================================
# 3. GPT 模型组件 (Model Components)
# ==========================================

class Head(nn.Module):
    """ 单个注意力头 """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # 计算注意力得分 (Affinity)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 因果掩码
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # 加权求和
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ 多头注意力 """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ 简单的线性层 + 激活函数 """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer 块 """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # 残差连接
        x = x + self.ffwd(self.ln2(x))  # 残差连接
        return x

class PoetryGPT(nn.Module):
    """ 完整的 GPT 模型 """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # 裁剪上下文
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            ### 生成的时候softmax结果以后可以有不同的decode方式： greedy， sample
            ## todo topp ,topk , temperature, beam search
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 4. 训练与测试 (Training & Inference)
# ==========================================

model = PoetryGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("开始训练...")
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"迭代次数 {iter}: 损失值 {loss.item():.4f}")

# 生成测试
print("\n--- 训练完成，开始作诗 ---")
# 给模型一个开头词 "春"
start_context = torch.tensor([encode("春")], dtype=torch.long, device=device)
output = model.generate(start_context, max_new_tokens=40)
print(decode(output[0].tolist()))