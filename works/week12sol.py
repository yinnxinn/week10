import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. 超参数设置 (Hyperparameters)
# ==========================================
batch_size = 32        # 每次训练样本数
block_size = 64        # 上下文长度（唐诗通常较短，64足够）
epochs = 5             # 训练轮数 (原 max_iters 改为 epochs 以确保遍历所有数据)
eval_interval = 50     # 每隔多少步打印一次效果
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
raw_text = ''
with open('D:/projects/classes/week10/week10/data/poetry.txt', 'r', encoding='utf-8') as r:
    for line in r.readlines():
        raw_text += '\n' + line.strip().split(':')[-1]


# 字符级词典
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 转化为 Tensor
data = torch.tensor(encode(raw_text), dtype=torch.long)
print(len(data))
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

# 创建 DataLoader 以遍历所有数据
def create_dataloader(data_tensor, batch_size, block_size):
    # 将数据切分为块，步长为 block_size (不重叠)
    chunks = []
    # 注意：我们需要取 block_size + 1 长度，作为 x(前block_size) 和 y(后block_size)
    for i in range(0, len(data_tensor) - block_size, block_size):
        chunk = data_tensor[i : i + block_size + 1]
        # 确保最后一个 chunk 长度足够
        if len(chunk) == block_size + 1:
            chunks.append(chunk)
            
    if not chunks:
        print("警告：数据不足以构建一个 batch")
        return None
        
    data_stack = torch.stack(chunks)
    x = data_stack[:, :-1]
    y = data_stack[:, 1:]
    
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_data, batch_size, block_size)

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

    def generate_acrostic(self, name, temperature=1.0):
        """
        name: 藏头的字符串，如 "林青霞"
        temperature: 控制随机性 (0.7-1.0 较好)
        """
        self.eval()
        # 初始 token：人名的第一个字
        idx = torch.tensor([[stoi[name[0]]]], dtype=torch.long, device=device)
        
        # 遍历人名的每一个字（除了第一个，因为已经作为起点）
        for i in range(len(name)):
            # 每一句生成的逻辑：直到生成句号 '。'
            # 或者达到一个安全长度（如 12 个字）
            count = 0
            while True:
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # 采样
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                char_next = itos[idx_next.item()]
                idx = torch.cat((idx, idx_next), dim=1)
                count += 1
                
                # 如果生成了句号，说明这一句结束了
                if char_next == '。' or count > 15:
                    break
            
            # 如果还没到人名的最后一个字，强制插入下一个藏头字
            if i < len(name) - 1:
                next_head_char = name[i+1]
                if next_head_char in stoi:
                    next_head_idx = torch.tensor([[stoi[next_head_char]]], dtype=torch.long, device=device)
                    idx = torch.cat((idx, next_head_idx), dim=1)
                else:
                    print(f"警告：字符 {next_head_char} 不在词典中")
        
        return decode(idx[0].tolist())

# ==========================================
# 4. 训练与测试 (Training & Inference)
# ==========================================

model = PoetryGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"开始训练... (Epochs: {epochs})")
global_step = 0
loss_history = []  # 用于记录loss变化

for epoch in range(epochs):
    if train_loader is None:
        print("Error: train_loader is None")
        break
        
    for step, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # 记录当前的loss值
        loss_history.append(loss.item())
        
        global_step += 1
        if global_step % eval_interval == 0:
            print(f"Epoch {epoch+1}, Step {step}, Global Step {global_step}: 损失值 {loss.item():.4f}")

print(f"训练结束，最终损失值: {loss.item():.4f}")

# ==========================================
# 5. 记录与可视化 (Logging & Visualization)
# ==========================================

# 保存Loss记录到文件
try:
    with open('loss_history.txt', 'w', encoding='utf-8') as f:
        for l in loss_history:
            f.write(f"{l}\n")
    print("Loss history saved to loss_history.txt")
except Exception as e:
    print(f"Error saving loss history: {e}")

# 尝试绘制Loss曲线
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved as loss_curve.png")
except ImportError:
    print("matplotlib module not found. Skipping plot generation.")
except Exception as e:
    print(f"Error generating plot: {e}")

# 生成测试
print("\n--- 训练完成，开始作诗 ---")
# 给模型一个开头词 "春"
start_context = torch.tensor([encode("春")], dtype=torch.long, device=device)
output = model.generate(start_context, max_new_tokens=40)
print(decode(output[0].tolist()))


# 藏头诗测试
print("\n" + "="*30)
print("【藏头诗生成测试】")
names = ["床白红", "海欲功"] # 使用数据集中存在的字进行测试
for n in names:
    result = model.generate_acrostic(n, temperature=0.8)
    print(f"人名: {n} -> 诗句: {result}")

'''
==============================
【藏头诗生成测试】
人名: 床白红 -> 诗句: 床边草围。白日寒江水，今朝梦寄空。红尘尽杖马，不似迎高儿。
人名: 海欲功 -> 诗句: 海云。欲去应不到，恩深太本无。功成无食事，心勤有业论。
'''