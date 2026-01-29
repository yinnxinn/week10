import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        
        # 1. 模拟预训练权重 (W)，在实际微调中它是冻结的
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False  # 冻结原始参数
        
        # 2. LoRA 的核心：两个低秩矩阵 A 和 B
        # A 矩阵将维度从 in_features 降到 rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        # B 矩阵将维度从 rank 升到 out_features
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank)) # 初始化为0以保证开始时不影响模型
        
        # 缩放系数 (Scaling factor)
        self.scaling = alpha / rank
        
        print(f"原始矩阵参数量: {in_features * out_features}")
        print(f"LoRA 矩阵参数量 (A+B): {rank * in_features + out_features * rank}")
        print(f"参数压缩比: {(rank * in_features + out_features * rank) / (in_features * out_features):.2%}")

    def forward(self, x):
        # 原始路径计算：h = Wx
        original_output = F.linear(x, self.weight)
        
        # LoRA 路径计算：Δh = (B * A)x = B(Ax)
        # 注意：先计算 Ax 再乘以 B，计算复杂度远低于直接计算 (BA)x
        lora_output = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        
        # 最终输出 = Wx + ΔWx
        return original_output + lora_output

# --- 演示运行 ---
# 假设输入维度 1024，输出维度 1024，秩(Rank)设为 8
in_dim, out_dim, r = 1024, 1024, 8

# 初始化层
lora_layer = LoRALayer(in_dim, out_dim, rank=r)

# 模拟输入数据
x = torch.randn(1, in_dim)

# 前向传播
output = lora_layer(x)

print(f"\n输入形状: {x.shape}")
print(f"输出形状: {output.shape}")