import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口



class MHA(nn.Module):  #
    def __init__(self, d_model, num_heads):  
        super().__init__()  
        self.num_heads = num_heads  # 保存头数
        self.d_head = d_model // num_heads  # 计算每个头的维度
        self.qkv = nn.Linear(d_model, d_model * 3)  # 定义 Q, K, V 的线性投影层
        ## self.q / self.k / self.v = nn.Linear(d_model, d_model )
        self.output = nn.Linear(d_model, d_model)  # 定义输出线性层

    def forward(self, x):  # 前向传播方法
        B, L, D = x.shape  # 获取输入的 Batch 大小、序列长度和维度
        # 生成 Q, K, V -> 形状: [B, L, 3, H, D_head]
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.d_head)  # 投影并重塑形状以分离头
        q, k, v = qkv.unbind(2)  # 在维度 2 上解绑，分别得到 Q, K, V

        # 线性注意力计算
        attn = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) / (self.d_head ** 0.5)  # 计算注意力分数 (Scaled Dot-Product)
        attn = F.softmax(attn, dim=-1)  # 在最后一个维度上应用 Softmax 进行归一化

        out = (attn @ v.transpose(1, 2)).transpose(1, 2).contiguous().view(B, L, D)  # 计算加权和并恢复原始形状
        return self.output(out)  # 通过输出层并返回结果

# 测试代码
model = MHA(d_model=512, num_heads=8)  # 实例化 MHA 模型，指定维度和头数
x = torch.randn(1, 10, 512)  # 生成一个随机输入张量 [Batch=1, SeqLen=10, Dim=512]
output = model(x)  # 执行前向传播
print(f"输出形状: {output.shape}") # [1, 10, 512]  # 打印输出张量的形状
