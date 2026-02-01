import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from pathlib import Path
# Set mirror for China BEFORE importing huggingface_hub
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def decoding_demonstration():
    # 1. 环境准备：加载预训练模型和分词器
    # 使用 GPT-2 作为演示模型
    base_dir = Path(__file__).resolve().parents[1]
    model_name = base_dir / "models" / "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    input_text = "The future of Artificial Intelligence is"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    print(f"输入文本: {input_text}\n" + "="*50)

    # --- 方法 1: Greedy Search (贪心搜索) ---
    # 原理：每一步都选择概率最高的词。
    # 优点：速度快。
    # 缺点：容易陷入循环，错过全局最优，生成文本单调。
    greedy_output = model.generate(
        input_ids, 
        max_length=50, 
        num_beams=1, 
        do_sample=False
    )
    print(f"1. Greedy Search:\n{tokenizer.decode(greedy_output[0], skip_special_tokens=True)}\n")


    # --- 方法 2: Beam Search (束搜索) ---
    # 原理：维护 num_beams 个候选序列，每一步保留总概率最高的路径。
    # 优点：比贪心搜索更易找到高质量序列。
    # 缺点：依然可能产生重复，且计算量增加。
    beam_output = model.generate(
        input_ids, 
        max_length=50, 
        num_beams=5, 
        early_stopping=True
    )
    print(f"2. Beam Search (5 beams):\n{tokenizer.decode(beam_output[0], skip_special_tokens=True)}\n")


    # --- 方法 3: Random Sampling (随机采样) ---
    # 原理：根据模型输出的概率分布随机抽取下一个词。
    # 优点：具有多样性和创造力。
    # 缺点：可能产生完全不通顺的句子。
    sample_output = model.generate(
        input_ids, 
        max_length=50, 
        do_sample=True, 
        top_k=0  # 关闭 top_k 以展示纯采样
    )
    print(f"3. Pure Random Sampling:\n{tokenizer.decode(sample_output[0], skip_special_tokens=True)}\n")


    # --- 方法 4: Temperature Scaling (温度缩放) ---
    # 原理：在 Softmax 前将 Logits 除以 T。
    #   - T < 1 (低): 分布变陡峭，模型更自信（趋向贪心）。
    #   - T > 1 (高): 分布变平缓，增加多样性（更随机）。
    temp_output = model.generate(
        input_ids, 
        max_length=50, 
        do_sample=True, 
        temperature=0.7, # 常见的平衡点
        top_k=0
    )
    print(f"4. Temperature Sampling (T=0.7):\n{tokenizer.decode(temp_output[0], skip_special_tokens=True)}\n")


    # --- 方法 5: Top-K Sampling ---
    # 原理：只从概率最高的 K 个词中进行采样。
    # 优点：将概率极低的“长尾”词过滤掉，减少语法错误。
    top_k_output = model.generate(
        input_ids, 
        max_length=50, 
        do_sample=True, 
        top_k=50 
    )
    print(f"5. Top-K Sampling (K=50):\n{tokenizer.decode(top_k_output[0], skip_special_tokens=True)}\n")


    # --- 方法 6: Top-P (Nucleus) Sampling (核采样) ---
    # 原理：从概率累加和达到 P 的最小候选集中采样。
    # 优点：候选集的大小是动态的。当模型很确定时，候选集小；不确定时，候选集变大。
    # 这是目前最常用的生成策略。
    top_p_output = model.generate(
        input_ids, 
        max_length=50, 
        do_sample=True, 
        top_p=0.92, 
        top_k=0
    )
    print(f"6. Top-P (Nucleus) Sampling (P=0.92):\n{tokenizer.decode(top_p_output[0], skip_special_tokens=True)}\n")


def manual_decode_logic_explanation():
    print("="*60)
    print("手动解码逻辑演示：从 Logits 到 Token")
    print("="*60)
    
    # 0. 初始数据：假设词表大小为 6
    # 索引 2 的分值最高 (3.5)，索引 5 的分值最低 (0.1)
    logits = torch.tensor([2.0, 1.5, 3.5, 0.5, 1.2, 0.1])
    print(f"原始 Logits (模型直接输出): \n{logits}\n")

    # --- 步骤 1: Temperature (温度缩放) ---
    # 逻辑：除以 T。T 越小，高分越高，低分越低。
    temp = 0.7
    temp_logits = logits / temp
    temp_probs = F.softmax(temp_logits, dim=-1)
    print(f"1. Temperature (T={temp}) 处理后概率:")
    print(f"   (你会发现大的值变得更突出): \n{temp_probs.round(decimals=4)}\n")


    # --- 步骤 2: Top-K 过滤 ---
    # 逻辑：只保留前 K 个最大的，其余统统抹杀（设为负无穷）。
    k = 3
    top_k_logits = logits.clone()
    # 找到第 K 个最大的值作为阈值
    values, indices = torch.topk(top_k_logits, k)
    min_value_in_top_k = values[-1] 
    
    # 将小于阈值的全部设为 -inf，这样经过 Softmax 后概率就是 0
    top_k_logits[top_k_logits < min_value_in_top_k] = -float('Inf')
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    print(f"2. Top-K (K={k}) 过滤后的概率:")
    print(f"   (只有前3个有概率，其余为0): \n{top_k_probs.round(decimals=4)}\n")


    # --- 步骤 3: Top-P (核采样) 核心实现 ---
    # 逻辑：将词按概率排序，取累计和达到 P 的最小集合。
    p = 0.9
    # 1. 先对原始 logits 进行排序 (从大到小)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # 2. 计算累计概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    print(f"3. Top-P (P={p}) 逻辑拆解:")
    print(f"   排序后概率: {sorted_probs.round(decimals=4)}")
    print(f"   累计概率流: {cumulative_probs.round(decimals=4)}")

    # 3. 找到哪些词该被移除 (累计概率 > p 的词)
    # 注意：我们要保留达到 P 的最后一个词，所以要做一个 shift
    sorted_indices_to_remove = cumulative_probs > p
    # 将掩码向右移：保留第一个超过 p 的索引
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 4. 将需要移除的索引映射回原始 Logits 并屏蔽
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    top_p_logits = logits.clone()
    top_p_logits[indices_to_remove] = -float('Inf')
    top_p_probs = F.softmax(top_p_logits, dim=-1)
    print(f"   Top-P 过滤后的最终概率: \n{top_p_probs.round(decimals=4)}\n")


    # --- 步骤 4: Multinomial Sampling (采样) ---
    # 逻辑：根据上面处理后的概率分布，像抽签一样抽一个词。
    # 概率越高，抽中的几率越大，但不是绝对的（区别于 argmax）。
    final_probs = top_p_probs # 假设我们使用 Top-P 的结果
    # 模拟抽样 10 次看看结果
    samples = [torch.multinomial(final_probs, num_samples=1).item() for _ in range(10)]
    print(f"4. 最终采样演示:")
    print(f"   根据 Top-P 概率抽样 10 次的结果: {samples}")
    print(f"   (你会发现索引 2 出现的次数最多，因为它的概率最高)")

if __name__ == "__main__":
    # 运行演示
    decoding_demonstration()
    #manual_decode_logic_explanation()