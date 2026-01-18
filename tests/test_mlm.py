import torch
import torch.nn as nn
import random
from transformers import BertConfig, BertModel
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 1. NSP 数据处理 (确保 random 已导入)
def create_nsp_data(paragraphs, max_seq_len):
    nsp_samples = []
    for paragraph in paragraphs:
        for i in range(len(paragraph) - 1):
            if random.random() > 0.5:
                is_next = True
                sentence_a = paragraph[i]
                sentence_b = paragraph[i + 1]
            else:
                is_next = False
                sentence_a = paragraph[i]
                random_para = random.choice(paragraphs)
                sentence_b = random.choice(random_para)
            
            # 这里简化处理，实际需要 tokenizer
            tokens = ["[CLS]"] + list(sentence_a) + ["[SEP]"] + list(sentence_b) + ["[SEP]"]
            sep_index = tokens.index("[SEP]")
            segment_ids = [0] * (sep_index + 1) + [1] * (len(tokens) - sep_index - 1)
            
            tokens = tokens[:max_seq_len]
            segment_ids = segment_ids[:max_seq_len]
            label = 1 if is_next else 0
            
            nsp_samples.append({
                "tokens": tokens,
                "segment_ids": segment_ids,
                "label": label
            })
    return nsp_samples

# 2. MLM 掩码逻辑
def mask_tokens(inputs, mask_token_id, vocab_size):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, 0.15)
    
    # 假设 ID < 10 的是特殊字符 [CLS], [SEP] 等
    special_tokens_mask = (inputs < 10) 
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  

    # 80% [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10% Random
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels

# 3. 修正后的 BertForMLM 类
class BertForMLM(nn.Module):
    def __init__(self, bert_model, vocab_size, hidden_size):
        super().__init__()
        self.bert = bert_model 
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size) 
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # --- 关键修改点：使用关键字参数调用，防止顺序错误 ---
        outputs = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        
        # outputs 是一个对象，last_hidden_state 的形状是 [batch, seq, hidden]
        sequence_output = outputs.last_hidden_state
        
        # 经过 MLM Head
        prediction_scores = self.mlm_head(sequence_output)
        return prediction_scores

# --- 4. 模拟训练步骤 ---
vocab_size = 30522
hidden_size = 768

config = BertConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    num_hidden_layers=4, # 演示用，减小层数加快速度
    num_attention_heads=8,
    intermediate_size=1024
)

# 初始化模型
bert_base_model = BertModel(config)
model = BertForMLM(bert_base_model, vocab_size=vocab_size, hidden_size=hidden_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 模拟数据
raw_input_ids = torch.randint(10, vocab_size, (4, 20)) 
token_type_ids = torch.zeros_like(raw_input_ids)

model.train()

# A. 掩码操作
mask_token_id = 105 
masked_input_ids, labels = mask_tokens(raw_input_ids, mask_token_id, vocab_size)

# B. 正向传播
# 显式传入参数名，更加安全
logits = model(input_ids=masked_input_ids, token_type_ids=token_type_ids)

# C. 计算 Loss
loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

# D. 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"MLM Training Success! Loss: {loss.item()}")