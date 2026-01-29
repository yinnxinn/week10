import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
# --- 新增导入 ---
from peft import LoraConfig, get_peft_model, TaskType

label2id = {
    "内科": 0,
    "外科": 1,
    "儿科": 2,
    "骨科": 3,
    "妇产科": 4,
}
id2label = {v: k for k, v in label2id.items()}

class TriageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

def build_dataloaders(tokenizer, batch_size=8):
    train_texts = [
        "反复咳嗽、咳痰一周，伴有低热",
        "右腿外伤后疼痛，活动受限",
        "两岁孩子发烧、咳嗽、流鼻涕",
        "摔倒后手腕肿胀疼痛",
        "怀孕三个月，下腹隐痛",
    ]
    train_labels = [
        label2id["内科"],
        label2id["外科"],
        label2id["儿科"],
        label2id["骨科"],
        label2id["妇产科"],
    ]

    dataset = TriageDataset(train_texts, train_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def predict_department(model, tokenizer, text, device):
    model.eval()
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        pred_id = outputs.logits.argmax(dim=-1).item()
    return id2label[pred_id]

# --- 新增：打印可训练参数量 ---
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"可训练参数量: {trainable_params} || 总参数量: {all_param} || 占比: {100 * trainable_params / all_param:.2f}%"
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "google-bert/bert-base-chinese"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    
    # 1. 加载基础模型
    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # 2. 配置 LoRA
    # r: 秩，通常设为 8, 16
    # target_modules: 哪些层应用 LoRA，BERT 通常是 query 和 value
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["query", "value"] 
    )

    # 3. 封装模型为 PEFT 模型
    model = get_peft_model(base_model, lora_config).to(device)
    
    # 验证可训练参数
    print_trainable_parameters(model)

    train_loader = build_dataloaders(tokenizer, batch_size=4)

    # 注意：此时 optimizer 只会优化 LoRA 层的参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # LoRA 通常可以使用稍大的学习率

    print("开始 LoRA 训练...")
    for epoch in range(5): # 增加一点 epoch 观察效果
        loss = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, train_loader, device)
        print(f"epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

    demo_text = "孩子发烧三天，伴有咳嗽流鼻涕，应挂哪个科室"
    pred_dept = predict_department(model, tokenizer, demo_text, device)
    print("预测科室:", pred_dept)
    
    # 4. 保存模型 (只会保存很小的 LoRA 权重文件)
    # model.save_pretrained("bert-lora-medical")