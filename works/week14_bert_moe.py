import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, Trainer, TrainingArguments, BertTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Define the MoE Layer
class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Gating Network: Decides which experts to use
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Experts: A list of Feed-Forward Networks
        # Using simple Linear layers as experts here. 
        # In more complex setups, these could be MLPs (Linear -> Gelu -> Linear).
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        
        # 1. Compute Gating Scores
        gate_logits = self.gate(x) # [batch_size, num_experts]
        
        # 2. Select Top-K Experts
        # weights: [batch_size, k], indices: [batch_size, k]
        weights, indices = torch.topk(gate_logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1) # Normalize weights for the selected experts
        
        # 3. Compute Expert Outputs
        # Ideally, we only compute forward pass for selected experts to save compute.
        # But for implementation simplicity (and because batch items select different experts),
        # we often compute all and mask, OR use advanced scatter/gather operations.
        # Given num_experts is small (e.g., 4), computing all is acceptable for this demo.
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) 
        # expert_outputs shape: [batch_size, num_experts, output_dim]
        
        # 4. Aggregate Results
        # We need to gather the outputs corresponding to the top-k indices.
        
        # Expand indices to [batch_size, k, output_dim] to gather along expert dimension
        # But expert_outputs is [batch, experts, out]. 
        # It's easier to create a sparse weight matrix.
        
        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for i in range(self.k):
            # For each of the k selected experts
            expert_idx = indices[:, i] # [batch_size]
            w = weights[:, i].unsqueeze(1) # [batch_size, 1]
            
            # Extract the specific expert output for each batch item
            # This is the tricky part without a loop or advanced indexing.
            # Let's iterate over the batch for clarity if batch size is small, or use gather.
            
            # Vectorized gather:
            # expert_outputs: [B, E, D]
            # expert_idx: [B] -> we want [B, 1, D]
            selected_expert_out = expert_outputs[torch.arange(batch_size, device=x.device), expert_idx] # [B, D]
            
            final_output += w * selected_expert_out
            
        return final_output

# 2. Define the BERT-MoE Model
class BertMoEForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_experts=4, k=2):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Load BERT Backbone
        self.bert = BertModel(config)
        
        # MoE Layer
        # Replacing the standard Classifier pooling/dense logic with MoE
        self.moe = MoELayer(config.hidden_size, config.hidden_size, num_experts=num_experts, k=k)
        
        # Final Classification Head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Remove arguments that BertModel doesn't accept but Trainer might pass
        kwargs.pop('num_items_in_batch', None)
        
        # 1. BERT Forward
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        # Use the pooled output (CLS token representation)
        pooled_output = outputs.pooler_output # [batch_size, hidden_size]
        
        # 2. MoE Forward
        moe_output = self.moe(pooled_output) # [batch_size, hidden_size]
        
        # 3. Classification
        logits = self.classifier(moe_output) # [batch_size, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# 3. Training Script
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def main():
    print("Loading tokenizer and dataset...")
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Load Dataset
    dataset = load_dataset("left0ver/sentiment-classification")
    
    # Preprocess
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Initialize Model
    print("Initializing BERT-MoE Model...")
    from transformers import BertConfig
    config = BertConfig.from_pretrained(model_name, num_labels=2)
    model = BertMoEForSequenceClassification.from_pretrained(model_name, config=config, num_experts=8, k=2)
    
    print(f"Model Structure: {model}")
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results_bert_moe",
        eval_strategy="epoch", # Updated from evaluation_strategy
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], # Assuming 'validation' split exists, or use 'test'
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting Training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating...")
    results = trainer.evaluate()
    print(results)
    
    # Save
    model.save_pretrained("./bert_moe_final")
    tokenizer.save_pretrained("./bert_moe_final")
    print("Model saved to ./bert_moe_final")

if __name__ == "__main__":
    main()

'''
{'loss': 0.0549, 'grad_norm': 0.2022971212863922, 'learning_rate': 2.4125874125874128e-06, 'epoch': 2.64}                                                                                      
{'loss': 0.0108, 'grad_norm': 0.3538878560066223, 'learning_rate': 2.296037296037296e-06, 'epoch': 2.66}                                                                                       
{'loss': 0.0705, 'grad_norm': 6.1891584396362305, 'learning_rate': 2.1794871794871797e-06, 'epoch': 2.67}                                                                                      
{'loss': 0.0978, 'grad_norm': 6.14323091506958, 'learning_rate': 2.0629370629370634e-06, 'epoch': 2.69}                                                                                        
{'loss': 0.0468, 'grad_norm': 15.990964889526367, 'learning_rate': 1.9463869463869462e-06, 'epoch': 2.71}                                                                                      
{'loss': 0.0869, 'grad_norm': 0.4167075753211975, 'learning_rate': 1.82983682983683e-06, 'epoch': 2.73}                                                                                        
{'loss': 0.0392, 'grad_norm': 0.07291598618030548, 'learning_rate': 1.7132867132867134e-06, 'epoch': 2.74}                                                                                     
{'loss': 0.0904, 'grad_norm': 0.48081865906715393, 'learning_rate': 1.5967365967365969e-06, 'epoch': 2.76}                                                                                     
{'loss': 0.0679, 'grad_norm': 0.0827532634139061, 'learning_rate': 1.4801864801864803e-06, 'epoch': 2.78}                                                                                      
{'loss': 0.0949, 'grad_norm': 15.873658180236816, 'learning_rate': 1.3636363636363636e-06, 'epoch': 2.8}                                                                                       
{'loss': 0.131, 'grad_norm': 0.32274043560028076, 'learning_rate': 1.247086247086247e-06, 'epoch': 2.81}                                                                                       
{'loss': 0.0924, 'grad_norm': 15.361115455627441, 'learning_rate': 1.1305361305361306e-06, 'epoch': 2.83}                                                                                      
{'loss': 0.1181, 'grad_norm': 5.580522060394287, 'learning_rate': 1.013986013986014e-06, 'epoch': 2.85}                                                                                        
{'loss': 0.043, 'grad_norm': 0.061106227338314056, 'learning_rate': 8.974358974358975e-07, 'epoch': 2.87}                                                                                      
{'loss': 0.04, 'grad_norm': 0.07089325785636902, 'learning_rate': 7.80885780885781e-07, 'epoch': 2.88}                                                                                         
{'loss': 0.0181, 'grad_norm': 0.08495590090751648, 'learning_rate': 6.643356643356644e-07, 'epoch': 2.9}                                                                                       
{'loss': 0.0804, 'grad_norm': 3.4806740283966064, 'learning_rate': 5.477855477855478e-07, 'epoch': 2.92}                                                                                       
{'loss': 0.1296, 'grad_norm': 37.59585189819336, 'learning_rate': 4.3123543123543126e-07, 'epoch': 2.94}                                                                                       
{'loss': 0.0467, 'grad_norm': 0.13264252245426178, 'learning_rate': 3.1468531468531473e-07, 'epoch': 2.95}                                                                                     
{'loss': 0.0431, 'grad_norm': 0.07681816071271896, 'learning_rate': 1.9813519813519813e-07, 'epoch': 2.97}                                                                                     
{'loss': 0.0495, 'grad_norm': 58.77288055419922, 'learning_rate': 8.158508158508159e-08, 'epoch': 2.99}                                                                                        
{'eval_loss': 0.28077107667922974, 'eval_accuracy': 0.9391666666666667, 'eval_runtime': 11.3542, 'eval_samples_per_second': 105.687, 'eval_steps_per_second': 6.605, 'epoch': 3.0}             
{'train_runtime': 901.0682, 'train_samples_per_second': 30.451, 'train_steps_per_second': 1.904, 'train_loss': 0.17962003033391108, 'epoch': 3.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1716/1716 [15:01<00:00,  1.90it/s]
Evaluating...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 75/75 [0
'''