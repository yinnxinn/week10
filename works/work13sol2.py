import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
import os

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def main():
    # 1. Load Data
    print("Loading dataset...")
    dataset = load_dataset("left0ver/sentiment-classification")
    
    # 2. Tokenization
    model_name = "bert-base-chinese"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 3. Model Initialization
    print(f"Loading model: {model_name}")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"] # Common for BERT
    )

    model = get_peft_model(model, peft_config)
    
    # Verify trainable parameters
    model.print_trainable_parameters()

    # 5. Training Setup
    output_dir = os.path.join(os.path.dirname(__file__), "results_lora")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,  # Standard LoRA learning rate
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("Starting LoRA training...")
    trainer.train()

    # 7. Final Evaluation
    print("Evaluating on validation set...")
    eval_result = trainer.evaluate()
    print(f"Final Test Accuracy: {eval_result['eval_accuracy']:.4f}")

    if eval_result['eval_accuracy'] >= 0.85:
        print("Success: Accuracy is >= 85%")
    else:
        print("Warning: Accuracy is < 85%, you might need more epochs or hyperparameter tuning.")

if __name__ == "__main__":
    main()

'''
trainable params: 296,450 || all params: 102,565,636 || trainable%: 0.2890
Starting LoRA training...
{'loss': 0.5577, 'grad_norm': 2.2033684253692627, 'learning_rate': 0.00019314685314685316, 'epoch': 0.17}
{'loss': 0.2939, 'grad_norm': 2.728065013885498, 'learning_rate': 0.00018615384615384617, 'epoch': 0.35}                       
{'loss': 0.2517, 'grad_norm': 3.5144190788269043, 'learning_rate': 0.00017916083916083916, 'epoch': 0.52}                      
{'loss': 0.2466, 'grad_norm': 5.491372585296631, 'learning_rate': 0.00017216783216783215, 'epoch': 0.7}                        
{'loss': 0.2055, 'grad_norm': 4.289031982421875, 'learning_rate': 0.0001651748251748252, 'epoch': 0.87}                        
{'eval_loss': 0.22315643727779388, 'eval_accuracy': 0.9133333333333333, 'eval_runtime': 16.4969, 'eval_samples_per_second': 72.741, 'eval_steps_per_second': 2.303, 'epoch': 1.0}
{'loss': 0.2255, 'grad_norm': 1.4659206867218018, 'learning_rate': 0.0001581818181818182, 'epoch': 1.05}                       
{'loss': 0.1923, 'grad_norm': 1.9639636278152466, 'learning_rate': 0.00015118881118881118, 'epoch': 1.22}
{'loss': 0.2154, 'grad_norm': 1.642274260520935, 'learning_rate': 0.0001441958041958042, 'epoch': 1.4}                         
{'loss': 0.1781, 'grad_norm': 4.55064058303833, 'learning_rate': 0.00013720279720279722, 'epoch': 1.57}                        
{'loss': 0.2112, 'grad_norm': 2.0464210510253906, 'learning_rate': 0.0001302097902097902, 'epoch': 1.75}                       
{'loss': 0.2047, 'grad_norm': 2.4403462409973145, 'learning_rate': 0.00012321678321678323, 'epoch': 1.92}                      
{'eval_loss': 0.20769211649894714, 'eval_accuracy': 0.9275, 'eval_runtime': 16.331, 'eval_samples_per_second': 73.48, 'eval_steps_per_second': 2.327, 'epoch': 2.0}
{'loss': 0.1848, 'grad_norm': 4.099738597869873, 'learning_rate': 0.00011622377622377623, 'epoch': 2.1}                        
{'loss': 0.1673, 'grad_norm': 1.0577630996704102, 'learning_rate': 0.00010923076923076922, 'epoch': 2.27}
{'loss': 0.165, 'grad_norm': 6.034125328063965, 'learning_rate': 0.00010223776223776225, 'epoch': 2.45}                        
{'loss': 0.1832, 'grad_norm': 2.8656773567199707, 'learning_rate': 9.524475524475524e-05, 'epoch': 2.62}                       
{'loss': 0.1487, 'grad_norm': 4.369003772735596, 'learning_rate': 8.825174825174826e-05, 'epoch': 2.8}                         
{'loss': 0.177, 'grad_norm': 0.731778621673584, 'learning_rate': 8.125874125874126e-05, 'epoch': 2.97}                         
{'eval_loss': 0.2050233781337738, 'eval_accuracy': 0.9258333333333333, 'eval_runtime': 17.2241, 'eval_samples_per_second': 69.67, 'eval_steps_per_second': 2.206, 'epoch': 3.0}
{'loss': 0.1213, 'grad_norm': 1.2144526243209839, 'learning_rate': 7.426573426573427e-05, 'epoch': 3.15}                       
{'loss': 0.1432, 'grad_norm': 2.790152072906494, 'learning_rate': 6.727272727272727e-05, 'epoch': 3.32}
{'loss': 0.1836, 'grad_norm': 5.142415523529053, 'learning_rate': 6.0279720279720284e-05, 'epoch': 3.5}                        
{'loss': 0.15, 'grad_norm': 3.756633758544922, 'learning_rate': 5.328671328671329e-05, 'epoch': 3.67}                          
{'loss': 0.142, 'grad_norm': 3.976266622543335, 'learning_rate': 4.629370629370629e-05, 'epoch': 3.85}                         
{'eval_loss': 0.19493532180786133, 'eval_accuracy': 0.9341666666666667, 'eval_runtime': 16.3271, 'eval_samples_per_second': 73.498, 'eval_steps_per_second': 2.327, 'epoch': 4.0}
{'loss': 0.1328, 'grad_norm': 3.168384552001953, 'learning_rate': 3.9300699300699304e-05, 'epoch': 4.02}                       
{'loss': 0.1358, 'grad_norm': 4.3254313468933105, 'learning_rate': 3.230769230769231e-05, 'epoch': 4.2}
{'loss': 0.1464, 'grad_norm': 1.6145237684249878, 'learning_rate': 2.5314685314685316e-05, 'epoch': 4.37}                      
{'loss': 0.1298, 'grad_norm': 2.589928150177002, 'learning_rate': 1.8321678321678323e-05, 'epoch': 4.55}                       
{'loss': 0.1265, 'grad_norm': 1.382814884185791, 'learning_rate': 1.132867132867133e-05, 'epoch': 4.72}                        
{'loss': 0.1339, 'grad_norm': 2.881481647491455, 'learning_rate': 4.335664335664335e-06, 'epoch': 4.9}                         
{'eval_loss': 0.19515588879585266, 'eval_accuracy': 0.9358333333333333, 'eval_runtime': 16.1743, 'eval_samples_per_second': 74.192, 'eval_steps_per_second': 2.349, 'epoch': 5.0}
{'train_runtime': 1484.9557, 'train_samples_per_second': 30.796, 'train_steps_per_second': 0.963, 'train_loss': 0.1901329454008516, 'epoch': 5.0}
100%|█████████████████████████████████████████████████████████████████████████████████████| 1430/1430 [24:44<00:00,  1.04s/it] 
Evaluating on validation set...
100%|█████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:15<00:00,  2.41it/s]
Final Test Accuracy: 0.9358
'''