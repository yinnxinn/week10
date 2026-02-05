import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
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
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=236)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 3. Model Initialization
    print(f"Loading model: {model_name}")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. No Freezing (Full Fine-tuning)
    print("Full Parameter Fine-tuning (No freezing)...")
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")

    # 5. Training Setup
    output_dir = os.path.join(os.path.dirname(__file__), "results_full")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,  # Standard learning rate for full fine-tuning
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
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
    print("Starting full parameter training...")
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
Trainable parameters: 102269186 / 102269186 (100.00%)
Starting full parameter training...
{'loss': 0.4109, 'grad_norm': 3.125576972961426, 'learning_rate': 1.8286713286713288e-05, 'epoch': 0.17}
{'loss': 0.2547, 'grad_norm': 6.744264125823975, 'learning_rate': 1.653846153846154e-05, 'epoch': 0.35}
{'loss': 0.2304, 'grad_norm': 7.394164562225342, 'learning_rate': 1.479020979020979e-05, 'epoch': 0.52}
{'loss': 0.2201, 'grad_norm': 11.593501091003418, 'learning_rate': 1.3041958041958043e-05, 'epoch': 0.7}
{'loss': 0.1906, 'grad_norm': 7.804779529571533, 'learning_rate': 1.1293706293706294e-05, 'epoch': 0.87}
{'eval_loss': 0.20306727290153503, 'eval_accuracy': 0.9308333333333333, 'eval_runtime': 14.4635, 'eval_samples_per_second': 82.967, 'eval_steps_per_second': 2.627, 'epoch': 1.0}
{'loss': 0.1891, 'grad_norm': 2.641162633895874, 'learning_rate': 9.545454545454547e-06, 'epoch': 1.05}
{'loss': 0.1345, 'grad_norm': 5.580107688903809, 'learning_rate': 7.797202797202798e-06, 'epoch': 1.22}
{'loss': 0.1299, 'grad_norm': 11.51944351196289, 'learning_rate': 6.04895104895105e-06, 'epoch': 1.4}
{'loss': 0.1243, 'grad_norm': 2.6811673641204834, 'learning_rate': 4.300699300699301e-06, 'epoch': 1.57}
{'loss': 0.1298, 'grad_norm': 1.0339874029159546, 'learning_rate': 2.5524475524475528e-06, 'epoch': 1.75}
{'loss': 0.1312, 'grad_norm': 10.58576488494873, 'learning_rate': 8.041958041958043e-07, 'epoch': 1.92}                        
{'eval_loss': 0.18395505845546722, 'eval_accuracy': 0.95, 'eval_runtime': 14.4385, 'eval_samples_per_second': 83.111, 'eval_steps_per_second': 2.632, 'epoch': 2.0}
{'train_runtime': 3051.3391, 'train_samples_per_second': 5.995, 'train_steps_per_second': 0.187, 'train_loss': 0.1919890967282382, 'epoch': 2.0}
100%|███████████████████████████████████████████████████████████████████████████████████████| 572/572 [50:51<00:00,  5.33s/it] 
Evaluating on validation set...
100%|█████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:14<00:00,  2.71it/s]
Final Test Accuracy: 0.9500
'''