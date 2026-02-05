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
    # Dataset has 'train' and 'validation' splits.
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

    # 4. Freeze BERT backbone 
    # Freezing all parameters in the 'bert' encoder, leaving only the classifier (pooler + linear) trainable.
    # Note: model.bert includes embeddings and encoder layers.
    print("Freezing BERT parameters...")
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Optional: Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")

    # 5. Training Setup
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-3,  # Higher learning rate since we are only training the head
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
        # Optimize for speed
        dataloader_num_workers=0, # Avoid multiprocessing issues on Windows if any
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("Starting training...")
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
Freezing BERT parameters...
Trainable parameters: 1538 / 102269186 (0.00%)
Starting training...
{'loss': 0.5683, 'grad_norm': 1.4508851766586304, 'learning_rate': 0.0009657342657342657, 'epoch': 0.17}
{'loss': 0.4755, 'grad_norm': 2.2573654651641846, 'learning_rate': 0.0009307692307692308, 'epoch': 0.35}                       
{'loss': 0.4519, 'grad_norm': 0.9722388386726379, 'learning_rate': 0.0008958041958041958, 'epoch': 0.52}                       
{'loss': 0.4347, 'grad_norm': 4.079524040222168, 'learning_rate': 0.0008608391608391609, 'epoch': 0.7}                         
{'loss': 0.4014, 'grad_norm': 5.6702799797058105, 'learning_rate': 0.0008258741258741259, 'epoch': 0.87}
{'eval_loss': 0.3783881664276123, 'eval_accuracy': 0.8425, 'eval_runtime': 15.5058, 'eval_samples_per_second': 77.39, 'eval_steps_per_second': 2.451, 'epoch': 1.0}
{'loss': 0.3858, 'grad_norm': 2.202078342437744, 'learning_rate': 0.0007909090909090909, 'epoch': 1.05}
{'loss': 0.3732, 'grad_norm': 2.58988618850708, 'learning_rate': 0.000755944055944056, 'epoch': 1.22}
{'loss': 0.4092, 'grad_norm': 2.185835599899292, 'learning_rate': 0.000720979020979021, 'epoch': 1.4}
{'loss': 0.4014, 'grad_norm': 5.6702799797058105, 'learning_rate': 0.0008258741258741259, 'epoch': 0.87}
{'eval_loss': 0.3783881664276123, 'eval_accuracy': 0.8425, 'eval_runtime': 15.5058, 'eval_samples_per_second': 77.39, 'eval_steps_per_second': 2.451, 'epoch': 1.0}
{'loss': 0.3858, 'grad_norm': 2.202078342437744, 'learning_rate': 0.0007909090909090909, 'epoch': 1.05}
{'loss': 0.3732, 'grad_norm': 2.58988618850708, 'learning_rate': 0.000755944055944056, 'epoch': 1.22}
{'loss': 0.4092, 'grad_norm': 2.185835599899292, 'learning_rate': 0.000720979020979021, 'epoch': 1.4}
{'loss': 0.3721, 'grad_norm': 0.796873152256012, 'learning_rate': 0.000686013986013986, 'epoch': 1.57}
{'loss': 0.3918, 'grad_norm': 1.0188435316085815, 'learning_rate': 0.0006510489510489511, 'epoch': 1.75}
{'loss': 0.3741, 'grad_norm': 2.9935355186462402, 'learning_rate': 0.0006160839160839161, 'epoch': 1.92}
{'eval_loss': 0.35675644874572754, 'eval_accuracy': 0.8516666666666667, 'eval_runtime': 15.4011, 'eval_samples_per_second': 77.917, 'eval_steps_per_second': 2.467, 'epoch': 2.0}
{'loss': 0.3677, 'grad_norm': 3.8044657707214355, 'learning_rate': 0.0005811188811188811, 'epoch': 2.1}
{'loss': 0.4014, 'grad_norm': 5.6702799797058105, 'learning_rate': 0.0008258741258741259, 'epoch': 0.87}
{'eval_loss': 0.3783881664276123, 'eval_accuracy': 0.8425, 'eval_runtime': 15.5058, 'eval_samples_per_second': 77.39, 'eval_steps_per_second': 2.451, 'epoch': 1.0}
{'loss': 0.3858, 'grad_norm': 2.202078342437744, 'learning_rate': 0.0007909090909090909, 'epoch': 1.05}
{'loss': 0.3732, 'grad_norm': 2.58988618850708, 'learning_rate': 0.000755944055944056, 'epoch': 1.22}
{'loss': 0.4092, 'grad_norm': 2.185835599899292, 'learning_rate': 0.000720979020979021, 'epoch': 1.4}
{'loss': 0.3721, 'grad_norm': 0.796873152256012, 'learning_rate': 0.000686013986013986, 'epoch': 1.57}
{'loss': 0.3918, 'grad_norm': 1.0188435316085815, 'learning_rate': 0.0006510489510489511, 'epoch': 1.75}
ps_per_second': 2.451, 'epoch': 1.0}
{'loss': 0.3858, 'grad_norm': 2.202078342437744, 'learning_rate': 0.0007909090909090909, 'epoch': 1.05}
{'loss': 0.3732, 'grad_norm': 2.58988618850708, 'learning_rate': 0.000755944055944056, 'epoch': 1.22}
{'loss': 0.4092, 'grad_norm': 2.185835599899292, 'learning_rate': 0.000720979020979021, 'epoch': 1.4}
{'loss': 0.4092, 'grad_norm': 2.185835599899292, 'learning_rate': 0.000720979020979021, 'epoch': 1.4}
{'loss': 0.3721, 'grad_norm': 0.796873152256012, 'learning_rate': 0.000686013986013986, 'epoch': 1.57}
{'loss': 0.3918, 'grad_norm': 1.0188435316085815, 'learning_rate': 0.0006510489510489511, 'epoch': 1.75}
{'loss': 0.3741, 'grad_norm': 2.9935355186462402, 'learning_rate': 0.0006160839160839161, 'epoch': 1.92}
{'eval_loss': 0.35675644874572754, 'eval_accuracy': 0.8516666666666667, 'eval_runtime': 15.4011, 'eval_samples_per_second': 77.917, 'eval_steps_per_second': 2.467, 'epoch': 2.0}
{'loss': 0.3677, 'grad_norm': 3.8044657707214355, 'learning_rate': 0.0005811188811188811, 'epoch': 2.1}
{'loss': 0.3748, 'grad_norm': 3.248927593231201, 'learning_rate': 0.0005461538461538461, 'epoch': 2.27}
{'loss': 0.3651, 'grad_norm': 2.1220643520355225, 'learning_rate': 0.0005111888111888112, 'epoch': 2.45}
{'loss': 0.3665, 'grad_norm': 2.4843504428863525, 'learning_rate': 0.00047622377622377624, 'epoch': 2.62}
{'loss': 0.3527, 'grad_norm': 1.86616849899292, 'learning_rate': 0.0004412587412587413, 'epoch': 2.8}
{'loss': 0.3835, 'grad_norm': 1.6668258905410767, 'learning_rate': 0.0004062937062937063, 'epoch': 2.97}
{'eval_loss': 0.3456900417804718, 'eval_accuracy': 0.8583333333333333, 'eval_runtime': 15.4781, 'eval_samples_per_second': 77.529, 'eval_steps_per_second': 2.455, 'epoch': 3.0}
{'loss': 0.3446, 'grad_norm': 0.7579076886177063, 'learning_rate': 0.00037132867132867134, 'epoch': 3.15}
{'loss': 0.3741, 'grad_norm': 1.4699918031692505, 'learning_rate': 0.0003363636363636364, 'epoch': 3.32}
{'loss': 0.394, 'grad_norm': 2.200326681137085, 'learning_rate': 0.0003013986013986014, 'epoch': 3.5}
{'loss': 0.3563, 'grad_norm': 1.4209778308868408, 'learning_rate': 0.0002664335664335664, 'epoch': 3.67}
{'loss': 0.3574, 'grad_norm': 3.874807357788086, 'learning_rate': 0.00023146853146853148, 'epoch': 3.85}
{'eval_loss': 0.34546560049057007, 'eval_accuracy': 0.8566666666666667, 'eval_runtime': 15.4657, 'eval_samples_per_second': 77.591, 'eval_steps_per_second': 2.457, 'epoch': 4.0}
{'loss': 0.3402, 'grad_norm': 0.8494391441345215, 'learning_rate': 0.0001965034965034965, 'epoch': 4.02}
{'loss': 0.355, 'grad_norm': 4.645857334136963, 'learning_rate': 0.00016153846153846155, 'epoch': 4.2}
{'loss': 0.3564, 'grad_norm': 2.5145318508148193, 'learning_rate': 0.0001265734265734266, 'epoch': 4.37}
{'loss': 0.3692, 'grad_norm': 2.228257179260254, 'learning_rate': 9.160839160839161e-05, 'epoch': 4.55}
{'loss': 0.3561, 'grad_norm': 1.3370815515518188, 'learning_rate': 5.6643356643356645e-05, 'epoch': 4.72}                      
{'loss': 0.3389, 'grad_norm': 2.2769477367401123, 'learning_rate': 2.1678321678321677e-05, 'epoch': 4.9}                       
{'eval_loss': 0.3384103775024414, 'eval_accuracy': 0.8608333333333333, 'eval_runtime': 15.4941, 'eval_samples_per_second': 77.449, 'eval_steps_per_second': 2.453, 'epoch': 5.0}
{'train_runtime': 694.0126, 'train_samples_per_second': 65.892, 'train_steps_per_second': 2.06, 'train_loss': 0.3851397287595522, 'epoch': 5.0}
100%|█████████████████████████████████████████████████████████████████████████████████████| 1430/1430 [11:34<00:00,  2.06it/s] 
Evaluating on validation set...
100%|█████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:15<00:00,  2.52it/s]
Final Test Accuracy: 0.8608
'''