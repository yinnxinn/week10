from datasets import load_dataset
import numpy as np

try:
    dataset = load_dataset("left0ver/sentiment-classification")
    print(dataset)
    print(dataset['train'][0])
    
    lengths = [len(x['text']) for x in dataset['train']]
    print(f"Average length: {np.mean(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95)}")
except Exception as e:
    print(e)
