import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

query = '小儿肥胖不爱运动该如何医治'
cadidates = ['孩子肥胖爱吃肉理应怎样治疗', '小儿肥胖不爱运动应怎样治效果才好', '儿童中耳炎流黄水理应如何治效果好', '孩子肥胖懒理应如何诊治', '孩童中耳炎流黄水要如何治疗']
## 精召回后的结果
['小儿肥胖不爱运动应怎样治效果才好', '孩子肥胖懒理应如何诊治', '孩子肥胖爱吃肉理应怎样治疗']


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

pairs = [[query, cadidate] for cadidate in cadidates]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)