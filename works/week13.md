**基于数据集left0ver/sentiment-classification构建一个情感分类模型，要求：
1. 模型在测试集上的准确率不低于85%
2. 采用freeze bert模型的参数，只训练分类层的参数
3. 采用lora方式训练参数，tips: 采用peft库
4. 采用全参微调的方式，训练分类模型
5. 比较几个模型的准确率，思考产生差异的原因