import torch, os
from transformers import AutoTokenizer, AutoModelForMaskedLM
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def test_mlm_accuracy(model_name, test_sentences):
    # 1. 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    top1_correct = 0
    top5_correct = 0
    total = len(test_sentences)

    print(f"开始测试模型: {model_name}\n" + "-" * 30)

    for sentence, target_word in test_sentences:
        # 构造带 [MASK] 的输入
        masked_sentence = sentence.replace(target_word, tokenizer.mask_token)
        inputs = tokenizer(masked_sentence, return_tensors="pt")

        # 获取目标词的 token ID
        target_id = tokenizer.convert_tokens_to_ids(target_word.lower() if "uncased" in model_name else target_word)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # 形状: [1, sequence_length, vocab_size]

        # 找到 [MASK] 所在的位置
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        # 获取 [MASK] 位置的预测概率分布
        mask_token_logits = logits[0, mask_token_index, :]

        # 获取前 5 个预测结果
        top_5_indices = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

        print('[MASK]位置的前5个候选为： ', [tokenizer.decode(item) for item in top_5_indices])
        # 统计准确率
        if target_id == top_5_indices[0]:
            top1_correct += 1
        if target_id in top_5_indices:
            top5_correct += 1

        # 打印部分结果
        predicted_word = tokenizer.decode([top_5_indices[0]])
        print(f"原句: {sentence}")
        print(f"目标词: {target_word} | 预测词: {predicted_word} | 是否命中Top5: {target_id in top_5_indices}")
        print("-" * 20)

    print(f"\n测试完成!")
    print(f"Top-1 Accuracy: {top1_correct / total:.2%}")
    print(f"Top-5 Accuracy: {top5_correct / total:.2%}")


# --- 测试用例 ---
# 注意：目标词必须在分词器的词表中，且最好是单个词
test_data = [
    ## London is the capital of [MASK].
    # ("London is the capital of England.", "England"),
    # ("Deep learning is a subset of machine learning.", "machine"),
    # ("The cat is sitting on the mat.", "mat"),
    # ("Python is a popular programming language.", "language"),
    # ("The sun rises in the east.", "east")
    ("下雪了，天空飘着雪花。" , '花'),
    ("北京天安门是一座伟大的建筑" , '京')

]

test_mlm_accuracy("google-bert/bert-base-chinese", test_data)