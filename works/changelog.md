#### 2026.01.18
* embedding模型跟换，使用bert模型
* faiss基础库更换，替换为milvus向量数据库
  ```熟悉docker```
* 采用医疗病历的数据，数据有一定的结构，需要对架构做出一定的调整
```https://github.com/Mengqi97/chinese-medical-dataset?tab=readme-ov-file#51%E6%95%B0%E6%8D%AE%E9%9B%86%E6%B1%87%E6%80%BB```
* 共享资料地址 https://github.com/yinnxinn/learningHub


#### 2026.01.24
* 增加test_gpt.py , 利用gpt模型训练一个生成模型
* 基于clip模型实现多模态知识库
* 基于pymilvus实现多模态混合检索
* 实现知识召回的全部完整链路，基于rainxx/Corvus-OCR-Caption-Mix数据集
* 增加了test_decode.py , 以gpt2模型为例，解释不同的decode方法对于模型生成结果的影响
* scripts增加数据下载脚本 download_dataset.py
* 增加模型下载脚本 download_clip_model.py


#### 2026.02.01
* 增加test_lora.py, 说明lora的工作原理
* 增加lora_cls.py, 解释模型训练时的lora使用流程，主要基于peft库
