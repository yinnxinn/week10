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


#### 2026.02.01
* 增加知识图谱数据: https://github.com/RuiqingDing/OpenCMKG.git
* 入库知识图谱，展示知识图谱的用法 test_kg.py
* 模型部署框架： vllm, ollama , xinference

#### 2026.02.26
* 增加 RAG / CoT / RAG+CoT 测试用例（test 目录）：
  * test_rag.py：RAG 检索、提示构建、空上下文回退、与 LLM 集成（mock）
  * test_cot.py：CoT 分步推理提示、回复解析（推理过程/最终答案）、步骤抽取、与 LLM 集成（mock）
  * test_rag_cot.py：RAG+CoT 联合流程（检索 + 分步推理提示）、端到端 mock 测试
* 前端流式输出优化（app/web_ui.py + app/services/llm.py）：
  * LLM 服务新增 `get_consultation_suggestion_stream()`，使用 `stream=True` 逐 chunk 返回内容
  * 检索阶段仅显示「正在检索知识库」spinner，检索完成后关闭
  * 智能分析/诊疗建议部分改为流式展示，随 LLM 返回实时更新 `message_placeholder.markdown()`
  * 对话历史仍保存完整回复内容