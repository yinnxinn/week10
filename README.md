# 知识库构建示例项目

该目录演示一个从零开始搭建本地知识库的最小流程：解析原始文本、生成语义向量、利用 FAISS 构建向量索引，并通过 FastAPI 提供统一接口。

## 目录结构

```
.
|-- app/
|   |-- api/            # FastAPI 路由与接口定义
|   |-- core/           # 应用配置与初始化
|   |-- models/         # Pydantic 数据模型
|   `-- services/       # 文本解析、向量化与知识库逻辑
|-- data/
|   `-- raw/            # 待解析的纯文本数据源
|-- scripts/
|   `-- ingest.py       # 批量入库脚本
|-- storage/            # FAISS 索引与元数据输出
|-- README.md
`-- requirements.txt
```

## 基础原则

- **统一数据入口**：原始语料以 `.txt` 形式放在 `data/raw`，保持格式一致、方便批量处理。
- **灵活文本切分**：`app/services/parser.py` 支持按词数切块并可配置重叠，兼顾语义连续性与索引精度。
- **向量标准化**：`app/services/embeddings.py` 使用 `sentence-transformers/all-MiniLM-L6-v2` 生成归一化向量，以内积分值近似余弦相似度。
- **向量+元数据双存储**：向量保存于 FAISS，文本与附加信息序列化到 `metadata.json`，便于追溯来源与扩展字段。
- **组件职责清晰**：CLI 入库脚本与 FastAPI 接口复用同一服务层，保证业务逻辑集中可维护。

## 快速开始

1. **安装依赖（建议先创建虚拟环境）**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **准备数据**  
   将需要入库的文本文件放入 `data/raw/`，仓库默认提供 `example.txt` 作为演示。

3. **批量入库**

   ```bash
   python scripts/ingest.py --source data/raw --chunk-size 300 --chunk-overlap 50
   ```

   命令执行后会生成或更新：

   - `storage/faiss.index`：FAISS 向量索引
   - `storage/metadata.json`：向量对应的文本与元数据

4. **启动 API 服务**

   ```bash
   uvicorn app.main:app --reload
   ```

   - `GET /api/health`：健康检查
   - `POST /api/ingest`：直接通过接口写入新的文本片段
   - `POST /api/query`：语义检索，返回相似文本及相似度

遵循以上结构与原则，即可在此基础上扩展更复杂的知识库需求。
