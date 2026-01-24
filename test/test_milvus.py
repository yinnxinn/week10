from pymilvus import MilvusClient
import numpy as np

# 1. 初始化客户端 (这里使用 Milvus Lite，会直接在本地生成一个 .db 文件，无需部署 Docker)
# 如果你有远程 Milvus 服务器，请改为: uri="http://localhost:19530"
client = MilvusClient("http://localhost:19530")

collection_name = "demo_collection"

# 2. 如果集合已存在则删除，方便重复运行测试
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# 3. 创建集合
# 只需要定义维度 (dimension)。MilvusClient 会自动创建一个名为 'id' 的主键和 'vector' 的向量列
client.create_collection(
    collection_name=collection_name,
    dimension=5,  # 向量维度，实际应用中一般是 768 (BERT) 或 1536 (OpenAI)
)

# 4. 准备数据
# 包含：向量 (vector)、文本内容 (text)、分类 (subject)
data = [
    {"id": i, "vector": np.random.random(5).tolist(), "text": f"这是文档内容 {i}", "subject": "history"}
    for i in range(5)
]
# 再加 5 条不同分类的数据
data += [
    {"id": i, "vector": np.random.random(5).tolist(), "text": f"这是文档内容 {i}", "subject": "science"}
    for i in range(5, 10)
]

# 5. 插入数据
print(f"正在插入 {len(data)} 条数据...")
res = client.insert(
    collection_name=collection_name,
    data=data
)
print(f"插入完成，返回值: {res}")

# 6. 向量相似度搜索
# 模拟一个查询向量
query_vector = [0.35, 0.42, 0.67, 0.12, 0.88]

print("\n--- 正在进行向量搜索 ---")
search_res = client.search(
    collection_name=collection_name,
    data=[query_vector],        # 查询向量
    limit=3,                    # 返回最近的前 3 条
    search_params={"metric_type": "COSINE", "params": {}}, # 使用欧氏距离
    output_fields=["text", "subject"]  # 指定返回的非向量字段
)

for hits in search_res:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']:.4f}, 内容: {hit['entity']}")

# 7. 带条件过滤的搜索 (Scalar Filtering)
# 搜索向量相似度，且 subject 必须是 'science'
print("\n--- 正在进行带条件过滤的搜索 (subject=='science') ---")
filter_res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    filter="subject == 'science'", # 标量过滤语句
    limit=3,
    output_fields=["text", "subject"]
)

for hits in filter_res:
    for hit in hits:
        print(f"ID: {hit['id']}, 分类: {hit['entity']['subject']}, 内容: {hit['entity']['text']}")

# 8. 属性查询 (Query)
print("\n--- 正在查询 ID 大于 8 的数据 ---")
query_res = client.query(
    collection_name=collection_name,
    filter="id > 8",
    output_fields=["text"]
)
print(query_res)