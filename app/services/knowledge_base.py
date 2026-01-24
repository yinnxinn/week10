from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)

from app.core.config import settings


@dataclass
class KnowledgeBase:
    index_path: Path | None = None
    metadata_path: Path | None = None
    client: MilvusClient | None = field(init=False, default=None)
    _rerank_model: object | None = field(init=False, default=None)
    _rerank_tokenizer: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.connect()

    def connect(self) -> None:
        if self.client is not None:
            return
        self.client = MilvusClient(settings.mivlus_host)
        
        # Check if we need to initialize the schema (drop existing for this setup)
        # In production, we would check schema compatibility or migrate.
        if self.client.has_collection(settings.collection_name):
            # For this setup task, we assume we can recreate to apply the hybrid schema
            # self.client.drop_collection(settings.collection_name)
            # Or just return if it exists, assuming it's correct.
            # Given the user wants to implement the instance, let's assume if it exists we use it,
            # but if it doesn't match, it might fail. 
            # To be safe for the user's "implement" request, we should probably ensure it's created correctly.
            # Let's skip recreation if exists to preserve data, but warn or assume user handles it.
            return

        # Define Hybrid Search Schema
        schema = self.client.create_schema(auto_id=False)
        
        # Primary Key
        schema.add_field(
            field_name="id", 
            datatype=DataType.VARCHAR, 
            max_length=64, 
            is_primary=True, 
            description="document id"
        )
        # Raw Text (for BM25 generation and retrieval)
        schema.add_field(
            field_name="text", 
            datatype=DataType.VARCHAR, 
            max_length=2000, 
            enable_analyzer=True, 
            description="raw text content"
        )
        # Dense Vector (CLIP/BERT)
        schema.add_field(
            field_name="dense_vector", 
            datatype=DataType.FLOAT_VECTOR, 
            dim=settings.dim, 
            description="dense embedding"
        )
        # Sparse Vector (BM25 auto-generated)
        schema.add_field(
            field_name="sparse_vector", 
            datatype=DataType.SPARSE_FLOAT_VECTOR, 
            description="sparse embedding (BM25)"
        )
        # Metadata fields
        schema.add_field(
            field_name="department", 
            datatype=DataType.VARCHAR, 
            max_length=64, 
            description="department tag"
        )
        # Dynamic fields for other metadata
        schema.enable_dynamic_field = True

        # Add BM25 Function
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # Create Indices
        index_params = self.client.prepare_index_params()
        
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",  # or AUTOINDEX
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25", # Metric for sparse
            params={"drop_ratio_build": 0.2}
        )

        # Create Collection
        self.client.create_collection(
            collection_name=settings.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def save(self, data: List[dict]) -> None:
        self.connect()
        # Ensure data matches schema
        # We need 'id', 'text', 'dense_vector' (renamed from vector if needed), 'department'
        # Data passed in usually has 'id', 'vector', 'text', 'department', etc.
        formatted_data = []
        for item in data:
            formatted_item = {
                "id": str(item.get("id")),
                "text": item.get("text") or item.get("question") or item.get("title") or "",
                "dense_vector": item.get("vector"),
                "department": item.get("department", "unknown"),
                # Dynamic fields
                "title": item.get("title", ""),
                "ask": item.get("ask", ""),
                "question": item.get("question", ""),
                **{k: v for k, v in item.items() if k not in ["id", "text", "vector", "department", "title", "ask", "question"]}
            }
            formatted_data.append(formatted_item)
            
        self.client.insert(
            collection_name=settings.collection_name,
            data=formatted_data,
        )

    def _ensure_rerank_model(self) -> None:
        if self._rerank_model is not None and self._rerank_tokenizer is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._rerank_tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model_name)
        self._rerank_model = AutoModelForSequenceClassification.from_pretrained(settings.rerank_model_name)
        self._rerank_model.eval()

    def rerank(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        import torch

        self._ensure_rerank_model()
        pairs = [[query, candidate] for candidate in candidates]
        with torch.no_grad():
            inputs = self._rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = self._rerank_model(**inputs, return_dict=True).logits.view(-1).float()
            return scores.tolist()

    def extract_tags(self, query: str) -> List[str]:
        """Extract potential tags (e.g., departments) from the query."""
        known_departments = [
            "儿科", "内科", "外科", "妇产科", "骨科", "耳鼻喉科", 
            "眼科", "口腔科", "皮肤科", "急诊科", "中医科"
        ]
        tags = []
        for dept in known_departments:
            if dept in query:
                tags.append(dept)
        return tags

    def hybrid_search(
        self,
        query_text: str,
        query_dense_vector: np.ndarray,
        top_k: int = 5,
        rerank: bool = True,
    ) -> list[dict]:
        """
        Perform Hybrid Search (Dense + Sparse/BM25) with RRF Reranking.
        """
        self.connect()
        
        # 1. Tag Filter
        filter_expr = None
        tags = self.extract_tags(query_text)
        if tags:
            tags_str = ", ".join([f"'{t}'" for t in tags])
            filter_expr = f"department in [{tags_str}]"
            
        coarse_limit = max(settings.rerank_candidates, top_k)

        # 2. Prepare Search Requests
        
        # Dense Search Request
        dense_req = AnnSearchRequest(
            data=[query_dense_vector],
            ann_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=coarse_limit,
            expr=filter_expr
        )
        
        # Sparse Search Request (BM25)
        # Using the server-side BM25 function, we can pass the raw text in the search request
        # if the client and server version support it.
        sparse_req = AnnSearchRequest(
            data=[query_text],
            ann_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=coarse_limit,
            expr=filter_expr
        )
        
        reqs = [dense_req, sparse_req]
        
        # 3. Execute Hybrid Search
        # Uses RRFRanker (Reciprocal Rank Fusion)
        ranker = RRFRanker(k=60)
        
        results = self.client.search(
            collection_name=settings.collection_name,
            data=reqs,
            rerank=ranker,
            limit=coarse_limit,
            output_fields=["*"]
        )
        
        if not results:
            return []
            
        hits = results[0]
        if not hits:
            return []

        # 4. Fine Recall (Rerank)
        if not rerank:
            return [hits[:top_k]]
            
        candidates_text: List[str] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            text = (
                entity.get("text")
                or entity.get("question")
                or entity.get("ask")
                or entity.get("title")
                or ""
            )
            candidates_text.append(text)
            
        scores = self.rerank(query_text, candidates_text)
        
        scored_hits = []
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)
            scored_hits.append(hit)
            
        scored_hits.sort(key=lambda item: item["rerank_score"], reverse=True)
        
        return [scored_hits[:top_k]]

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        query_text: str | None = None,
    ) -> list[dict]:
        """Wrapper for hybrid search to maintain compatibility."""
        if query_text:
            return self.hybrid_search(query_text, embedding, top_k)
        else:
            # Fallback to simple dense search if no text provided
            self.connect()
            results = self.client.search(
                collection_name=settings.collection_name,
                data=[embedding],
                limit=top_k,
                search_params={"metric_type": "COSINE", "params": {}},
                output_fields=["*"],
            )
            return results

if __name__ == "__main__":
    # import pandas as pd
    # path = 'D:/projects/classes/week10/week10/data/me/儿科5-14000.csv'
    # datas = pd.read_csv(path, encoding='utf-8')
    # print(datas.head())
    # import csv
    # with open('D:/projects/classes/week10/week10/data/me/儿科5-14000.csv', 'r') as r:
    #     reader = csv.reader(r)
    #
    #     datas = list()
    #     try:
    #         for item in reader:
    #             datas.append(item)
    #     except:
    #         print(f'读取到{len(datas)}条有效数据')
    #
    #     from app.services.embeddings import EmbeddingService
    #     es = EmbeddingService("sentence-transformers/clip-ViT-B-32")
    #
    #     processed_data = list()
    #     for idx, data in enumerate(datas[1:100]):
    #         processed_data.append(
    #             {
    #                 "id": idx,
    #                 'department':data[0],
    #                 "title": data[1],
    #                 "ask": data[2],
    #                 "question": data[3],
    #                 "vector": es.embed_query(data[2] + data[3])[0]
    #
    #             }
    #         )
        kg = KnowledgeBase()
        # kg.save(processed_data)
        query = '男孩子，已经2岁了，这几天，孩子说自己耳朵又痒又疼，早上，有黄色的耳屎流出，另外，好像没什么食欲也很乏力，请问：孩童中耳炎流黄水要如何治疗。 抗生素药物是目前治疗中耳炎比较常用的，可酌情选。如果孩子情况比较严重的话也可配合一些局部治疗，比如消炎型的滴耳剂，孩子耳痛严重的时候，也是可以适量的使用点止痛的药物，要是伴随发高烧的情况，那么根据孩子的症状使用药物，严重的情况请尽快去医院进行救治，以上都是比较常用的治疗方法，但是如果孩子出现了耳膜穿孔的症状，需要及时的去医院进行手术治疗，治疗期间主要要给孩子做好保暖工作，避免着凉加剧症状。'
        from app.services.embeddings import EmbeddingService
        es = EmbeddingService("sentence-transformers/clip-ViT-B-32")
        results = kg.query(es.embed_query(query)[0])
        cadidates = []
        for item in results[0]:

            print(item['distance'], item['entity']['title'], item['entity']['ask'], item['entity']['question'])


        print(cadidates)


            

