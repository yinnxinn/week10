from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from pymilvus import MilvusClient

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
        if not self.client.has_collection(settings.collection_name):
            self.client.create_collection(
                collection_name=settings.collection_name,
                dimension=settings.dim,
            )

    def save(self, data) -> None:
        self.connect()
        self.client.insert(
            collection_name=settings.collection_name,
            data=data,
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

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        query_text: str | None = None,
    ):
        self.connect()
        coarse_limit = top_k
        if query_text is not None:
            coarse_limit = max(settings.rerank_candidates, top_k)
        results = self.client.search(
            collection_name=settings.collection_name,
            data=[embedding],
            limit=coarse_limit,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["*"],
        )
        if not results:
            return results
        if query_text is None:
            if coarse_limit == top_k:
                return results
            trimmed_results = []
            for hits in results:
                trimmed_results.append(hits[:top_k])
            return trimmed_results
        hits = results[0]
        if not hits:
            return results
        candidates: List[str] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            text = (
                entity.get("text")
                or entity.get("question")
                or entity.get("ask")
                or entity.get("title")
                or ""
            )
            candidates.append(text)
        scores = self.rerank(query_text, candidates)
        scored_hits = []
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)
            scored_hits.append(hit)
        scored_hits.sort(key=lambda item: item["rerank_score"], reverse=True)
        return [scored_hits[:top_k]]

    def hybrid_search(self, query: str):
        return None

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


            

