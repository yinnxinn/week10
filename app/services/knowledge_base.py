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
            max_length=8192, 
            enable_analyzer=True, 
            description="raw text content"
        )
        # Dense Vector (CLIP/BERT)
        schema.add_field(
            field_name="dense_vector", 
            datatype=DataType.FLOAT_VECTOR, 
            dim=settings.dim, 
            description="text embedding"
        )

        schema.add_field(
            field_name="image_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=settings.dim,
            description="image embedding"
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
            field_name="image_vector",
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
            filter_expr = '' #f"department in [{tags_str}]"
            
        coarse_limit = max(settings.rerank_candidates, top_k)

        # 2. Prepare Search Requests
        
        # Dense Search Request
        dense_req = AnnSearchRequest(
            data=[query_dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=coarse_limit,
            expr=filter_expr
        )
        
        # Sparse Search Request (BM25)
        # Using the server-side BM25 function, we can pass the raw text in the search request
        # if the client and server version support it.
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=coarse_limit,
            expr=filter_expr
        )
        
        reqs = [dense_req, sparse_req]
        
        # 3. Execute Hybrid Search
        # Uses RRFRanker (Reciprocal Rank Fusion)
        ranker = RRFRanker(k=60)
        
        # NOTE: MilvusClient.search() validation might fail with AnnSearchRequest list in some versions.
        # We use the underlying Collection object to perform hybrid search.
        from pymilvus import Collection
        
        # Use the connection established by MilvusClient (usually 'default' or internal alias)
        # MilvusClient(uri=...) sets up a connection. 
        # If we encounter issues finding the connection, we might need to explicitly connect, 
        # but self.connect() does create a MilvusClient.
        # However, MilvusClient manages its own connection. 
        # To use Collection(), we need a registered connection. 
        # Since MilvusClient might use a generated alias, let's try to get it or fallback.
        
        # Safe way: use the client's internal alias if available, or just create a temp Collection with connection reuse
        # But Collection() needs 'using' alias. 
        # self.client._using is the alias used by MilvusClient.
        
        try:
            col = Collection(settings.collection_name, using=self.client._using)
            results = col.hybrid_search(
                reqs,
                ranker,
                limit=coarse_limit,
                output_fields=["*"]
            )
        except Exception as e:
            # Fallback if _using is not accessible or other error
            print(f"Hybrid search via Collection failed: {e}. Trying alternative...")
            # If explicit connection is needed (MilvusClient might not register it globally as we expect)
            # This is a fallback but unlikely needed if _using works.
            raise e
        
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
    
    import json
    import random
    from pathlib import Path
    from app.services.embeddings import EmbeddingService

    embedding_service = EmbeddingService("openai/clip-vit-base-patch32")
    kb = KnowledgeBase()

    '''
    # 1. Load Metadata
    base_dir = Path(__file__).resolve().parents[2]
    metadata_path = base_dir / "data" / "small_dataset" / "metadata.jsonl"
    
    print(f"Loading data from {metadata_path}...")
    data_items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and json.loads(line).get('text','').startswith("The image"):
                data_items.append(json.loads(line))
    
    print(f"Loaded {len(data_items)} items.")
    
    # 2. Initialize Services
    # Use the local CLIP model we downloaded
    
    
    # 3. Reset Collection (to ensure schema matches and start clean)
    collection_name = settings.collection_name
    if kb.client.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        kb.client.drop_collection(collection_name)
        # Re-connect to trigger schema creation
        kb.client = None 
        kb.connect()
        print("Collection recreated with hybrid schema.")

    # 4. Prepare Data for Ingestion
    print("Generating embeddings and preparing data...")
    processed_data = []
    
    # departments = [
    #     "儿科", "内科", "外科", "妇产科", "骨科", "耳鼻喉科",
    #     "眼科", "口腔科", "皮肤科", "急诊科", "中医科"
    # ]
    
    # Process in batches to show progress (optional, but good for 1000 items)
    batch_size = 50
    for i in range(0, len(data_items), batch_size):
        batch = data_items[i:i+batch_size]
        
        # Extract texts for batch embedding
        texts = [item["text"] for item in batch]
        
        # Generate dense vectors (Text Embedding via CLIP)
        # Note: CLIP text encoder has a limit of 77 tokens. Long texts will be truncated.
        vectors = embedding_service.embed_documents(texts)
        
        for j, item in enumerate(batch):
            # Assign a random department for demonstration of filtering
            dept = '' #random.choice(departments)

            image_path = base_dir / "data" / "small_dataset" / item["image_path"]

            processed_item = {
                "id": item["id"],
                "text": item["text"],
                "vector": vectors[j],
                'image_vector': embedding_service.embed_image(image_path)[0],
                "department": dept,
                # Store image path in dynamic field for retrieval
                "image_path": item["image_path"], 
                # Add other metadata if needed
                "type": "image-text-pair"
            }
            processed_data.append(processed_item)
        print(f"Processed {min(i + batch_size, len(data_items))}/{len(data_items)}")

    # 5. Save to Milvus
    print(f"Inserting {len(processed_data)} items into Milvus...")
    kb.save(processed_data)
    print("Ingestion complete.")


    # 6. Hybrid Search Example
    print("\n" + "="*50)
    print("Running Hybrid Search Example")
    print("="*50)
    
    # Example Query: Combine keywords and semantic meaning
    # Query intent: Looking for physics formulas related to force, possibly in '内科' (just to test filter, though unlikely match, let's try matching dept)
    # Let's try a query that matches some content.
    # Item 1 has "Newton's second law... F=ma".
    # Let's pretend we are looking for "force formula" in "儿科" (Pediatrics) - unlikely to find, 
    # but let's try a matching department if we assigned one, or just general search.
    # We assigned random departments. Let's do a general search first.
    '''
    search_text = "military parade"
    print(f"Query: '{search_text}'")
    
    # Generate query vector
    query_vector = embedding_service.embed_query(search_text)[0]
    
    # Perform Hybrid Search
    # Note: verify_hybrid_search_logic showed us how to call it.
    # kb.hybrid_search(query_text, query_dense_vector, top_k, rerank)
    
    results = kb.hybrid_search(
        query_text=search_text,
        query_dense_vector=query_vector,
        top_k=5,
        rerank=True
    )
    
    print(f"\nFound {len(results[0])} results:")
    for idx, hit in enumerate(results[0]):
        entity = hit.get("entity", {})
        print(f"\nResult {idx+1}:")
        print(f"  Score: {hit.get('score')} (Rerank: {hit.get('rerank_score', 'N/A')})")
        print(f"  ID: {hit.get('id')}")
        print(f"  Department: {entity.get('department')}")
        print(f"  Text Snippet: {entity.get('text')[:100]}...")
        print(f"  Image Path: {entity.get('image_path')}")

    print("\n" + "="*50)


            

