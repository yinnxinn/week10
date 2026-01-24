
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.knowledge_base import KnowledgeBase
from app.services.embeddings import EmbeddingService
from app.core.config import settings

def main():
    print("Initializing services...")
    
    # 1. Initialize Embedding Service (CLIP or All-MiniLM)
    # Using 'clip' as requested for multi-modal support
    embedding_service = EmbeddingService("clip")
    
    # 2. Initialize Knowledge Base
    # This attempts to connect to Milvus at settings.mivlus_host
    try:
        kb = KnowledgeBase()
        print(f"Connected to Milvus at {settings.mivlus_host}")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        print("Please ensure Milvus is running (e.g., using Docker).")
        return

    # 3. Prepare Dummy Data
    print("Preparing data...")
    documents = [
        {
            "id": "1",
            "text": "儿科医生建议：孩子发烧超过38.5度需要服用退烧药。",
            "department": "儿科",
            "title": "儿童发烧指南",
            "type": "medical_advice"
        },
        {
            "id": "2",
            "text": "骨科专家指出：经常运动有助于骨骼健康。",
            "department": "骨科",
            "title": "骨骼健康",
            "type": "medical_advice"
        },
        {
            "id": "3",
            "text": "感冒引起的咳嗽可以使用止咳糖浆。",
            "department": "内科",
            "title": "感冒咳嗽",
            "type": "medical_advice"
        }
    ]
    
    # Generate embeddings for documents
    print("Generating embeddings...")
    for doc in documents:
        # Combine text fields for embedding
        content = f"{doc['title']} {doc['text']}"
        # embed_query returns shape (1, dim), flatten to (dim,)
        doc['vector'] = embedding_service.embed_query(content)[0]
    
    # 4. Save Data (Insert into Milvus)
    print("Saving data to Knowledge Base...")
    kb.save(documents)
    print("Data saved.")
    
    # 5. Perform Hybrid Search
    query_text = "孩子发烧怎么办？"
    print(f"\nPerforming Hybrid Search for query: '{query_text}'")
    
    # Generate query embedding
    query_vector = embedding_service.embed_query(query_text)[0]
    
    # Execute Search
    # This uses:
    # - Tag extraction (finds "儿科" if present, though "孩子" might not trigger it, "儿科" in text will)
    # - Dense Vector Search
    # - Sparse Vector Search (BM25 via Function) - implicit in Milvus logic if enabled, 
    #   but my implementation currently focuses on Dense + Filter + RRF.
    results = kb.hybrid_search(
        query_text=query_text,
        query_dense_vector=query_vector,
        top_k=3,
        rerank=True # Enable RRF and Fine Reranking
    )
    
    print(f"Found {len(results[0])} results:")
    for res in results[0]:
        entity = res.get('entity', {})
        score = res.get('score') # RRF score or Rerank score
        rerank_score = res.get('rerank_score')
        print(f"- [Score: {rerank_score:.4f}] {entity.get('title')}: {entity.get('text')}")

if __name__ == "__main__":
    main()
