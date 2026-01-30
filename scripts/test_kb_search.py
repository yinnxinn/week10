import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is in PYTHONPATH
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from app.services.knowledge_base import KnowledgeBase
from app.services.embeddings import EmbeddingService
from app.core.config import settings

def test_text_search():
    print("\n" + "="*50)
    print("Testing Text Search (Hybrid)")
    print("="*50)
    
    # Initialize Services
    embedding_service = EmbeddingService("openai/clip-vit-base-patch32")
    kb = KnowledgeBase()
    
    # Query: Medical Question
    query_text = "0.9cm的肾结石怎么排出来"
    print(f"Query Text: {query_text}")
    
    # Generate Embedding
    # embed_query returns (1, dim), we need (dim,)
    query_vector = embedding_service.embed_query(query_text)[0]
    
    # Perform Hybrid Search
    try:
        results = kb.hybrid_search(
            query_text=query_text,
            query_dense_vector=query_vector,
            top_k=5,
            rerank=True 
        )
        
        if not results or not results[0]:
            print("No results found.")
        else:
            print(f"Found {len(results[0])} results:")
            for i, hit in enumerate(results[0]):
                entity = hit.get('entity', {})
                text = entity.get('text', '')[:100] + "..."
                score = hit.get('rerank_score', hit.get('score', 0))
                tags = entity.get('tags', '')
                print(f"[{i+1}] Score: {score:.4f} | Tags: {tags}")
                print(f"    Text: {text}")
                
    except Exception as e:
        print(f"Text Search Failed: {e}")

def test_image_search():
    print("\n" + "="*50)
    print("Testing Image Search")
    print("="*50)
    
    # Initialize Services
    embedding_service = EmbeddingService("openai/clip-vit-base-patch32")
    kb = KnowledgeBase()
    
    # Case 1: Text to Image Search (Find images matching text)
    query_text = "thyroid nodule ultrasound"
    print(f"\n[Case 1] Text-to-Image Search: '{query_text}'")
    
    # Embed text into shared CLIP space
    # embed_query returns (1, dim), we need (dim,)
    text_vector = embedding_service.embed_query(query_text)[0]
    
    # Search in 'image_vector' field
    try:
        results = kb.search_images(
            query_vector=text_vector,
            top_k=3,
            search_field="image_vector"
        )
        
        if not results or not results[0]:
            print("No results found.")
        else:
            for i, hit in enumerate(results[0]):
                entity = hit.get('entity', {})
                score = hit.get('distance', 0) # Milvus returns distance/score
                # For cosine similarity, higher is better
                print(f"[{i+1}] Score: {score:.4f}")
                print(f"    Source: {entity.get('source')}")
                print(f"    Caption: {entity.get('text', '')[:100]}...")
                
    except Exception as e:
        print(f"Text-to-Image Search Failed: {e}")

    # Case 2: Image to Image Search (Find images similar to an image)
    # Find a sample image
    roco_dir = project_root / "data" / "ROCOv2_Thyroid" / "data"
    # We can't easily pick a raw image file since they are in parquet. 
    # But for testing, we can simulate an image embedding or use a placeholder if we don't have a raw image file handy.
    # However, EmbeddingService needs a file path or PIL image.
    # Let's see if we can find any image file in the project or create a dummy one.
    # Or just skip if no image file is available.
    
    # Actually, we can use the text embedding of a description as a proxy for a "perfect" image embedding 
    # since CLIP aligns them. But better to use an actual image if possible.
    # Let's try to find an image in data/small_dataset if it exists (from previous tasks).
    
    small_dataset_path = project_root / "data" / "small_dataset" / "images"
    image_files = list(small_dataset_path.glob("*.jpg")) + list(small_dataset_path.glob("*.png"))
    
    if image_files:
        test_image_path = image_files[0]
        print(f"\n[Case 2] Image-to-Image Search using: {test_image_path.name}")
        
        image = Image.open(test_image_path)
        image_vector = embedding_service.embed_image(image)
        
        try:
            results = kb.search_images(
                query_vector=image_vector[0], # embed_image returns batch
                top_k=3,
                search_field="image_vector"
            )
            
            if not results or not results[0]:
                print("No results found.")
            else:
                for i, hit in enumerate(results[0]):
                    entity = hit.get('entity', {})
                    score = hit.get('distance', 0)
                    print(f"[{i+1}] Score: {score:.4f}")
                    print(f"    Source: {entity.get('source')}")
                    print(f"    Caption: {entity.get('text', '')[:100]}...")

        except Exception as e:
            print(f"Image-to-Image Search Failed: {e}")
    else:
        print("\n[Case 2] Image-to-Image Search skipped (no test image found).")

if __name__ == "__main__":
    test_text_search()
    test_image_search()
