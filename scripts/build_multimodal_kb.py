import sys
import os
from pathlib import Path

# Ensure project root is in PYTHONPATH
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import uuid
import io
import json
from PIL import Image
from tqdm import tqdm

from app.services.knowledge_base import KnowledgeBase
from app.services.embeddings import EmbeddingService
from app.core.config import settings

def build_multimodal_kb():
    # 1. Initialize Services
    print("Initializing services...")
    # Use CLIP model for both text and image
    embedding_service = EmbeddingService("openai/clip-vit-base-patch32")
    kb = KnowledgeBase()
    
    # 2. Reset Collection
    print(f"Resetting collection: {settings.collection_name}")
    kb.connect()
    if kb.client.has_collection(settings.collection_name):
        #kb.client.drop_collection(settings.collection_name)
        # Re-connect to trigger schema creation
        kb.client = None 
        kb.connect()
    print("Collection recreated with multimodal schema.")

    batch_size = 50
    
    # 3. Process Medical Data (Text Only)
    medical_csv_path = project_root / "data" / "me" / "result.csv"
    if medical_csv_path.exists():
        print(f"\nProcessing Medical Data from {medical_csv_path}...")
        try:
            df_med = pd.read_csv(medical_csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df_med = pd.read_csv(medical_csv_path, encoding='gb18030')
        
        # Prepare batch insertion
        
        med_data = []
        
        print(f"Total medical records: {len(df_med)}")
        
        for idx, row in tqdm(df_med.iterrows(), total=len(df_med), desc="Processing Medical Text"):
            # Construct composite text
            title = str(row.get('title', ''))
            ask = str(row.get('ask', ''))
            answer = str(row.get('answer', ''))
            department = str(row.get('department', ''))
            
            # Merge entities and llm_entities as tags
            entities = str(row.get('entities', ''))
            llm_entities = str(row.get('llm_entities', ''))
            
            tag_set = set()
            for s in [entities, llm_entities]:
                if s and str(s).lower() != 'nan':
                    # Split by space or comma just in case
                    parts = s.replace(',', ' ').split()
                    tag_set.update(parts)
            
            tags_str = ",".join(list(tag_set))
            
            # Combine fields for rich context
            text_content = f"Title: {title}\nQuestion: {ask}\nAnswer: {answer}"
            # Truncate to safe length (Milvus VARCHAR limit is often bytes, e.g., 8192 bytes. 2000 chars * 3-4 bytes/char ~ 6000-8000 bytes)
            if len(text_content) > 1500:
                text_content = text_content[:1500] 
            
            med_data.append({
                "text": text_content,
                "department": department,
                "title": title,
                "ask": ask,
                "answer": answer,
                "entities": entities,
                "llm_entities": llm_entities,
                "tags": tags_str,
                "source": "medical_qa"
            })
            
            if len(med_data) >= batch_size:
                insert_batch(kb, embedding_service, med_data, is_image=False)
                med_data = []
        
        # Insert remaining
        if med_data:
            insert_batch(kb, embedding_service, med_data, is_image=False)
            
    else:
        print(f"Warning: Medical data file not found at {medical_csv_path}")

    # 4. Process ROCO Data (Multimodal)
    roco_path = project_root / "data" / "ROCOv2_Thyroid" / "data" / "train-00000-of-00001-7d53673dfb4331dc.parquet"
    images_dir = project_root / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if roco_path.exists():
        print(f"\nProcessing ROCO Multimodal Data from {roco_path}...")
        df_roco = pd.read_parquet(roco_path)
        
        roco_data = []
        print(f"Total ROCO records: {len(df_roco)}")
        
        for idx, row in tqdm(df_roco.iterrows(), total=len(df_roco), desc="Processing Multimodal Data"):
            # Extract Image
            image_bytes = row['image']['bytes']
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Save Image to Disk
            image_id = str(uuid.uuid4())
            image_filename = f"{image_id}.jpg"
            image_path = images_dir / image_filename
            image.save(image_path)
            
            # Extract Text (Caption/Description)
            messages = row['messages']
            # Messages structure: User (Image query) -> Assistant (Description) -> User (QA) -> Assistant (Answer)
            # We want the first Assistant response which describes the image.
            # Usually messages[1] is the assistant response to the image.
            
            caption = ""
            qa_pairs = []
            
            # Use list comprehension to flatten and find text content
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except:
                    messages = []
            
            if not isinstance(messages, list):
                messages = []
                
            for msg in messages:
                if isinstance(msg, str):
                    try:
                        msg = json.loads(msg)
                    except:
                        continue
                        
                if not isinstance(msg, dict):
                    continue

                role = msg.get('role')
                content = msg.get('content', [])
                
                text_parts = [c.get('text', '') for c in content if c.get('type') == 'text' and c.get('text')]
                text_full = " ".join(text_parts)
                
                if role == 'assistant' and not caption:
                    caption = text_full
                
                # Collect QA history for metadata
                if text_full:
                    qa_pairs.append(f"{role}: {text_full}")
            
            # If no specific caption found, use all text
            if not caption:
                caption = "\n".join(qa_pairs)
                
            if len(caption) > 8000:
                caption = caption[:8000]

            roco_data.append({
                "image_obj": image, # Pass PIL Image object
                "text": caption,
                "department": "Radiology", # Inferred
                "source": "roco_thyroid",
                "image_path": str(image_path), # Save absolute path
                "qa_history": json.dumps(qa_pairs, ensure_ascii=False)
            })
            
            if len(roco_data) >= batch_size:
                insert_batch(kb, embedding_service, roco_data, is_image=True)
                roco_data = []
                
        # Insert remaining
        if roco_data:
            insert_batch(kb, embedding_service, roco_data, is_image=True)
            
    else:
        print(f"Warning: ROCO data file not found at {roco_path}")

    print("\nMultimodal Knowledge Base Construction Completed!")

def insert_batch(kb, embedding_service, batch_data, is_image=False):
    """
    Helper to embed and insert a batch of data.
    """
    texts = [item["text"] for item in batch_data]
    
    # 1. Text Embeddings
    text_embeddings = embedding_service.embed_documents(texts)
    
    # 2. Image Embeddings
    if is_image:
        images = [item["image_obj"] for item in batch_data]
        # CLIP supports batch image embedding? 
        # EmbeddingService.embed_image takes a single source or list?
        # Looking at implementation:
        # inputs = self._clip_processor(images=image, return_tensors="pt")
        # Processor handles lists of images.
        image_embeddings = embedding_service.embed_image(images)
    else:
        # Zero vectors for text-only data
        # Dim is 512 for CLIP
        image_embeddings = np.zeros((len(batch_data), settings.dim), dtype="float32")

    # 3. Format for Milvus
    insert_rows = []
    for i, item in enumerate(batch_data):
        row = {
            "id": str(uuid.uuid4()),
            "text": item["text"],
            "dense_vector": text_embeddings[i],
            "image_vector": image_embeddings[i],
            "department": item["department"],
            "tags": item.get("tags", ""),
            # Dynamic fields
            "source": item.get("source", ""),
        }
        # Add extra metadata
        for k, v in item.items():
            if k not in ["image_obj", "text", "department", "source", "tags"]:
                row[k] = v
        
        insert_rows.append(row)
        
    # 4. Insert
    try:
        kb.client.insert(
            collection_name=settings.collection_name,
            data=insert_rows
        )
    except Exception as e:
        print(f"Error inserting batch: {e}")

if __name__ == "__main__":
    build_multimodal_kb()
