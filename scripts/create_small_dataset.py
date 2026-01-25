
import os
import json
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm

def create_small_dataset():
    # Paths
    base_dir = Path(__file__).resolve().parents[1] / "data"
    source_dataset_path = base_dir / "Corvus-OCR-Caption-Mix"
    target_dir = base_dir / "small_dataset"
    images_dir = target_dir / "images"
    metadata_path = target_dir / "metadata.jsonl"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {source_dataset_path}...")
    try:
        dataset = load_from_disk(str(source_dataset_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Dataset loaded. Processing 1000 items...")
    
    # We can iterate directly or select range
    # dataset[i] accesses the item
    
    count = 0
    limit = 1000
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        # Using tqdm for progress bar
        for i in tqdm(range(limit)):
            try:
                item = dataset[i]
                image = item["image"]
                text = item["text"]
                
                # Generate an ID
                item_id = f"img_{i:04d}"
                
                # Save image
                image_filename = f"{item_id}.png" # Assuming PNG is safer for OCR data usually, but dataset has PIL objects
                image_path = images_dir / image_filename
                
                # Convert to RGB if necessary (though inspection showed RGB)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                image.save(image_path)
                
                # Prepare metadata
                metadata = {
                    "id": item_id,
                    "image_path": f"images/{image_filename}", # Relative path usually better
                    "text": text
                }
                
                # Write metadata
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                
                count += 1
                
            except IndexError:
                print(f"Dataset has fewer than {limit} items. Stopped at {count}.")
                break
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
                
    print(f"Successfully processed {count} items.")
    print(f"Images saved to: {images_dir}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    create_small_dataset()
