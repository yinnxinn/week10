
import os
from datasets import load_from_disk
from pathlib import Path

def inspect_dataset():
    # Path to the downloaded dataset
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "Corvus-OCR-Caption-Mix"
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        # Try loading as a saved dataset (Arrow format)
        # Note: snapshot_download downloads the raw files. 
        # If it's a standard HF dataset structure with arrow files, 'load_dataset' with 'arrow' or loading via 'load_from_disk' might work if it was saved that way.
        # However, usually downloaded raw files need 'load_dataset' with the path.
        from datasets import load_dataset
        
        # Try loading using load_from_disk as suggested by error message
        # Since snapshot_download might have downloaded a saved-to-disk dataset structure
        dataset = load_from_disk(str(dataset_path))
        
        print("Dataset loaded successfully (load_from_disk).")
        
        # Take a peek at the first item
        item = next(iter(dataset))
        print("\nFirst item keys:", item.keys())
        print("First item example:", {k: str(v)[:100] for k, v in item.items()})
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    inspect_dataset()
