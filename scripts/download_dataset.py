
import os
# Set mirror for China BEFORE importing huggingface_hub
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from huggingface_hub import snapshot_download

def download_dataset():
    repo_id = "rainxx/Corvus-OCR-Caption-Mix"
    
    # Base data directory
    base_dir = Path(__file__).resolve().parents[1] / "data"
    
    # Target directory for this dataset
    local_dir = base_dir / "Corvus-OCR-Caption-Mix"
    
    print(f"Downloading {repo_id} to {local_dir}...")
    
    max_retries = 10
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}/{max_retries}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                resume_download=True,
                max_workers=2 # Reduce workers to avoid timeouts
            )
            print(f"Download completed successfully to {local_dir}")
            break
        except Exception as e:
            print(f"Error downloading dataset (attempt {i+1}): {e}")
            if i == max_retries - 1:
                print("Failed to download after multiple attempts.")

if __name__ == "__main__":
    download_dataset()
