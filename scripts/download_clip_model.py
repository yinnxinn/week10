
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Set mirror for China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_model():
    repo_id = "openai/clip-vit-base-patch32"
    
    # Base directory
    base_dir = Path(__file__).resolve().parents[1]
    models_dir = base_dir / "models" / "clip-vit-base-patch32"
    
    print(f"Downloading {repo_id} to {models_dir}...")
    
    max_retries = 5
    for i in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=models_dir,
                resume_download=True,
                max_workers=4
            )
            print(f"Model downloaded successfully to {models_dir}")
            break
        except Exception as e:
            print(f"Error downloading model (attempt {i+1}): {e}")

if __name__ == "__main__":
    download_model()
