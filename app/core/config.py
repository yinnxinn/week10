import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Settings:
    """Centralised application settings with environment variable support."""

    # Base Paths
    base_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    storage_dir: Path = base_dir / "storage"
    index_path: Path = storage_dir / "faiss.index"
    metadata_path: Path = storage_dir / "metadata.json"

    model_name: str = os.getenv("MODEL_NAME", "sentence-transformers/clip-ViT-B-32")
    rerank_model_name: str = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))

    mivlus_host = "http://localhost:19530"
    collection_name = "MEDICINE_KG"
    dim = 512

    # Text Processing Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "300"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Server Settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # API Settings
    api_title: str = os.getenv("API_TITLE", "Knowledge Base Demo")
    api_version: str = os.getenv("API_VERSION", "0.1.0")
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    @classmethod
    def ensure_directories(cls) -> None:
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        cls.raw_data_dir.mkdir(parents=True, exist_ok=True)
        cls.storage_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
