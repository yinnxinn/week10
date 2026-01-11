from pathlib import Path


class Settings:
    """Centralised application settings."""

    base_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    storage_dir: Path = base_dir / "storage"
    index_path: Path = storage_dir / "faiss.index"
    metadata_path: Path = storage_dir / "metadata.json"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # model_name: str = "C:/Users/www19/.cache/huggingface/hub/models--sentence-transformers--paraphrase-albert-small-v2"#"sentence-transformers/paraphrase-albert-small-v2"

    @classmethod
    def ensure_directories(cls) -> None:
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        cls.raw_data_dir.mkdir(parents=True, exist_ok=True)
        cls.storage_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
