from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "SHL Assessment Recommender"
    API_VERSION: str = "v1"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    RAW_DATA_PATH: str = "data/raw"
    PROCESSED_DATA_PATH: str = "data/processed"
    VECTOR_DB_PATH: str = "data/embeddings/faiss_index"
    TOP_K_RESULTS: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()