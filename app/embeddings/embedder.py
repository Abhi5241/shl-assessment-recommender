from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


class EmbeddingGenerator:

    def __init__(self):
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.data_path = Path(settings.PROCESSED_DATA_PATH) / "shl_assessments.csv"
        self.output_path = Path("data/embeddings")
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        logger.info("Loading processed dataset...")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} assessments")
        return df

    def generate_embeddings(self, texts):
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.success("Embeddings generated")
        return embeddings

    def save_embeddings(self, df, embeddings):
        emb_file = self.output_path / "embeddings.npy"
        meta_file = self.output_path / "metadata.pkl"

        np.save(emb_file, embeddings)

        with open(meta_file, "wb") as f:
            pickle.dump(df.to_dict("records"), f)

        logger.success(f"Embeddings saved at {emb_file}")
        logger.success(f"Metadata saved at {meta_file}")

    def run(self):
        df = self.load_data()

        texts = df["combined_text"].tolist()

        embeddings = self.generate_embeddings(texts)

        self.save_embeddings(df, embeddings)


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.run()