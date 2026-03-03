import faiss
import numpy as np
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


class FAISSVectorStore:

    def __init__(self):
        self.embedding_path = Path("data/embeddings/embeddings.npy")
        self.metadata_path = Path("data/embeddings/metadata.pkl")
        self.index_path = Path(settings.VECTOR_DB_PATH)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.index = None
        self.metadata = None

    def load_embeddings(self):
        logger.info("Loading embeddings...")
        embeddings = np.load(self.embedding_path)

        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)

        logger.success(f"Loaded {len(metadata)} embeddings")

        return embeddings.astype("float32"), metadata

    def build_index(self):
        embeddings, metadata = self.load_embeddings()

        dimension = embeddings.shape[1]

        logger.info("Building FAISS index...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))

        logger.success("FAISS index saved")

        self.index = index
        self.metadata = metadata

    def load_index(self):
        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.success("FAISS index loaded")

    def search(self, query, top_k=10):
        if self.index is None:
            self.load_index()

        logger.info(f"Searching for: {query}")

        query_vector = self.model.encode([query]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results


if __name__ == "__main__":
    store = FAISSVectorStore()
    store.build_index()