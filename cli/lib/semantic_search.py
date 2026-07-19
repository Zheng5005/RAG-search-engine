import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, device: str | None = None):
        kwargs = {"device": device} if device else {}
        self.model = SentenceTransformer("all-MiniLM-L6-v2", **kwargs)
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, dict] = {}

    def generate_embedding(self, text: str):
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        base = Path(__file__).resolve().parent.parent.parent
        cache_dir = base / "cache"
        cache_dir.mkdir(exist_ok=True)
        np.save(cache_dir / "movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        base = Path(__file__).resolve().parent.parent.parent
        cache_path = base / "cache" / "movie_embeddings.npy"

        if cache_path.exists():
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    base = Path(__file__).resolve().parent.parent.parent
    with open(base / "data" / "course-rag-movies.json") as f:
        documents = json.load(f)["movies"]
    ss = SemanticSearch()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
