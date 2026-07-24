from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.semantic_search import load_movies


class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32", documents: list[dict] | None = None) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [
            f"{doc['title']}: {doc.get('description', '')}" for doc in self.documents
        ]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=bool(self.texts))

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        img = Image.open(image_path)
        embedding = self.model.encode([img])[0]
        return embedding

    def search_with_image(self, image_path: str | Path) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        similarities = np.dot(self.text_embeddings, image_embedding) / (
            np.linalg.norm(self.text_embeddings, axis=1) * np.linalg.norm(image_embedding)
        )
        ranked = sorted(
            enumerate(similarities), key=lambda x: x[1], reverse=True
        )[:5]
        results = []
        for idx, score in ranked:
            doc = self.documents[idx]
            results.append({
                "id": idx,
                "title": doc["title"],
                "description": doc.get("description", ""),
                "similarity": float(score),
            })
        return results


def verify_image_embedding(image_path: str | Path) -> None:
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search(image_path: str | Path) -> list[dict]:
    documents = load_movies()
    ms = MultimodalSearch(documents=documents)
    return ms.search_with_image(image_path)
