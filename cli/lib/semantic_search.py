import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunk_text(
    text: str, max_chunk_size: int = 4, overlap: int = 1
) -> list[str]:
    """Split text into sentence-boundary chunks."""
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    # single non-punctuated sentence — keep as one chunk
    if len(sentences) == 1 and not sentences[0][-1] in ".!?":
        return [sentences[0]]

    chunks: list[str] = []
    step = max_chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + max_chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + max_chunk_size >= len(sentences):
            break
    return chunks


def load_movies() -> list[dict]:
    base = Path(__file__).resolve().parent.parent.parent
    with open(base / "data" / "course-rag-movies.json") as f:
        return json.load(f)["movies"]


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

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            doc = self.documents[i]
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for idx, doc in enumerate(documents):
            desc = doc.get("description", "")
            if not desc:
                continue
            chunks = semantic_chunk_text(desc, max_chunk_size=4, overlap=1)
            total_in_doc = len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_in_doc,
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        base = Path(__file__).resolve().parent.parent.parent
        cache_dir = base / "cache"
        cache_dir.mkdir(exist_ok=True)
        np.save(cache_dir / "chunk_embeddings.npy", self.chunk_embeddings)
        with open(cache_dir / "chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        base = Path(__file__).resolve().parent.parent.parent
        emb_path = base / "cache" / "chunk_embeddings.npy"
        meta_path = base / "cache" / "chunk_metadata.json"

        if emb_path.exists() and meta_path.exists():
            self.chunk_embeddings = np.load(emb_path)
            with open(meta_path) as f:
                meta = json.load(f)
            self.chunk_metadata = meta["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None:
            raise ValueError(
                "No chunk embeddings loaded. "
                "Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        from .search_utils import format_search_result

        chunk_scores: list[dict] = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_emb)
            meta = self.chunk_metadata[i]
            chunk_scores.append({
                "movie_idx": meta["movie_idx"],
                "chunk_idx": meta["chunk_idx"],
                "score": score,
            })

        # aggregate: keep best chunk score per movie
        best_scores: dict[int, dict] = {}
        for cs in chunk_scores:
            midx = cs["movie_idx"]
            if midx not in best_scores or cs["score"] > best_scores[midx]["score"]:
                best_scores[midx] = cs

        # sort by score desc, limit
        sorted_items = sorted(
            best_scores.values(), key=lambda x: x["score"], reverse=True
        )[:limit]

        results: list[dict] = []
        for item in sorted_items:
            doc = self.documents[item["movie_idx"]]
            results.append(format_search_result(
                doc, item["score"],
                metadata={"best_chunk_idx": item["chunk_idx"]},
            ))

        return results


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


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None:
    base = Path(__file__).resolve().parent.parent.parent
    with open(base / "data" / "course-rag-movies.json") as f:
        documents = json.load(f)["movies"]
    ss = SemanticSearch()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
