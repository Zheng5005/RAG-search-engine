from pathlib import Path

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        cache_dir = Path(__file__).resolve().parent.parent.parent / "cache"
        index_ready = all(
            (cache_dir / name).exists()
            for name in ("index.pkl", "docmap.pkl", "term_frequencies.pkl", "doc_lengths.pkl")
        )
        if not index_ready:
            self.idx.build()
            self.idx.save()
        self.idx.load()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
        fetch = 500 * limit

        bm25_results = self._bm25_search(query, fetch)
        # bm25_results: list[tuple[int, str, float]] → (doc_id, title, score)
        bm25_ids = [r[0] for r in bm25_results]
        bm25_raw_scores = [r[2] for r in bm25_results]

        semantic_results = self.semantic_search.search_chunks(query, fetch)
        # semantic_results: list[dict] with "id", "score", etc.
        sem_ids = [r["id"] for r in semantic_results]
        sem_raw_scores = [r["score"] for r in semantic_results]

        bm25_norm = normalize_scores(bm25_raw_scores) if bm25_raw_scores else []
        sem_norm = normalize_scores(sem_raw_scores) if sem_raw_scores else []

        # build lookup: doc_id → normalized scores
        combined: dict[int, dict] = {}
        for doc_id, ns in zip(bm25_ids, bm25_norm):
            combined[doc_id] = {"doc": self.idx.docmap[doc_id], "bm25": ns, "semantic": 0.0}
        for doc_id, ns in zip(sem_ids, sem_norm):
            if doc_id in combined:
                combined[doc_id]["semantic"] = ns
            else:
                # doc only found by semantic search — use the raw document from documents list
                doc = self.documents[doc_id] if doc_id < len(self.documents) else {"id": doc_id, "title": "", "description": ""}
                combined[doc_id] = {"doc": doc, "bm25": 0.0, "semantic": ns}

        for entry in combined.values():
            entry["hybrid"] = hybrid_score(entry["bm25"], entry["semantic"], alpha)

        ranked = sorted(combined.values(), key=lambda x: x["hybrid"], reverse=True)[:limit]
        return ranked

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        fetch = 500 * limit

        bm25_results = self._bm25_search(query, fetch)
        semantic_results = self.semantic_search.search_chunks(query, fetch)

        combined: dict[int, dict] = {}

        for rank, (doc_id, _title, _score) in enumerate(bm25_results, start=1):
            combined[doc_id] = {
                "doc": self.idx.docmap[doc_id],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf": rrf_score(rank, k),
            }

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            if doc_id in combined:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["rrf"] += rrf_score(rank, k)
            else:
                doc = self.documents[doc_id] if doc_id < len(self.documents) else {"id": doc_id, "title": "", "description": ""}
                combined[doc_id] = {
                    "doc": doc,
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf": rrf_score(rank, k),
                }

        ranked = sorted(combined.values(), key=lambda x: x["rrf"], reverse=True)[:limit]
        return ranked
