import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

load_dotenv()

MODEL = "openrouter/free"

SPELL_PROMPT = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{{query}}"
"""

REWRITE_PROMPT = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{{query}}"
"""

EXPAND_PROMPT = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{{query}}"
"""


def _llm_query(prompt: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def correct_spelling(query: str) -> str:
    return _llm_query(SPELL_PROMPT.format(query=query))


def rewrite_query(query: str) -> str:
    return _llm_query(REWRITE_PROMPT.format(query=query))


def expand_query(query: str) -> str:
    return _llm_query(EXPAND_PROMPT.format(query=query))


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


RERANK_INDIVIDUAL_PROMPT = f"""Rate how well this movie matches the search query.

Query: "{{query}}"
Movie: {{title}} - {{document}}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""


def rerank_individual(results: list[dict], query: str) -> list[dict]:
    for i, r in enumerate(results):
        doc = r["doc"]
        prompt = RERANK_INDIVIDUAL_PROMPT.format(
            query=query,
            title=doc.get("title", ""),
            document=doc.get("description", "")[:100],
        )
        try:
            raw = _llm_query(prompt)
            score = float(raw.strip())
            score = max(0.0, min(10.0, score))
        except (ValueError, RuntimeError):
            score = 0.0
        r["rerank_score"] = score
        if i < len(results) - 1:
            time.sleep(3)
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results


RERANK_BATCH_PROMPT = f"""Rank the movies listed below by relevance to the following search query.

Query: "{{query}}"

Movies:
{{doc_list_str}}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""


def rerank_batch(results: list[dict], query: str) -> list[dict]:
    doc_list_str = "\n".join(
        f"- ID: {r['doc']['id']}, Title: {r['doc'].get('title', '')}"
        for r in results
    )
    prompt = RERANK_BATCH_PROMPT.format(query=query, doc_list_str=doc_list_str)
    try:
        raw = _llm_query(prompt)
        ranked_ids = json.loads(raw.strip())
    except (json.JSONDecodeError, RuntimeError):
        ranked_ids = []

    rank_map = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
    for r in results:
        r["rerank_rank"] = rank_map.get(r["doc"]["id"], len(results) + 1)
    results.sort(key=lambda x: x["rerank_rank"])
    return results
