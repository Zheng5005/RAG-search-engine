import json
import math
import pickle
import string
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer


_translate_table = str.maketrans("", "", string.punctuation)


def _strip_punct(text: str) -> str:
    return text.lower().translate(_translate_table)


def _load_stopwords(path: Path) -> list[str]:
    with open(path) as f:
        return [_strip_punct(w) for w in f.read().splitlines()]


def tokenize_text(text: str, stopwords: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    tokens = _strip_punct(text).split()
    tokens = [t for t in tokens if t not in stopwords]
    return [stemmer.stem(t) for t in tokens]


def tokenize_term(term: str, stopwords: list[str]) -> str:
    tokens = tokenize_text(term, stopwords)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token, got {len(tokens)}: {tokens}")
    return tokens[0]


def load_movies() -> list[dict]:
    base = Path(__file__).resolve().parent.parent
    with open(base / "data" / "course-rag-movies.json") as f:
        return json.load(f)["movies"]


BM25_K1 = 1.5
BM25_B = 0.75


def bm25_idf_command(term: str) -> float:
    stopwords = _load_stopwords(Path(__file__).resolve().parent.parent / "data" / "stopwords.txt")
    index = InvertedIndex()
    index.load()
    token = tokenize_term(term, stopwords)
    return index.get_bm25_idf(token)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    stopwords = _load_stopwords(Path(__file__).resolve().parent.parent / "data" / "stopwords.txt")
    index = InvertedIndex()
    index.load()
    token = tokenize_term(term, stopwords)
    return index.get_bm25_tf(doc_id, token, k1, b)


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str, stopwords: list[str]) -> None:
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        tokens = tokenize_text(text, stopwords)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, []))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index.get(term, []))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        avg_len = self.__get_avg_doc_length()
        doc_len = self.doc_lengths.get(doc_id, 0)
        length_norm = 1 - b + b * (doc_len / avg_len) if avg_len > 0 else 1.0
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, str, float]]:
        base = Path(__file__).resolve().parent.parent
        stopwords = _load_stopwords(base / "data" / "stopwords.txt")
        query_tokens = tokenize_text(query, stopwords)

        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            total = 0.0
            for token in query_tokens:
                total += self.bm25(doc_id, token)
            if total > 0:
                scores[doc_id] = total

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [(doc_id, self.docmap[doc_id]["title"], score) for doc_id, score in ranked]

    def build(self) -> None:
        base = Path(__file__).resolve().parent.parent
        stopwords = _load_stopwords(base / "data" / "stopwords.txt")
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}", stopwords)

    def save(self) -> None:
        base = Path(__file__).resolve().parent.parent
        cache_dir = base / "cache"
        cache_dir.mkdir(exist_ok=True)
        with open(cache_dir / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open(cache_dir / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open(cache_dir / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(cache_dir / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        base = Path(__file__).resolve().parent.parent
        index_path = base / "cache" / "index.pkl"
        docmap_path = base / "cache" / "docmap.pkl"
        tf_path = base / "cache" / "term_frequencies.pkl"
        dl_path = base / "cache" / "doc_lengths.pkl"
        if not index_path.exists() or not docmap_path.exists() or not tf_path.exists() or not dl_path.exists():
            raise FileNotFoundError("Index files not found. Run 'build' first.")
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(dl_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
