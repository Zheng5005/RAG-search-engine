import json
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


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str, stopwords: list[str]) -> None:
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokenize_text(text, stopwords):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, []))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

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

    def load(self) -> None:
        base = Path(__file__).resolve().parent.parent
        index_path = base / "cache" / "index.pkl"
        docmap_path = base / "cache" / "docmap.pkl"
        tf_path = base / "cache" / "term_frequencies.pkl"
        if not index_path.exists() or not docmap_path.exists() or not tf_path.exists():
            raise FileNotFoundError("Index files not found. Run 'build' first.")
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
