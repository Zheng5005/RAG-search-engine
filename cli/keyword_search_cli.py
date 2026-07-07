import argparse
import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer

_translate_table = str.maketrans("", "", string.punctuation)


def _strip_punct(text: str) -> str:
    return text.lower().translate(_translate_table)


def _load_stopwords(path: Path) -> list[str]:
    with open(path) as f:
        return [_strip_punct(w) for w in f.read().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            base = Path(__file__).resolve().parent.parent
            stopwords = _load_stopwords(base / "data" / "stopwords.txt")
            stemmer = PorterStemmer()

            query_tokens = [
                stemmer.stem(t)
                for t in _strip_punct(args.query).split()
                if t not in stopwords
            ]

            with open(base / "data" / "course-rag-movies.json") as f:
                movies = json.load(f)["movies"]

            results = []
            for m in movies:
                title_tokens = [
                    stemmer.stem(t)
                    for t in _strip_punct(m["title"]).split()
                    if t not in stopwords
                ]
                if any(qt in title_tokens for qt in query_tokens):
                    results.append(m)
                    if len(results) == 5:
                        break

            print(f"Searching for: {args.query}")
            for i, m in enumerate(results, 1):
                print(f"{i}. {m['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
