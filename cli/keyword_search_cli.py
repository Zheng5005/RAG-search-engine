import argparse
import json
import string
from pathlib import Path

from nltk.stem import PorterStemmer

from inverted_index import InvertedIndex

_translate_table = str.maketrans("", "", string.punctuation)


def _strip_punct(text: str) -> str:
    return text.lower().translate(_translate_table)


def _load_stopwords(path: Path) -> list[str]:
    with open(path) as f:
        return [_strip_punct(w) for w in f.read().splitlines()]


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and save the inverted index")

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()
        case "search":
            base = Path(__file__).resolve().parent.parent
            stopwords = _load_stopwords(base / "data" / "stopwords.txt")
            stemmer = PorterStemmer()

            query_tokens = [
                stemmer.stem(t)
                for t in _strip_punct(args.query).split()
                if t not in stopwords
            ]

            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return

            seen = set()
            results = []
            for token in query_tokens:
                for doc_id in index.get_documents(token):
                    if doc_id not in seen:
                        seen.add(doc_id)
                        results.append(index.docmap[doc_id])
                        if len(results) == 5:
                            break
                if len(results) == 5:
                    break

            print(f"Searching for: {args.query}")
            for i, m in enumerate(results, 1):
                print(f"{i}. {m['title']} (ID: {m['id']})")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
