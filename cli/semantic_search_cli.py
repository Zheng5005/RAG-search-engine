import json
import argparse
from pathlib import Path

from lib.semantic_search import SemanticSearch, embed_query_text, embed_text, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model is loaded correctly")

    embed_parser = subparsers.add_parser("embed", help="Generate embedding for a text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify-embeddings", help="Verify movie embeddings are built correctly")

    embed_query_parser = subparsers.add_parser("embed-query", help="Generate embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Search query to embed")

    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify-embeddings":
            verify_embeddings()
        case "embed-query":
            embed_query_text(args.query)
        case "search":
            base = Path(__file__).resolve().parent.parent
            with open(base / "data" / "course-rag-movies.json") as f:
                documents = json.load(f)["movies"]
            ss = SemanticSearch()
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"  {r['description'][:80]}...")
                print()
        case "embed":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
