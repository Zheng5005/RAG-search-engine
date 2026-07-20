import json
import argparse
from pathlib import Path

from lib.semantic_search import ChunkedSemanticSearch, SemanticSearch, embed_query_text, embed_text, load_movies, semantic_chunk_text, verify_embeddings, verify_model


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

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size word chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks")

    subparsers.add_parser("embed-chunks", help="Build and cache chunked embeddings for all movies")

    search_chunked_parser = subparsers.add_parser("search-chunked", help="Search movies by chunked semantic similarity")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    semantic_chunk_parser = subparsers.add_parser("semantic-chunk", help="Split text into sentence-boundary chunks")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences between chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify-embeddings":
            verify_embeddings()
        case "embed-query":
            embed_query_text(args.query)
        case "semantic-chunk":
            chunks = semantic_chunk_text(
                args.text,
                max_chunk_size=args.max_chunk_size,
                overlap=args.overlap,
            )
            print(f"Semantically chunking {sum(len(c) for c in chunks)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "chunk":
            words = args.text.split()
            step = args.chunk_size - args.overlap
            chunks = []
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + args.chunk_size])
                chunks.append(chunk)
            print(f"Chunking {sum(len(c) for c in chunks)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
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
        case "embed-chunks":
            documents = load_movies()
            cse = ChunkedSemanticSearch()
            embeddings = cse.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search-chunked":
            documents = load_movies()
            cse = ChunkedSemanticSearch()
            cse.load_or_create_chunk_embeddings(documents)
            results = cse.search_chunks(args.query, args.limit)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"   {r['document']}...")
        case "embed":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
