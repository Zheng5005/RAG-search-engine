import argparse

from lib.hybrid_search import HybridSearch, correct_spelling, expand_query, rewrite_query
from lib.semantic_search import load_movies


def normalize(scores: list[float]) -> None:
    if not scores:
        return
    lo, hi = min(scores), max(scores)
    if lo == hi:
        for _ in scores:
            print(f"* 1.0000")
    else:
        for s in scores:
            print(f"* {(s - lo) / (hi - lo):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

    ws_parser = subparsers.add_parser("weighted-search", help="Hybrid search with configurable alpha")
    ws_parser.add_argument("query", type=str, help="Search query")
    ws_parser.add_argument("--alpha", type=float, default=0.5, help="BM25 vs semantic weight (default: 0.5)")
    ws_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter (default: 60)")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)
        case "weighted-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results, 1):
                doc = r["doc"]
                desc = doc.get("description", "")[:100]
                print(f"{i}. {doc['title']}")
                print(f"  Hybrid Score: {r['hybrid']:.3f}")
                print(f"  BM25: {r['bm25']:.3f}, Semantic: {r['semantic']:.3f}")
                print(f"  {desc}...")
                print()
        case "rrf-search":
            query = args.query
            if args.enhance == "spell":
                enhanced = correct_spelling(query)
                print(f"Enhanced query (spell): '{query}' -> '{enhanced}'\n")
                query = enhanced
            elif args.enhance == "rewrite":
                enhanced = rewrite_query(query)
                print(f"Enhanced query (rewrite): '{query}' -> '{enhanced}'\n")
                query = enhanced
            elif args.enhance == "expand":
                expanded = expand_query(query)
                enhanced = f"{query} {expanded}"
                print(f"Enhanced query (expand): '{query}' -> '{enhanced}'\n")
                query = enhanced
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, args.k, args.limit)
            for i, r in enumerate(results, 1):
                doc = r["doc"]
                desc = doc.get("description", "")[:100]
                bm25_rank = str(r["bm25_rank"]) if r["bm25_rank"] is not None else "N/A"
                sem_rank = str(r["semantic_rank"]) if r["semantic_rank"] is not None else "N/A"
                print(f"{i}. {doc['title']}")
                print(f"  RRF Score: {r['rrf']:.3f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                print(f"  {desc}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
