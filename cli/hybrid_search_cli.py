import argparse
import json
import logging

from lib.hybrid_search import HybridSearch, _llm_query, correct_spelling, expand_query, rewrite_query, rerank_batch, rerank_cross_encoder, rerank_individual
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
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
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method",
    )
    rrf_parser.add_argument("--evaluate", action="store_true", help="Evaluate results with LLM")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

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
            logging.debug("Original query: %s", query)
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
            logging.debug("Enhanced query: %s", query)
            documents = load_movies()
            hs = HybridSearch(documents)
            fetch = 5 * args.limit if args.rerank_method else args.limit
            results = hs.rrf_search(query, args.k, fetch)
            logging.debug("RRF results (%d): %s", len(results), [(r["doc"]["title"], round(r["rrf"], 4)) for r in results[:10]])
            if args.rerank_method == "individual":
                print(f"Re-ranking top {len(results)} results using individual method...\n")
                results = rerank_individual(results, query)
            elif args.rerank_method == "batch":
                print(f"Re-ranking top {len(results)} results using batch method...\n")
                results = rerank_batch(results, query)
            elif args.rerank_method == "cross_encoder":
                print(f"Re-ranking top {len(results)} results using cross_encoder method...\n")
                results = rerank_cross_encoder(results, query)
            if args.rerank_method:
                logging.debug("Re-ranked results (%d): %s", len(results), [(r["doc"]["title"], round(r.get("rerank_score", r.get("rerank_rank", r.get("cross_encoder_score", 0))), 3)) for r in results[:10]])
            for i, r in enumerate(results[:args.limit], 1):
                doc = r["doc"]
                desc = doc.get("description", "")[:100]
                bm25_rank = str(r["bm25_rank"]) if r["bm25_rank"] is not None else "N/A"
                sem_rank = str(r["semantic_rank"]) if r["semantic_rank"] is not None else "N/A"
                print(f"{i}. {doc['title']}")
                if "rerank_score" in r:
                    print(f"  Re-rank Score: {r['rerank_score']:.3f}/10")
                elif "rerank_rank" in r:
                    print(f"  Re-rank Rank: {r['rerank_rank']}")
                elif "cross_encoder_score" in r:
                    print(f"  Cross Encoder Score: {r['cross_encoder_score']:.3f}")
                print(f"  RRF Score: {r['rrf']:.3f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                print(f"  {desc}...")
                print()
            if args.evaluate:
                print("\nEvaluating results with LLM...\n")
                formatted_results = []
                for r in results[:args.limit]:
                    doc = r["doc"]
                    formatted_results.append(f"{doc['title']}: {doc.get('description', '')[:200]}")
                eval_prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
                try:
                    raw = _llm_query(eval_prompt)
                    scores = json.loads(raw.strip())
                    print("Evaluation Report:")
                    for i, (r, score) in enumerate(zip(results[:args.limit], scores), 1):
                        print(f"{i}. {r['doc']['title']}: {score}/3")
                except (json.JSONDecodeError, RuntimeError) as e:
                    print(f"Evaluation failed: {e}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
