import argparse
import json
from pathlib import Path

from lib.hybrid_search import HybridSearch
from lib.semantic_search import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    base = Path(__file__).resolve().parent.parent
    with open(base / "data" / "golden_dataset.json") as f:
        golden = json.load(f)

    documents = load_movies()
    hs = HybridSearch(documents)

    print(f"k={limit}\n")

    for tc in golden["test_cases"]:
        query = tc["query"]
        relevant = set(tc["relevant_docs"])
        results = hs.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [r["doc"]["title"] for r in results]
        relevant_retrieved = sum(1 for t in retrieved_titles if t in relevant)
        precision = relevant_retrieved / limit if limit > 0 else 0.0
        recall = relevant_retrieved / len(relevant) if relevant else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(sorted(relevant))}")
        print()


if __name__ == "__main__":
    main()
