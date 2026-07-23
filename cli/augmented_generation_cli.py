import argparse

from lib.hybrid_search import HybridSearch, _llm_query
from lib.semantic_search import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser(
        "summarize", help="Summarize search results for a query"
    )
    sum_parser.add_argument("query", type=str, help="Search query to summarize")
    sum_parser.add_argument("--limit", type=int, default=5, help="Number of results to summarize")

    cite_parser = subparsers.add_parser(
        "citations", help="Answer a query with cited sources"
    )
    cite_parser.add_argument("query", type=str, help="Search query for citations")
    cite_parser.add_argument("--limit", type=int, default=5, help="Number of results to cite")

    qa_parser = subparsers.add_parser(
        "question", help="Answer a conversational question about movies"
    )
    qa_parser.add_argument("question", type=str, help="Question to answer")
    qa_parser.add_argument("--limit", type=int, default=5, help="Number of results to consider")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=5)

            docs = "\n".join(
                f"- {r['doc']['title']}: {r['doc'].get('description', '')[:200]}"
                for r in results
            )

            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

            answer = _llm_query(prompt)

            print("Search Results:")
            for r in results:
                print(f"- {r['doc']['title']}")
            print()
            print("RAG Response:")
            print(answer)
        case "summarize":
            query = args.query
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            docs = "\n".join(
                f"- {r['doc']['title']}: {r['doc'].get('description', '')[:200]}"
                for r in results
            )

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""

            summary = _llm_query(prompt)

            print("Search Results:")
            for r in results:
                print(f"  - {r['doc']['title']}")
            print()
            print("LLM Summary:")
            print(summary)
        case "citations":
            query = args.query
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, k=60, limit=args.limit)

            docs = "\n".join(
                f"- {r['doc']['title']}: {r['doc'].get('description', '')[:200]}"
                for r in results
            )

            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

            answer = _llm_query(prompt)

            print("Search Results:")
            for r in results:
                print(f"  - {r['doc']['title']}")
            print()
            print("LLM Answer:")
            print(answer)
        case "question":
            question = args.question
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(question, k=60, limit=args.limit)

            context = "\n".join(
                f"- {r['doc']['title']}: {r['doc'].get('description', '')[:200]}"
                for r in results
            )

            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

            answer = _llm_query(prompt)

            print("Search Results:")
            for r in results:
                print(f"  - {r['doc']['title']}")
            print()
            print("Answer:")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
