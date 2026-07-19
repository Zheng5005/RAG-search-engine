import argparse

from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model is loaded correctly")

    embed_parser = subparsers.add_parser("embed", help="Generate embedding for a text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify-embeddings", help="Verify movie embeddings are built correctly")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify-embeddings":
            verify_embeddings()
        case "embed":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
