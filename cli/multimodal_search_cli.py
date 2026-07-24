import argparse

from lib.multimodal_search import image_search, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify CLIP image embedding works"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to an image file")

    search_parser = subparsers.add_parser(
        "image_search", help="Search movies by image"
    )
    search_parser.add_argument("image_path", type=str, help="Path to an image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search(args.image_path)
            for i, r in enumerate(results, 1):
                desc = r["description"][:100]
                print(f"{i}. {r['title']} (similarity: {r['similarity']:.3f})")
                print(f"   {desc}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
