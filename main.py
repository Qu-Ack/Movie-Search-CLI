import argparse


def search(query: str) -> None:
    print(f"searching for: {query}")

def main():
    parser = argparse.ArgumentParser(
        prog="Movie Search",
        description="A cli tool that let's you enter natural language queries and search for movies."
    )

    subparsers = parser.add_subparsers(dest="command", help="availaible sub commands")
    search_parser = subparsers.add_parser("search", help="search a query across the movie db")
    search_parser.add_argument("-q", "--query", type=str, required=True, help="The query you want to search")

    args = parser.parse_args()
    match args.command:
        case "search":
            search(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
