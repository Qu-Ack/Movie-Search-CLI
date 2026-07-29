import argparse
import utils
from inverted_index import InvertedIndex 
from process_text import process_text


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

    build_parser = subparsers.add_parser("build", help="build index for the movie database")

    args = parser.parse_args()
    match args.command:
        case "search":
            movies = utils.load_movies("data/movies.json")
            for i in range(5):
                processed_title = process_text(movies[i]["title"])
                print(processed_title)
        
        case "build":

            inverted_index = InvertedIndex()
            inverted_index.build()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
