import argparse
import utils
from inverted_index import InvertedIndex 
from process_text import process_text

def search(query: str) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()
    query_tokens = process_text(query)
    scores: dict[int, float] = {}

    for term in query_tokens:
        for doc_id in inverted_index.get_documents(term):
            scores[doc_id] = scores.get(doc_id, 0.0) + inverted_index.get_tfidf(
                doc_id, term
            )

    ranked_doc_ids = sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], inverted_index.docmap[doc_id]["title"]),
    )

    if not ranked_doc_ids:
        print("No movies found.")
        return

    print("\nTop movie matches:\n")
    for position, doc_id in enumerate(ranked_doc_ids[:5], start=1):
        print(f"{position}. {inverted_index.docmap[doc_id]['title']}")
        print(f"{inverted_index.docmap[doc_id]["description"]}")


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
            search(args.query)

        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
