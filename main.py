import argparse
import utils
from inverted_index import InvertedIndex 
from process_text import process_text
from semantic_index import SemanticIndex


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    max_score = max(scores.values())
    if max_score <= 0:
        return {doc_id: 0.0 for doc_id in scores}

    return {doc_id: score / max_score for doc_id, score in scores.items()}


def tfidf_search_scores(
    inverted_index: InvertedIndex, query: str, top_k: int = 50
) -> dict[int, float]:
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
    return {doc_id: scores[doc_id] for doc_id in ranked_doc_ids[:top_k]}

def search(query: str) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()
    scores = tfidf_search_scores(inverted_index, query, top_k=5)
    ranked_doc_ids = list(scores)

    if not ranked_doc_ids:
        print("No movies found.")
        return

    print("\nTop movie matches:\n")
    for position, doc_id in enumerate(ranked_doc_ids[:5], start=1):
        print(f"{position}. {inverted_index.docmap[doc_id]['title']}")
        print(inverted_index.docmap[doc_id]["description"])


def hybrid_search(query: str, alpha: float = 0.7, top_n: int = 5) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()
    semantic_index = SemanticIndex()
    semantic_scores = semantic_index.query(query, n_results=50)

    tfidf_scores = tfidf_search_scores(inverted_index, query, top_k=50)
    normalized_tfidf_scores = normalize_scores(tfidf_scores)
    normalized_semantic_scores = normalize_scores(semantic_scores)
    candidates = set(tfidf_scores) | set(semantic_scores)

    hybrid_scores = {
        doc_id: (
            alpha * normalized_semantic_scores.get(doc_id, 0.0)
            + (1 - alpha) * normalized_tfidf_scores.get(doc_id, 0.0)
        )
        for doc_id in candidates
    }

    ranked_doc_ids = sorted(
        candidates,
        key=lambda doc_id: (
            -hybrid_scores[doc_id],
            inverted_index.docmap[doc_id]["title"],
        ),
    )

    if not ranked_doc_ids:
        print("No movies found.")
        return

    print("\nTop hybrid movie matches:\n")
    for position, doc_id in enumerate(ranked_doc_ids[:top_n], start=1):
        movie = inverted_index.docmap[doc_id]
        print(f"{position}. {movie['title']} ({hybrid_scores[doc_id]:.3f})")
        print(movie["description"])


def main():
    parser = argparse.ArgumentParser(
        prog="Movie Search",
        description="A cli tool that let's you enter natural language queries and search for movies."
    )

    subparsers = parser.add_subparsers(dest="command", help="availaible sub commands")

    search_parser = subparsers.add_parser("search", help="search a query across the movie db")
    search_parser.add_argument("-q", "--query", type=str, required=True, help="The query you want to search")

    build_parser = subparsers.add_parser("build", help="build index for the movie database")

    hybrid_parser = subparsers.add_parser(
        "hybrid_search", help="search with TF-IDF and semantic similarity"
    )
    hybrid_parser.add_argument("-q", "--query", type=str, required=True, help="The query you want to search")
    hybrid_parser.add_argument("--alpha", type=float, default=0.7, help="Semantic score weight from 0 to 1")
    hybrid_parser.add_argument("--top-n", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()
    match args.command:
        case "search":
            search(args.query)

        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            semantic_index = SemanticIndex()
            semantic_index.build()

        case "hybrid_search":
            hybrid_search(args.query, alpha=args.alpha, top_n=args.top_n)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
