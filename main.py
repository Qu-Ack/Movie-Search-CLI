import argparse
import json
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
    ranked_doc_ids = keyword_search_doc_ids(inverted_index, query, top_k=5)

    if not ranked_doc_ids:
        print("No movies found.")
        return

    print("\nTop movie matches:\n")
    for position, doc_id in enumerate(ranked_doc_ids[:5], start=1):
        print(f"{position}. {inverted_index.docmap[doc_id]['title']}")
        print(inverted_index.docmap[doc_id]["description"])


def keyword_search_doc_ids(
    inverted_index: InvertedIndex, query: str, top_k: int = 50
) -> list[int]:
    return list(tfidf_search_scores(inverted_index, query, top_k=top_k))


def hybrid_search_scores(
    inverted_index: InvertedIndex,
    semantic_index: SemanticIndex,
    query: str,
    alpha: float = 0.7,
    top_n: int = 50,
) -> dict[int, float]:
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
    return {doc_id: hybrid_scores[doc_id] for doc_id in ranked_doc_ids[:top_n]}


def hybrid_search_doc_ids(
    inverted_index: InvertedIndex,
    semantic_index: SemanticIndex,
    query: str,
    alpha: float = 0.7,
    top_n: int = 50,
) -> list[int]:
    return list(
        hybrid_search_scores(
            inverted_index,
            semantic_index,
            query,
            alpha=alpha,
            top_n=top_n,
        )
    )


def hybrid_search(query: str, alpha: float = 0.7, top_n: int = 5) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()
    semantic_index = SemanticIndex()
    scores = hybrid_search_scores(
        inverted_index, semantic_index, query, alpha=alpha, top_n=top_n
    )
    ranked_doc_ids = list(scores)

    if not ranked_doc_ids:
        print("No movies found.")
        return

    print("\nTop hybrid movie matches:\n")
    for position, doc_id in enumerate(ranked_doc_ids[:top_n], start=1):
        movie = inverted_index.docmap[doc_id]
        print(f"{position}. {movie['title']} ({scores[doc_id]:.3f})")
        print(movie["description"])


def calculate_metrics(
    retrieved: list[str], relevant: list[str]
) -> tuple[float, float, float]:
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    true_positives = len(retrieved_set & relevant_set)

    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


def evaluate(
    dataset_path: str = "data/golden_dataset.json",
    top_n: int = 5,
    alpha: float = 0.7,
) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    inverted_index = InvertedIndex()
    inverted_index.load()
    semantic_index = SemanticIndex()

    rows = []
    for test_case in dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        keyword_titles = [
            inverted_index.docmap[doc_id]["title"]
            for doc_id in keyword_search_doc_ids(inverted_index, query, top_k=top_n)
        ]
        hybrid_titles = [
            inverted_index.docmap[doc_id]["title"]
            for doc_id in hybrid_search_doc_ids(
                inverted_index,
                semantic_index,
                query,
                alpha=alpha,
                top_n=top_n,
            )
        ]

        rows.append(
            ("Keyword", query, *calculate_metrics(keyword_titles, relevant_docs))
        )
        rows.append(("Hybrid", query, *calculate_metrics(hybrid_titles, relevant_docs)))

    print(f"\nEvaluation metrics @ top {top_n}\n")
    print(f"{'Search':<10} {'Query':<42} {'Precision':>9} {'Recall':>8} {'F1':>8}")
    print("-" * 82)
    for search_type, query, precision, recall, f1 in rows:
        display_query = query if len(query) <= 42 else f"{query[:39]}..."
        print(
            f"{search_type:<10} {display_query:<42} "
            f"{precision:>9.3f} {recall:>8.3f} {f1:>8.3f}"
        )


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

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate keyword and hybrid search against the golden dataset"
    )
    evaluate_parser.add_argument(
        "--dataset", default="data/golden_dataset.json", help="Path to the golden dataset"
    )
    evaluate_parser.add_argument(
        "--top-n", type=int, default=5, help="Number of results to evaluate"
    )
    evaluate_parser.add_argument(
        "--alpha", type=float, default=0.7, help="Semantic score weight from 0 to 1"
    )

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

        case "evaluate":
            evaluate(dataset_path=args.dataset, top_n=args.top_n, alpha=args.alpha)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
