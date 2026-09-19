import json
from pathlib import Path
from typing import Optional, Union

from movie_search.config import DEFAULT_ALPHA, DEFAULT_TOP_N, GOLDEN_DATASET_FILE
from movie_search.indexing.inverted_index import InvertedIndex
from movie_search.indexing.semantic_index import SemanticIndex
from movie_search.search.hybrid import HybridSearcher
from movie_search.search.keyword import KeywordSearcher


def calculate_metrics(
    retrieved: list[str], relevant: list[str]
) -> tuple[float, float, float]:
    """Calculate Precision, Recall, and F1 score."""
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
    dataset_path: Optional[Union[str, Path]] = None,
    top_n: int = DEFAULT_TOP_N,
    alpha: float = DEFAULT_ALPHA,
) -> list[dict]:
    """
    Evaluate keyword and hybrid search against a golden evaluation dataset.
    """
    path = Path(dataset_path) if dataset_path else GOLDEN_DATASET_FILE
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    inverted_index = InvertedIndex()
    inverted_index.load()
    semantic_index = SemanticIndex()
    semantic_index.load()

    keyword_searcher = KeywordSearcher(inverted_index)
    hybrid_searcher = HybridSearcher(inverted_index, semantic_index)

    results = []
    print(f"\nEvaluation metrics @ top {top_n} (alpha={alpha})\n")
    print(f"{'Search':<10} {'Query':<42} {'Precision':>9} {'Recall':>8} {'F1':>8}")
    print("-" * 82)

    for test_case in dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        keyword_results = keyword_searcher.search(query, top_k=top_n)
        keyword_titles = [res.title for res in keyword_results]

        hybrid_results = hybrid_searcher.search(query, alpha=alpha, top_n=top_n)
        hybrid_titles = [res.title for res in hybrid_results]

        k_prec, k_rec, k_f1 = calculate_metrics(keyword_titles, relevant_docs)
        h_prec, h_rec, h_f1 = calculate_metrics(hybrid_titles, relevant_docs)

        results.append({
            "query": query,
            "keyword": {"precision": k_prec, "recall": k_rec, "f1": k_f1},
            "hybrid": {"precision": h_prec, "recall": h_rec, "f1": h_f1},
        })

        display_query = query if len(query) <= 42 else f"{query[:39]}..."
        print(f"{'Keyword':<10} {display_query:<42} {k_prec:>9.3f} {k_rec:>8.3f} {k_f1:>8.3f}")
        print(f"{'Hybrid':<10} {display_query:<42} {h_prec:>9.3f} {h_rec:>8.3f} {h_f1:>8.3f}")

    return results
