from typing import Optional

from movie_search.indexing.inverted_index import InvertedIndex
from movie_search.models import SearchResult
from movie_search.text import process_text


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    max_score = max(scores.values())
    if max_score <= 0:
        return {doc_id: 0.0 for doc_id in scores}

    return {doc_id: score / max_score for doc_id, score in scores.items()}


class KeywordSearcher:
    def __init__(self, inverted_index: Optional[InvertedIndex] = None):
        if inverted_index is None:
            self.inverted_index = InvertedIndex()
            self.inverted_index.load()
        else:
            self.inverted_index = inverted_index

    def score(self, query: str, top_k: int = 50) -> dict[int, float]:
        query_tokens = process_text(query)
        scores: dict[int, float] = {}

        for term in query_tokens:
            for doc_id in self.inverted_index.get_documents(term):
                scores[doc_id] = scores.get(doc_id, 0.0) + self.inverted_index.get_tfidf(
                    doc_id, term
                )

        ranked_doc_ids = sorted(
            scores,
            key=lambda doc_id: (-scores[doc_id], self.inverted_index.docmap[doc_id]["title"]),
        )
        return {doc_id: scores[doc_id] for doc_id in ranked_doc_ids[:top_k]}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        scores = self.score(query, top_k=top_k)
        results: list[SearchResult] = []
        for doc_id, raw_score in scores.items():
            movie = self.inverted_index.docmap[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=movie["title"],
                    description=movie["description"],
                    score=raw_score,
                    keyword_score=raw_score,
                )
            )
        return results
