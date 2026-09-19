from typing import Optional

from movie_search.indexing.inverted_index import InvertedIndex
from movie_search.indexing.semantic_index import SemanticIndex
from movie_search.models import SearchResult


class SemanticSearcher:
    def __init__(
        self,
        semantic_index: Optional[SemanticIndex] = None,
        inverted_index: Optional[InvertedIndex] = None,
    ):
        if semantic_index is None:
            self.semantic_index = SemanticIndex()
            self.semantic_index.load()
        else:
            self.semantic_index = semantic_index

        if inverted_index is None:
            self.inverted_index = InvertedIndex()
            self.inverted_index.load()
        else:
            self.inverted_index = inverted_index

    def score(self, query: str, top_k: int = 50) -> dict[int, float]:
        return self.semantic_index.query(query, n_results=top_k)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        scores = self.score(query, top_k=top_k)
        results: list[SearchResult] = []
        for doc_id, raw_score in scores.items():
            movie = self.inverted_index.docmap.get(doc_id)
            if not movie:
                continue
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=movie["title"],
                    description=movie["description"],
                    score=raw_score,
                    semantic_score=raw_score,
                )
            )
        return results
