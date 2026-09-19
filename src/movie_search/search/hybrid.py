from typing import Optional

from movie_search.config import DEFAULT_ALPHA, DEFAULT_RETRIEVAL_LIMIT, DEFAULT_TOP_N
from movie_search.indexing.inverted_index import InvertedIndex
from movie_search.indexing.semantic_index import SemanticIndex
from movie_search.models import SearchResult
from movie_search.search.keyword import KeywordSearcher, normalize_scores
from movie_search.search.semantic import SemanticSearcher


class HybridSearcher:
    """
    Hybrid retriever that combines keyword (TF-IDF) and semantic (dense embedding) retrieval.
    Hybrid score = alpha * normalized_semantic + (1 - alpha) * normalized_keyword
    """

    def __init__(
        self,
        inverted_index: Optional[InvertedIndex] = None,
        semantic_index: Optional[SemanticIndex] = None,
    ):
        if inverted_index is None:
            self.inverted_index = InvertedIndex()
            self.inverted_index.load()
        else:
            self.inverted_index = inverted_index

        if semantic_index is None:
            self.semantic_index = SemanticIndex()
            self.semantic_index.load()
        else:
            self.semantic_index = semantic_index

        self.keyword_searcher = KeywordSearcher(self.inverted_index)
        self.semantic_searcher = SemanticSearcher(self.semantic_index, self.inverted_index)

    def score(
        self,
        query: str,
        alpha: float = DEFAULT_ALPHA,
        top_n: int = DEFAULT_RETRIEVAL_LIMIT,
    ) -> dict[int, float]:
        semantic_scores = self.semantic_searcher.score(query, top_k=top_n)
        tfidf_scores = self.keyword_searcher.score(query, top_k=top_n)

        normalized_tfidf = normalize_scores(tfidf_scores)
        normalized_semantic = normalize_scores(semantic_scores)

        candidates = set(tfidf_scores) | set(semantic_scores)

        hybrid_scores = {
            doc_id: (
                alpha * normalized_semantic.get(doc_id, 0.0)
                + (1.0 - alpha) * normalized_tfidf.get(doc_id, 0.0)
            )
            for doc_id in candidates
        }

        ranked_doc_ids = sorted(
            candidates,
            key=lambda doc_id: (
                -hybrid_scores[doc_id],
                self.inverted_index.docmap[doc_id]["title"],
            ),
        )
        return {doc_id: hybrid_scores[doc_id] for doc_id in ranked_doc_ids[:top_n]}

    def search(
        self,
        query: str,
        alpha: float = DEFAULT_ALPHA,
        top_n: int = DEFAULT_TOP_N,
        retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ) -> list[SearchResult]:
        semantic_scores = self.semantic_searcher.score(query, top_k=retrieval_limit)
        tfidf_scores = self.keyword_searcher.score(query, top_k=retrieval_limit)

        normalized_tfidf = normalize_scores(tfidf_scores)
        normalized_semantic = normalize_scores(semantic_scores)

        candidates = set(tfidf_scores) | set(semantic_scores)

        hybrid_scores = {
            doc_id: (
                alpha * normalized_semantic.get(doc_id, 0.0)
                + (1.0 - alpha) * normalized_tfidf.get(doc_id, 0.0)
            )
            for doc_id in candidates
        }

        ranked_doc_ids = sorted(
            candidates,
            key=lambda doc_id: (
                -hybrid_scores[doc_id],
                self.inverted_index.docmap[doc_id]["title"],
            ),
        )

        results: list[SearchResult] = []
        for doc_id in ranked_doc_ids[:top_n]:
            movie = self.inverted_index.docmap[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=movie["title"],
                    description=movie["description"],
                    score=hybrid_scores[doc_id],
                    keyword_score=tfidf_scores.get(doc_id, 0.0),
                    semantic_score=semantic_scores.get(doc_id, 0.0),
                )
            )
        return results
