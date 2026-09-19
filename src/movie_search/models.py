from dataclasses import dataclass, field
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


@dataclass
class SearchResult:
    doc_id: int
    title: str
    description: str
    score: float
    keyword_score: float = 0.0
    semantic_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "description": self.description,
            "score": self.score,
            "keyword_score": self.keyword_score,
            "semantic_score": self.semantic_score,
        }


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    model: str = ""
