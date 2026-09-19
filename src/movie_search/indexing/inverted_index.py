from collections import Counter
import json
import math
from pathlib import Path
import pickle
from typing import Optional

from movie_search.config import (
    DOCMAP_CACHE_FILE,
    INDEX_CACHE_FILE,
    MOVIES_FILE,
    TF_CACHE_FILE,
    ensure_cache_dir,
)
from movie_search.models import Movie
from movie_search.text import process_text


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.tf: dict[int, Counter] = {}
        self.docmap: dict[int, Movie] = {}

    def _add_document(self, doc_idx: int, text_tokens: list[str]):
        self.tf[doc_idx] = Counter(set(text_tokens))
        for word in set(text_tokens):
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_idx)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.tf.get(doc_id, {}).get(term, 0)

    def get_idf(self, term: str) -> float:
        total_match_count = len(self.index.get(term, set()))
        total_doc_count = len(self.docmap)
        if total_doc_count == 0:
            return 0.0
        return math.log((total_doc_count + 1) / (total_match_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_documents(self, term: str) -> list[int]:
        return sorted(list(self.index.get(term, set())))

    def is_built(self) -> bool:
        return (
            INDEX_CACHE_FILE.exists()
            and DOCMAP_CACHE_FILE.exists()
            and TF_CACHE_FILE.exists()
        )

    def build(self, movies_path: Optional[str | Path] = None) -> None:
        path = Path(movies_path) if movies_path else MOVIES_FILE
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        movies: list[Movie] = data["movies"]

        self.index.clear()
        self.tf.clear()
        self.docmap.clear()

        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            combined_text = f"{movie['title']} {movie['description']}"
            tokens = process_text(combined_text)
            self._add_document(doc_id, tokens)

        ensure_cache_dir()
        self.save(self.index, INDEX_CACHE_FILE)
        self.save(self.docmap, DOCMAP_CACHE_FILE)
        self.save(self.tf, TF_CACHE_FILE)

    def save(self, obj, file_path: Path) -> None:
        ensure_cache_dir()
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    def load(self) -> None:
        if not self.is_built():
            raise FileNotFoundError(
                f"Inverted index cache files not found in {INDEX_CACHE_FILE.parent}. "
                f"Please run 'build' first."
            )

        with open(INDEX_CACHE_FILE, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_CACHE_FILE, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TF_CACHE_FILE, "rb") as f:
            self.tf = pickle.load(f)
