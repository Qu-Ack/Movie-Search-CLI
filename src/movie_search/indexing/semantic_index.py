import json
from pathlib import Path
import pickle
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from movie_search.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDINGS_CACHE_FILE,
    MOVIES_FILE,
    ensure_cache_dir,
)
from movie_search.models import Movie


class SemanticIndex:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.embeddings: dict[int, list[float]] = {}
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="movies",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    def _movie_text(self, movie: Movie) -> str:
        return f"{movie['title']} {movie['description']}"

    def is_built(self) -> bool:
        return EMBEDDINGS_CACHE_FILE.exists()

    def build(self, movies_path: Optional[str | Path] = None) -> None:
        path = Path(movies_path) if movies_path else MOVIES_FILE
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        movies: list[Movie] = data["movies"]

        texts = [self._movie_text(movie) for movie in movies]
        embeddings = self.embed_many(texts)
        self.embeddings = {
            movie["id"]: embedding for movie, embedding in zip(movies, embeddings)
        }

        self.save()
        self._build_collection(movies)

    def save(self) -> None:
        ensure_cache_dir()
        with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(self.embeddings, f)

    def load(self, movies_path: Optional[str | Path] = None) -> None:
        if not self.is_built():
            raise FileNotFoundError(
                f"Embeddings cache file not found at {EMBEDDINGS_CACHE_FILE}. "
                f"Please run 'build' first."
            )

        with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
            self.embeddings = pickle.load(f)

        path = Path(movies_path) if movies_path else MOVIES_FILE
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        movies: list[Movie] = data["movies"]
        self._build_collection(movies)

    def _build_collection(self, movies: list[Movie]) -> None:
        movie_by_id = {movie["id"]: movie for movie in movies}
        ids = [str(doc_id) for doc_id in self.embeddings]
        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            embeddings=[self.embeddings[int(doc_id)] for doc_id in ids],
            documents=[self._movie_text(movie_by_id[int(doc_id)]) for doc_id in ids],
            metadatas=[
                {"movie_id": int(doc_id), "title": movie_by_id[int(doc_id)]["title"]}
                for doc_id in ids
            ],
        )

    def query(self, query_text: str, n_results: int = 50) -> dict[int, float]:
        if not self.embeddings:
            self.load()

        results = self.collection.query(
            query_embeddings=[self.embed(query_text)],
            n_results=min(n_results, len(self.embeddings)),
        )

        doc_ids = results.get("ids")
        distances = results.get("distances")

        if not doc_ids or not distances or not doc_ids[0]:
            return {}

        return {
            int(doc_id): max(0.0, 1.0 - distance)
            for doc_id, distance in zip(doc_ids[0], distances[0])
        }
