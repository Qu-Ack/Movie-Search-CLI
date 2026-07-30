import pickle
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from utils import load_movies, Movie


class SemanticIndex:

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = {}
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="movies",
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, text: str):
        return self.model.encode(text).tolist()

    def embed_many(self, texts: list[str]):
        return self.model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    def _movie_text(self, movie: Movie) -> str:
        return f'{movie["title"]} {movie["description"]}'

    def build(self):
        movies = load_movies("data/movies.json")
        texts = [self._movie_text(movie) for movie in movies]
        embeddings = self.embed_many(texts)
        self.embeddings = {
            movie["id"]: embedding for movie, embedding in zip(movies, embeddings)
        }

        self.save()
        self._build_collection(movies)

    def save(self):
        with open("cache/embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)

    def load(self):
        with open("cache/embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)
        movies = load_movies("data/movies.json")
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

    def query(self, query: str, n_results: int = 50) -> dict[int, float]:
        if not self.embeddings:
            self.load()

        results = self.collection.query(
            query_embeddings=[self.embed(query)],
            n_results=min(n_results, len(self.embeddings)),
        )

        doc_ids = results.get("ids")
        distances = results.get("distances")

        if not doc_ids or not distances:
            return {}

        return {
            int(doc_id): max(0.0, 1.0 - distance)
            for doc_id, distance in zip(doc_ids[0], distances[0])
        }
