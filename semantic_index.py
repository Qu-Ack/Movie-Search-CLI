import pickle
from sentence_transformers import SentenceTransformer
from utils import load_movies


class SemanticIndex:

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = {}

    def embed(self, text:str):
        return self.model.encode(str)

    def build(self):
        movies = load_movies("data/movies.json")
        for movie in movies:
            text = f"{movie["title"]} {movie["description"]}"
            self.embeddings[movie["id"]] = self.embed(text)

        self.save()

    def save(self):
        with open("cache/embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)

    def load(self):
        with open("cache/embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)
