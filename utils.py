import json
from movie_search.config import MOVIES_FILE
from movie_search.models import Movie

__all__ = ["Movie", "load_movies"]


def load_movies(path: str = str(MOVIES_FILE)) -> list[Movie]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["movies"]
