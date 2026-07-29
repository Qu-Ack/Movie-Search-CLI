import json
from typing import TypedDict

class Movie(TypedDict):
    id: int
    title: str
    description: str


def load_movies(path: str) -> list[Movie]:
    with open(path, "r") as file:
        data = json.load(file)
    return data["movies"]
