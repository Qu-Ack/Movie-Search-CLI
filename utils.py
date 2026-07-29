import json

def load_movies(path: str):
    with open(path, "r") as file:
        data = json.load(file)
        print(data)



