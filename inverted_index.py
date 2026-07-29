import utils
import pickle
from process_text import process_text

class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_idx: int, text: list[str]):

        if doc_idx in self.docmap:
            print("index already exists in docmap")
            return

        self.docmap[doc_idx] = text

        for word in text:
            if word not in self.index:
                self.index[word] = []
            
            self.index[word].append(doc_idx)

    def get_documents(self, term: str):
        if term not in self.index:
            print("term not found in index")
            return

        return sorted(self.index[term])

    def build(self):
        movies = utils.load_movies("data/movies.json")
        for movie in movies:
            text = f"{movie["title"]} {movie["description"]}"
            processed_text = process_text(text)
            self.__add_document(movie["id"], processed_text)

        self.save(self.index, "cache/index.pkl")
        self.save(self.docmap, "cache/docmap.pkl")
        
    def save(self, obj, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    
    def load(self, obj, file_path):
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
