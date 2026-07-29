import utils
import math
import pickle
from process_text import process_text
from collections import Counter

class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.tf : dict[int, Counter] = {}
        self.docmap = {}

    def __add_document(self, doc_idx: int, text: list[str]):
        self.tf[doc_idx] = Counter(set(text))
        for word in set(text):
            if word not in self.index:
                self.index[word] = set() 
            self.index[word].add(doc_idx)

    def get_tf(self, doc_id: int, term: str):
        return self.tf[doc_id][term]

    def get_idf(self, term: str):
        total_match_count = len(self.index.get(term, set()))
        total_doc_count = len(self.docmap)

        return math.log((total_doc_count + 1) / (total_match_count + 1)) 

    def get_tfidf(self, doc_id: int, term: str):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf

    def get_documents(self, term: str):
        return sorted(list(self.index.get(term, set())))


    def build(self):
        movies = utils.load_movies("data/movies.json")
        for movie in movies:
            self.docmap[movie["id"]] = movie;
            text = f"{movie["title"]} {movie["description"]}"
            processed_text = process_text(text)
            self.__add_document(movie["id"], processed_text)

        self.save(self.index, "cache/index.pkl")
        self.save(self.docmap, "cache/docmap.pkl")
        self.save(self.tf, "cache/tf.pkl")

    def save(self, obj, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    
    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/tf.pkl", "rb") as f:
            self.tf = pickle.load(f)










