import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def process_text(text: str) -> list[str]:
    lower_text = text.lower()
    punctuation = string.punctuation
    without_punctuation = []

    for char in lower_text:
        if char in punctuation:
            without_punctuation.append(" ")
        else:
            without_punctuation.append(char)

    cleaned_text = "".join(without_punctuation)

    tokens = word_tokenize(cleaned_text)

    meaningful_tokens = []

    for token in tokens:
        if token not in stop_words:
            meaningful_tokens.append(token)

    stemmed_tokens = []

    for token in meaningful_tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens


