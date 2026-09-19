import string
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_stemmer: Optional[PorterStemmer] = None
_stop_words: Optional[set[str]] = None


def _ensure_nltk_resources() -> None:
    """Ensure required NLTK tokenizers and corpus data are available."""
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def get_stemmer() -> PorterStemmer:
    global _stemmer
    if _stemmer is None:
        _stemmer = PorterStemmer()
    return _stemmer


def get_stop_words() -> set[str]:
    global _stop_words
    if _stop_words is None:
        _ensure_nltk_resources()
        try:
            _stop_words = set(stopwords.words("english"))
        except Exception:
            _stop_words = set()
    return _stop_words


def process_text(text: str) -> list[str]:
    """
    Tokenize, normalize, remove punctuation and stopwords, and stem tokens.
    """
    if not text:
        return []

    lower_text = text.lower()
    punctuation = string.punctuation
    cleaned_chars = [
        " " if char in punctuation else char
        for char in lower_text
    ]
    cleaned_text = "".join(cleaned_chars)

    _ensure_nltk_resources()
    tokens = word_tokenize(cleaned_text)

    stop_words = get_stop_words()
    stemmer = get_stemmer()

    meaningful_tokens = [
        token for token in tokens
        if token not in stop_words
    ]

    return [stemmer.stem(token) for token in meaningful_tokens]
