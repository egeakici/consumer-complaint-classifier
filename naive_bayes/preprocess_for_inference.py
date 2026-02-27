import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_STOPWORDS.add("xxxx")
_STOPWORDS.add("xx")

_STEMMER = PorterStemmer()


def preprocess(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"http\S+|www\S+", " ", text)
    text   = re.sub(r"[^a-z\s]", " ", text)
    text   = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [
        _STEMMER.stem(token)
        for token in tokens
        if token not in _STOPWORDS and len(token) > 2
    ]
    return " ".join(tokens)
