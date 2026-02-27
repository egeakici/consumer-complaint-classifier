import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

nltk.download("stopwords",                      quiet=True)
nltk.download("wordnet",                        quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt",                          quiet=True)
nltk.download("punkt_tab",                      quiet=True)

_STOPWORDS  = set(stopwords.words("english"))
_STOPWORDS.add("xxxx")
_STOPWORDS.add("xx")

_LEMMATIZER = WordNetLemmatizer()


def _get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess(text: str) -> str:
    text     = text.lower()
    text     = re.sub(r"http\S+|www\S+", " ", text)
    text     = re.sub(r"[^a-z\s]", " ", text)
    text     = re.sub(r"\s+", " ", text).strip()
    tokens   = [t for t in text.split() if t not in _STOPWORDS and len(t) > 2]
    pos_tags = pos_tag(tokens)
    tokens   = [
        _LEMMATIZER.lemmatize(token, _get_wordnet_pos(tag))
        for token, tag in pos_tags
    ]
    return " ".join(tokens)
