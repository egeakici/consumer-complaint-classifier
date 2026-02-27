import re
import math
from collections import defaultdict


class TfidfVectorizer:
    def __init__(self, min_df: int = 1, ngram_range: tuple = (1, 1)):
        self.min_df      = min_df
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_        = []

    def _tokenize(self, text: str) -> list:
        return re.findall(r"[a-z]+", text.lower())

    def _ngrams(self, tokens: list) -> list:
        min_n, max_n = self.ngram_range
        result = []
        n_tok  = len(tokens)
        for n in range(min_n, max_n + 1):
            for i in range(n_tok - n + 1):
                result.append(" ".join(tokens[i: i + n]))
        return result

    def _doc_to_ngrams(self, text: str) -> list:
        return self._ngrams(self._tokenize(text))

    def fit(self, documents: list) -> "TfidfVectorizer":
        n_docs     = len(documents)
        df_counter = defaultdict(int)

        for doc in documents:
            for term in set(self._doc_to_ngrams(doc)):
                df_counter[term] += 1

        vocab = sorted(term for term, cnt in df_counter.items() if cnt >= self.min_df)
        self.vocabulary_ = {term: idx for idx, term in enumerate(vocab)}

        self.idf_ = [
            math.log((1 + n_docs) / (1 + df_counter[term])) + 1
            for term in vocab
        ]
        return self

    def transform(self, documents: list) -> list:
        result = []
        for doc in documents:
            ngrams = self._doc_to_ngrams(doc)
            total  = len(ngrams) if ngrams else 1

            tf_counts = defaultdict(int)
            for ng in ngrams:
                if ng in self.vocabulary_:
                    tf_counts[ng] += 1

            vec = {}
            for term, cnt in tf_counts.items():
                idx      = self.vocabulary_[term]
                tf       = cnt / total
                vec[idx] = tf * self.idf_[idx]

            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {k: v / norm for k, v in vec.items()}

            result.append(vec)
        return result

    def fit_transform(self, documents: list) -> list:
        self.fit(documents)
        return self.transform(documents)
