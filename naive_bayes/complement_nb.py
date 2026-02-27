import math
from collections import defaultdict


class ComplementNB:
    def __init__(self, alpha: float = 1.0, norm: bool = True):
        self.alpha            = alpha
        self.norm             = norm
        self.classes_         = []
        self.log_weights_     = {}
        self.class_log_prior_ = {}
        self.n_features_      = 0

    def fit(self, X_sparse: list, y) -> "ComplementNB":
        labels           = list(y)
        self.classes_    = sorted(set(labels))
        n_docs           = len(labels)
        self.n_features_ = self._compute_n_features(X_sparse)

        class_counts = defaultdict(int)
        for lbl in labels:
            class_counts[lbl] += 1

        for c in self.classes_:
            self.class_log_prior_[c] = math.log(class_counts[c] / n_docs)

        total_feat_sum = defaultdict(float)
        class_feat_sum = {c: defaultdict(float) for c in self.classes_}

        for doc, lbl in zip(X_sparse, labels):
            for feat_idx, val in doc.items():
                total_feat_sum[feat_idx]      += val
                class_feat_sum[lbl][feat_idx] += val

        for c in self.classes_:
            comp_sum = {
                feat_idx: total_feat_sum[feat_idx] - class_feat_sum[c].get(feat_idx, 0.0)
                for feat_idx in total_feat_sum
            }

            total_comp = sum(comp_sum.values())
            denom      = self.alpha * self.n_features_ + total_comp

            weights = {
                feat_idx: math.log((self.alpha + comp_val) / denom)
                for feat_idx, comp_val in comp_sum.items()
            }

            if self.norm:
                norm_val = math.sqrt(sum(w * w for w in weights.values()))
                if norm_val > 0:
                    weights = {k: v / norm_val for k, v in weights.items()}

            self.log_weights_[c] = weights

        return self

    def predict(self, X_sparse: list) -> list:
        predictions = []
        for doc in X_sparse:
            scores = {}
            for c in self.classes_:
                weights  = self.log_weights_[c]
                scores[c] = sum(
                    val * weights.get(feat_idx, 0.0)
                    for feat_idx, val in doc.items()
                )
            predictions.append(min(scores, key=lambda c: scores[c]))
        return predictions

    @staticmethod
    def _compute_n_features(X_sparse: list) -> int:
        max_idx = -1
        for doc in X_sparse:
            if doc:
                m = max(doc.keys())
                if m > max_idx:
                    max_idx = m
        return max_idx + 1
