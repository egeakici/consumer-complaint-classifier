import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


class LogisticRegressionCustom:
    def __init__(self, learning_rate: float = 0.1, max_iter: int = 50,
                 batch_size: int = 512, C: float = 10.0, random_state: int = 42,
                 lr_decay: float = 0.95):
        self.learning_rate = learning_rate
        self.max_iter      = max_iter
        self.batch_size    = batch_size
        self.C             = C
        self.random_state  = random_state
        self.lr_decay      = lr_decay
        self.weights       = None
        self.bias          = 0.0
        self.classes_      = None
        self.loss_history  = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _list_of_dicts_to_csr(docs: list, n_features: int) -> csr_matrix:
        rows, cols, data = [], [], []
        for i, doc in enumerate(docs):
            for feat_idx, val in doc.items():
                rows.append(i)
                cols.append(feat_idx)
                data.append(val)
        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(docs), n_features),
            dtype=np.float64
        )

    def _encode_labels(self, y: list) -> np.ndarray:
        return np.array([1.0 if lbl == self.classes_[1] else 0.0 for lbl in y])

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        bce = -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )
        l2  = (1 / (2 * self.C)) * np.sum(self.weights ** 2)
        return float(bce + l2)

    def fit(self, X_sparse: list, y: list) -> "LogisticRegressionCustom":
        np.random.seed(self.random_state)
        self.classes_ = sorted(set(y))

        n_features = max(max(doc.keys()) for doc in X_sparse if doc) + 1

        print(f"\n      Converting to sparse matrix ({len(X_sparse):,} documents)...")
        X_csr = self._list_of_dicts_to_csr(X_sparse, n_features)
        y_enc = self._encode_labels(y)
        print(f"      Done. Matrix: {X_csr.shape[0]:,} x {X_csr.shape[1]:,}")

        self.weights = np.random.randn(n_features).astype(np.float64) * 0.01
        self.bias    = 0.0

        n_samples  = X_csr.shape[0]
        indices    = np.arange(n_samples)
        current_lr = self.learning_rate

        print(f"\n      Training: {self.max_iter} epochs | "
              f"batch_size={self.batch_size} | lr={self.learning_rate} | "
              f"C={self.C} | lr_decay={self.lr_decay}")

        with tqdm(total=self.max_iter, desc="      Epoch", unit="epoch",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}") as pbar:

            for epoch in range(self.max_iter):
                np.random.shuffle(indices)
                epoch_losses = []

                for start in range(0, n_samples, self.batch_size):
                    batch_idx = indices[start: start + self.batch_size]
                    X_batch   = X_csr[batch_idx]
                    y_batch   = y_enc[batch_idx]
                    m         = len(y_batch)

                    z      = X_batch @ self.weights + self.bias
                    y_pred = self._sigmoid(z)

                    epoch_losses.append(self._compute_loss(y_batch, y_pred))

                    error  = y_pred - y_batch
                    grad_w = (np.asarray(X_batch.T @ error).ravel() / m
                              + self.weights / self.C)
                    grad_b = float(np.mean(error))

                    self.weights -= current_lr * grad_w
                    self.bias    -= current_lr * grad_b

                avg_loss = float(np.mean(epoch_losses))
                self.loss_history.append(avg_loss)
                pbar.set_postfix_str(f"{avg_loss:.6f}  lr={current_lr:.5f}")
                pbar.update(1)

                current_lr *= self.lr_decay

        print("\n      Training complete.")
        return self

    def predict_proba(self, X_sparse: list) -> np.ndarray:
        n_features = len(self.weights)
        X_csr      = self._list_of_dicts_to_csr(X_sparse, n_features)
        z          = X_csr @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X_sparse: list) -> list:
        proba    = self.predict_proba(X_sparse)
        pred_idx = (proba >= 0.5).astype(int)
        return [self.classes_[i] for i in pred_idx]
