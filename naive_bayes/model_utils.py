import os
import pickle

from tfidf         import TfidfVectorizer
from complement_nb import ComplementNB
import metrics as m


def build_vectorizer(X_train: list, min_df: int, ngram_range: tuple):
    print("\n[3/5] Building vocabulary (train set only)...")
    vectorizer    = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    vocab_size  = len(vectorizer.vocabulary_)
    nnz         = sum(len(doc) for doc in X_train_tfidf)
    total_cells = len(X_train_tfidf) * vocab_size
    density     = nnz / total_cells * 100 if total_cells > 0 else 0

    print(f"      Vectorizer   : TF-IDF  |  N-gram: {ngram_range}")
    print(f"      Vocabulary   : {vocab_size:,} tokens")
    print(f"      Matrix shape : {len(X_train_tfidf):,} x {vocab_size:,}")
    print(f"      Density      : {density:.4f}%")
    return vectorizer, X_train_tfidf


def train_model(X_train_tfidf: list, y_train: list, alpha: float, norm: bool):
    print("\n[4/5] Training model...")
    model = ComplementNB(alpha=alpha, norm=norm)
    model.fit(X_train_tfidf, y_train)
    print(f"      Algorithm : ComplementNB  |  alpha={alpha}  |  norm={norm}")
    print("      Training complete.")
    return model


def evaluate_model(model, X_test_tfidf: list, y_test: list):
    print("\n[5/5] Evaluating model...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = m.accuracy_score(y_test, y_pred)
    f1_macro = m.f1_score(y_test, y_pred, average="macro")
    f1_yes   = m.f1_score(y_test, y_pred, average="binary", pos_label="Yes")
    f1_no    = m.f1_score(y_test, y_pred, average="binary", pos_label="No")
    cm       = m.confusion_matrix(y_test, y_pred, labels=["Yes", "No"])
    report   = m.classification_report(y_test, y_pred, target_names=["Yes", "No"])

    print("\n" + "=" * 50)
    print("     COMPLEMENT NAIVE BAYES — RESULTS")
    print("=" * 50)
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  F1 (macro) : {f1_macro:.4f}")
    print(f"  F1 (Yes)   : {f1_yes:.4f}")
    print(f"  F1 (No)    : {f1_no:.4f}")
    print("\n  Confusion Matrix (row=actual, col=predicted):")
    print(f"               Pred Yes    Pred No")
    print(f"  Actual Yes   {cm[0][0]:<11} {cm[0][1]}")
    print(f"  Actual No    {cm[1][0]:<11} {cm[1][1]}")
    print("\n  Classification Report:")
    print(report)
    print("=" * 50)


def save_artifacts(model, vectorizer, model_path: str, vectorizer_path: str):
    print("\nSaving model and vectorizer...")
    for path in (model_path, vectorizer_path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"      Model      → {model_path}")
    print(f"      Vectorizer → {vectorizer_path}")
