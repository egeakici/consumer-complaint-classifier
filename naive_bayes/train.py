import os
import pickle

import config
from data_utils  import load_data, split_data
from model_utils import build_vectorizer, train_model, save_artifacts


def main():
    X, y = load_data(config.INPUT_CSV)

    X_train, X_test, y_train, y_test = split_data(
        X, y, config.TEST_SIZE, config.RANDOM_STATE
    )

    vectorizer, X_train_tfidf = build_vectorizer(
        X_train, config.MIN_DF, config.NGRAM_RANGE
    )
    X_test_tfidf = vectorizer.transform(X_test)

    model = train_model(X_train_tfidf, y_train, config.ALPHA, config.NORM)

    save_artifacts(model, vectorizer, config.MODEL_PATH, config.VECTORIZER_PATH)

    for path in (config.X_TEST_PATH, config.Y_TEST_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(config.X_TEST_PATH, "wb") as f:
        pickle.dump(X_test_tfidf, f)
    with open(config.Y_TEST_PATH, "wb") as f:
        pickle.dump(y_test, f)

    print(f"      X_test → {config.X_TEST_PATH}")
    print(f"      y_test → {config.Y_TEST_PATH}")
    print("\nTraining complete. Run test.py to evaluate.")


if __name__ == "__main__":
    main()
