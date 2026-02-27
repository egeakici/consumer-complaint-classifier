import pickle

import config
from model_utils import evaluate_model


def main():
    print("Loading model...")
    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"      Model      <- {config.MODEL_PATH}")

    print("Loading test data...")
    with open(config.X_TEST_PATH, "rb") as f:
        X_test_tfidf = pickle.load(f)
    with open(config.Y_TEST_PATH, "rb") as f:
        y_test = pickle.load(f)
    print(f"      Test rows  : {len(y_test):,}")

    evaluate_model(model, X_test_tfidf, y_test)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
