import os

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR         = os.path.join(BASE_DIR, "..", "data", "processed")

INPUT_CSV        = os.path.join(DATA_DIR, "dataset_naive_bayes.csv")
MODEL_PATH       = os.path.join(SAVED_MODELS_DIR, "model.pkl")
VECTORIZER_PATH  = os.path.join(SAVED_MODELS_DIR, "vectorizer.pkl")
X_TEST_PATH      = os.path.join(SAVED_MODELS_DIR, "x_test.pkl")
Y_TEST_PATH      = os.path.join(SAVED_MODELS_DIR, "y_test.pkl")

TEST_SIZE    = 0.2
RANDOM_STATE = 42

MIN_DF       = 5
NGRAM_RANGE  = (1, 2)

ALPHA        = 1.0
NORM         = True

FLASK_HOST   = "0.0.0.0"
FLASK_PORT   = 5000
