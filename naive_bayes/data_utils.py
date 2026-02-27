import random
import pandas as pd
from collections import defaultdict


def load_data(csv_path: str):
    print("[1/5] Loading data...")
    df = pd.read_csv(csv_path)
    X  = list(df["Consumer complaint narrative"].astype(str))
    y  = list(df["Consumer disputed?"])
    print(f"      Total rows : {len(df):,}")
    print(f"      Yes        : {sum(1 for lbl in y if lbl == 'Yes'):,}")
    print(f"      No         : {sum(1 for lbl in y if lbl == 'No'):,}")
    return X, y


def _stratified_split(indices: list, labels: list,
                      test_size: float, random_state: int) -> tuple:
    rng = random.Random(random_state)

    class_indices = defaultdict(list)
    for idx, lbl in zip(indices, labels):
        class_indices[lbl].append(idx)

    train_idx, test_idx = [], []
    for lbl, idxs in class_indices.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def split_data(X: list, y: list, test_size: float, random_state: int) -> tuple:
    print("\n[2/5] Splitting data (80% train / 20% test)...")
    indices = list(range(len(X)))

    train_idx, test_idx = _stratified_split(indices, y, test_size, random_state)

    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test  = [y[i] for i in test_idx]

    yes_train = sum(1 for lbl in y_train if lbl == "Yes") / len(y_train)
    yes_test  = sum(1 for lbl in y_test  if lbl == "Yes") / len(y_test)

    print(f"      Train: {len(X_train):,} rows  |  Yes ratio: {yes_train:.3f}")
    print(f"      Test : {len(X_test):,} rows  |  Yes ratio: {yes_test:.3f}")
    return X_train, X_test, y_train, y_test
