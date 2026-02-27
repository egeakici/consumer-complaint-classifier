"""
End-to-end pipeline for Logistic Regression model.

Steps:
    1. Filter raw CFPB data                (data_pipeline/data_filter.py)
    2. Preprocess for Logistic Regression  (data_pipeline/preprocess_logistic_regression.py)
    3. Train Logistic Regression           (logistic_regression/train.py)
    4. Evaluate on test set                (logistic_regression/test.py)

Usage:
    python pipeline_logistic_regression.py                  # run all steps
    python pipeline_logistic_regression.py --start-from 3  # skip data steps, start from training
"""

import argparse
import importlib.util
import sys
import os


def run_step(step_name: str, module_path: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    spec   = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "main"):
        module.main()


def main():
    parser = argparse.ArgumentParser(description="Logistic Regression end-to-end pipeline")
    parser.add_argument(
        "--start-from", type=int, default=1, choices=[1, 2, 3, 4],
        help="Start from step N (1=filter, 2=preprocess, 3=train, 4=evaluate)"
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    steps = [
        (1, "Data Filtering",                  os.path.join(base, "data_pipeline", "data_filter.py")),
        (2, "Preprocessing (Lemmatization)",   os.path.join(base, "data_pipeline", "preprocess_logistic_regression.py")),
        (3, "Training (Logistic Regression)",  os.path.join(base, "logistic_regression", "train.py")),
        (4, "Evaluation",                      os.path.join(base, "logistic_regression", "test.py")),
    ]

    sys.path.insert(0, os.path.join(base, "logistic_regression"))

    for step_num, step_name, module_path in steps:
        if step_num < args.start_from:
            print(f"Skipping step {step_num}: {step_name}")
            continue
        run_step(step_name, module_path)

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}")
    print("\nTo serve the model: cd logistic_regression && python app.py")


if __name__ == "__main__":
    main()
