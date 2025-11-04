"""
run_optiforest_ad_creditcard.py
ADS Final Project – Default OptIForest Project
This script is designed to satisfy the ADS Final Project (Document 2) requirements:
1. Replication of original work:
   - Run OptIForest on the original "AD" dataset (data/ad.csv).
2. Construction/use of NEW data:
   - Use the Kaggle "Credit Card Fraud Detection" dataset (data/creditcard.csv)
     as NEW data.
   - Apply a clear, documented preprocessing pipeline:
       * find label column
       * drop Time column
       * scale features
3. Results on new data:
   - Evaluate OptIForest on the credit card dataset using the same metrics
     as the original: ROC AUC and PR AUC.
4. Reproducibility:
   - Save all metrics to results/ad_creditcard_metrics.json for inclusion
     in the LaTeX report and GitHub repo.
Usage (from repo root):
    python -m experiments.run_optiforest_ad_creditcard
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


# Import OptIForest from the existing project

try:
    # Most repos expose OptIForest in detectors/__init__.py
    from detectors import OptIForest
except ImportError:
    try:
        # Fallback if class is defined in a module
        from detectors.opt_iforest import OptIForest
    except ImportError as e:
        raise ImportError(
            "Could not import OptIForest. "
            "Check detectors/ and adjust the import line in "
            "experiments/optiforest_ad_creditcard.py if needed."
        ) from e


# 1. ORIGINAL AD DATASET LOADER
def load_ad_dataset(path: str):
    """
    Load the original AD dataset.

    Expected CSV format:
        - No header row.
        - All columns except the last are numeric features.
        - Last column is binary label: 0 = normal, 1 = anomaly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"AD dataset not found at {path}")

    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise ValueError("AD dataset must have at least 2 columns (features + label).")

    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)

    unique = set(np.unique(y))
    if not unique.issubset({0, 1}):
        raise ValueError(f"AD labels must be 0/1, got {unique}")

    return X, y



# 2. NEW CREDIT CARD DATASET LOADER

def load_creditcard_dataset(
    path: str,
    drop_time: bool = True,
    max_samples: int | None = 50000,
    scale_features: bool = True,
    random_seed: int = 42,
):
    """
    Load and preprocess the Kaggle credit card fraud dataset.

    Typical format (creditcard.csv):
        - Columns: Time, V1..V28, Amount, Class
        - 'Class' is label: 0 = normal, 1 = fraud

    Steps:
        1. Read CSV with header.
        2. Identify label column (Class / label / Label / y / target).
        3. Optionally drop 'Time' from features.
        4. Optionally subsample to max_samples.
        5. Optionally standardise features with StandardScaler.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Credit card CSV not found at {path}")

    df = pd.read_csv(path)

    # Find label column
    label_col = None
    for cand in ["Class", "label", "Label", "y", "target"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError(
            "Could not find label column in credit card CSV. "
            "Expected one of: Class, label, Label, y, target."
        )

    feature_cols = [c for c in df.columns if c != label_col]

    # Optionally drop "Time"
    if drop_time and "Time" in feature_cols:
        feature_cols.remove("Time")

    X = df[feature_cols].values.astype(float)
    y = df[label_col].values.astype(int)

    # Optional subsampling for efficiency
    if max_samples is not None and max_samples < X.shape[0]:
        rng = np.random.RandomState(random_seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # Optional feature scaling
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    unique = set(np.unique(y))
    if not unique.issubset({0, 1}):
        raise ValueError(f"Credit card labels must be 0/1, got {unique}")

    return X, y, feature_cols, label_col



# 3. COMMON OPTIFOREST EXPERIMENT FUNCTION

def run_optiforest_experiment(
    X: np.ndarray,
    y: np.ndarray,
    threshold: int,
    branch: int,
    num_trees: int,
    lsh_family: str,
    granularity: int,
    n_runs: int,
    random_seed: int = 123,
):
    """
    Run OptIForest multiple times on a dataset and compute metrics.

    Returns a dict with:
        - mean/std ROC AUC
        - mean/std PR AUC
        - mean train/test time
    """
    np.random.seed(random_seed)

    auc_list = []
    pr_list = []
    train_times = []
    test_times = []

    for _ in range(n_runs):
        detector = OptIForest(
            lsh_family=lsh_family,
            num_trees=num_trees,
            threshold=threshold,
            branch=branch,
            granularity=granularity,
        )

        t0 = time.time()
        detector.fit(X)
        train_times.append(time.time() - t0)

        t1 = time.time()
        scores = detector.decision_function(X)
        test_times.append(time.time() - t1)

        # OptIForest: lower score -> more anomalous
        scores_for_metrics = -1.0 * scores

        auc = roc_auc_score(y, scores_for_metrics)
        pr = average_precision_score(y, scores_for_metrics)

        auc_list.append(auc)
        pr_list.append(pr)

    return {
        "roc_auc_mean": float(np.mean(auc_list)),
        "roc_auc_std": float(np.std(auc_list)),
        "pr_auc_mean": float(np.mean(pr_list)),
        "pr_auc_std": float(np.std(pr_list)),
        "train_time_mean": float(np.mean(train_times)),
        "test_time_mean": float(np.mean(test_times)),
        "n_runs": int(n_runs),
        "num_trees": int(num_trees),
        "threshold": int(threshold),
        "branch": int(branch),
        "lsh_family": str(lsh_family),
        "granularity": int(granularity),
    }



# 4. MAIN: RUN AD + CREDITCARD EXPERIMENTS

def main():
    parser = argparse.ArgumentParser(
        description="Run OptIForest on AD (original) and creditcard (new) datasets."
    )

    # AD (original) dataset options
    parser.add_argument(
        "--ad_path",
        type=str,
        default="data/ad.csv",
        help="Path to original AD dataset CSV.",
    )
    parser.add_argument(
        "--ad_threshold",
        type=int,
        default=403,
        help="OptIForest cut threshold for AD dataset.",
    )
    parser.add_argument(
        "--ad_branch",
        type=int,
        default=0,
        help="OptIForest branch parameter for AD dataset.",
    )

    # Credit card (new) dataset options
    parser.add_argument(
        "--cc_path",
        type=str,
        default="data/creditcard.csv",
        help="Path to creditcard.csv downloaded from Kaggle.",
    )
    parser.add_argument(
        "--cc_threshold",
        type=int,
        default=403,
        help="OptIForest cut threshold for credit card dataset.",
    )
    parser.add_argument(
        "--cc_branch",
        type=int,
        default=0,
        help="OptIForest branch parameter for credit card dataset.",
    )
    parser.add_argument(
        "--cc_max_samples",
        type=int,
        default=50000,
        help="Maximum number of credit card rows to use (None = all).",
    )

    # Shared OptIForest parameters
    parser.add_argument(
        "--num_trees",
        type=int,
        default=10,
        help="Number of trees in OptIForest (increase for final runs).",
    )
    parser.add_argument(
        "--lsh_family",
        type=str,
        default="L2OPT",
        choices=["L2OPT", "L1OPT", "ALOPT"],
        help="LSH family for OptIForest.",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=1,
        help="Granularity parameter.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of repetitions to average results.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save metrics JSON.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- AD dataset (original) ---
    X_ad, y_ad = load_ad_dataset(args.ad_path)
    n_ad, d_ad = X_ad.shape
    n_ad_anom = int(y_ad.sum())

    print(f"[INFO] AD dataset: {n_ad} samples, {d_ad} features, {n_ad_anom} anomalies.")

    ad_results = run_optiforest_experiment(
        X=X_ad,
        y=y_ad,
        threshold=args.ad_threshold,
        branch=args.ad_branch,
        num_trees=args.num_trees,
        lsh_family=args.lsh_family,
        granularity=args.granularity,
        n_runs=args.n_runs,
    )

    # --- Credit card dataset (new data) ---
    X_cc, y_cc, feature_cols_cc, label_col_cc = load_creditcard_dataset(
        args.cc_path,
        drop_time=True,
        max_samples=args.cc_max_samples,
        scale_features=True,
        random_seed=42,
    )
    n_cc, d_cc = X_cc.shape
    n_cc_anom = int(y_cc.sum())

    print(
        f"[INFO] Credit card dataset: {n_cc} samples, {d_cc} features, "
        f"{n_cc_anom} anomalies."
    )

    cc_results = run_optiforest_experiment(
        X=X_cc,
        y=y_cc,
        threshold=args.cc_threshold,
        branch=args.cc_branch,
        num_trees=args.num_trees,
        lsh_family=args.lsh_family,
        granularity=args.granularity,
        n_runs=args.n_runs,
    )

    # --- Save metrics to JSON ---
    metrics = {
        "optiforest_params": {
            "num_trees": int(args.num_trees),
            "n_runs": int(args.n_runs),
            "lsh_family": args.lsh_family,
            "granularity": int(args.granularity),
        },
        "original_ad": {
            "path": args.ad_path,
            "n_samples": int(n_ad),
            "n_features": int(d_ad),
            "n_anomalies": int(n_ad_anom),
            "threshold": int(args.ad_threshold),
            "branch": int(args.ad_branch),
            "results": ad_results,
        },
        "new_creditcard": {
            "path": args.cc_path,
            "n_samples": int(n_cc),
            "n_features": int(d_cc),
            "n_anomalies": int(n_cc_anom),
            "threshold": int(args.cc_threshold),
            "branch": int(args.cc_branch),
            "label_column": label_col_cc,
            "feature_columns": feature_cols_cc,
            "max_samples": int(args.cc_max_samples)
            if args.cc_max_samples is not None
            else None,
            "preprocessing": {
                "drop_time": True,
                "scale_features": True,
            },
            "results": cc_results,
        },
    }

    metrics_path = os.path.join(args.output_dir, "ad_creditcard_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # --- Console summary ---
    print("\n=== ORIGINAL AD DATA RESULTS ===")
    for k, v in ad_results.items():
        print(f"{k}: {v}")

    print("\n=== NEW CREDIT CARD DATA RESULTS ===")
    for k, v in cc_results.items():
        print(f"{k}: {v}")

    print(f"\n[INFO] Metrics saved to: {metrics_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
