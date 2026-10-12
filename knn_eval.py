"""
knn_eval.py
-----------
Step 2 of the project: classify images with K-Nearest Neighbors using the
features saved by extract_features.py. Runs on CPU in seconds.

- Features are L2-normalised, so Euclidean distance ranks neighbours exactly
  like cosine distance.
- k is chosen with 5-fold cross-validation on the TRAIN set only, so the
  evaluation split is never used for tuning.
- Evaluated on 'test' if it exists, otherwise on 'val'.
- Optional: --linear also trains a logistic-regression classifier on the
  same features for comparison.

Usage:
    python knn_eval.py --dataset flowers --model resnet50
    python knn_eval.py --dataset flowers --model dinov3 --linear
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

K_VALUES = [1, 3, 5, 10, 20]


def load(features_dir, dataset, tag, split):
    """Load one saved split. Returns (features, labels, class_names) or None."""
    path = os.path.join(features_dir, f"{dataset}_{tag}_{split}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    return normalize(d["features"]), d["labels"], list(d["class_names"])


def choose_k(X, y):
    """Pick k by 5-fold cross-validation on the training features."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = {}
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        scores[k] = cross_val_score(knn, X, y, cv=cv).mean()
        print(f"  k={k:<3d} CV accuracy = {scores[k]:.4f}")
    best_k = max(scores, key=scores.get)
    return best_k, scores


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name:>8s}: accuracy = {acc:.4f}, macro-F1 = {f1:.4f}")
    return {"accuracy": acc, "macro_f1": f1}


def save_confusion(y_true, y_pred, class_names, path, title):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Name of the data folder, e.g. 'flowers'.")
    parser.add_argument("--model", required=True,
                        help="resnet50, dinov3, resnet50_random, dinov3_random")
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--linear", action="store_true",
                        help="Also evaluate a logistic-regression classifier.")
    args = parser.parse_args()

    train = load(args.features_dir, args.dataset, args.model, "train")
    if train is None:
        raise FileNotFoundError("Train features not found; run extract_features.py first.")
    test = load(args.features_dir, args.dataset, args.model, "test")
    eval_split = "test"
    if test is None:
        test = load(args.features_dir, args.dataset, args.model, "val")
        eval_split = "val"

    X_tr, y_tr, class_names = train
    X_te, y_te, _ = test
    print(f"Train: {X_tr.shape}, eval ({eval_split}): {X_te.shape}")

    # 1) Choose k on the training set only.
    print("Choosing k with 5-fold CV on train:")
    best_k, cv_scores = choose_k(X_tr, y_tr)
    print(f"Best k = {best_k}\n")

    # 2) Fit KNN on all training features, evaluate on the held-out split.
    knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance").fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    results = {"dataset": args.dataset, "model": args.model,
               "eval_split": eval_split, "best_k": best_k,
               "cv_scores": cv_scores, "knn": evaluate("KNN", y_te, y_pred)}

    os.makedirs(args.results_dir, exist_ok=True)
    prefix = os.path.join(args.results_dir, f"{args.dataset}_{args.model}")
    save_confusion(y_te, y_pred, class_names, f"{prefix}_confusion.png",
                   f"{args.model} + KNN (k={best_k})")

    # 3) Optional linear classifier on the same frozen features.
    if args.linear:
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_tr)
        results["linear"] = evaluate("Linear", y_te, clf.predict(X_te))

    with open(f"{prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results to {prefix}_results.json and confusion matrix png.")


if __name__ == "__main__":
    main()
