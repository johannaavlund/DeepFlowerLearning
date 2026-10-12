"""
knn_eval.py
-----------
Step 2 of the project: classify images with K-Nearest Neighbors using the
features saved by extract_features.py. Runs on CPU in seconds.

- Features are L2-normalised, so Euclidean distance ranks neighbours exactly
  like cosine distance.
- If the dataset has a 'test' split it is used for evaluation. Otherwise a
  stratified 20% of 'train' is held out as test set (saved to disk so all
  models use the same images), and the rest is the KNN reference set.
- k is chosen with 5-fold cross-validation on the reference set only, so
  the test images are never used for tuning.
- The dataset's original 'val' split (if any) is reported as a secondary result.
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
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

K_VALUES = [1, 3, 5, 10, 20]


def load(features_dir, dataset, tag, split):
    """Load one saved split.

    Returns (features, labels, class_names, image_paths) or None.
    """
    path = os.path.join(features_dir, f"{dataset}_{tag}_{split}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    return (normalize(d["features"]), d["labels"],
            list(d["class_names"]), d["paths"])


def held_out_mask(features_dir, dataset, paths, labels, test_size):
    """Return a boolean mask marking which training images belong to our test set.

    The first time this runs for a dataset, a stratified random split
    (same class proportions in both parts, fixed seed) is created and the
    test image paths are saved to features/<dataset>_split.npz. Later runs,
    including runs for other models, reuse that file so all models are
    compared on exactly the same test images.
    """
    split_path = os.path.join(features_dir, f"{dataset}_split.npz")
    if os.path.exists(split_path):
        test_paths = set(np.load(split_path, allow_pickle=False)["test_paths"])
        print(f"Using saved split: {split_path}")
    else:
        _, test_idx = train_test_split(np.arange(len(labels)), test_size=test_size,
                                       stratify=labels, random_state=0)
        test_paths = set(paths[test_idx])
        np.savez_compressed(split_path, test_paths=np.array(sorted(test_paths)))
        print(f"Created new stratified split: {split_path}")

    mask = np.array([p in test_paths for p in paths])
    if mask.sum() == 0:
        raise ValueError("Saved split does not match these features. "
                         f"Delete {split_path} and run again.")
    return mask


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
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of train held out as test set "
                             "(only used if the dataset has no 'test' split).")
    args = parser.parse_args()

    train = load(args.features_dir, args.dataset, args.model, "train")
    if train is None:
        raise FileNotFoundError("Train features not found; run extract_features.py first.")
    X_all, y_all, class_names, paths_all = train

    # 1) Build the evaluation set(s).
    #    - If the dataset has its own 'test' split, use it.
    #    - Otherwise hold out a stratified part of 'train' as our test set.
    #      The split is saved to disk and reused, so every model is
    #      evaluated on exactly the same images.
    test = load(args.features_dir, args.dataset, args.model, "test")
    if test is not None:
        X_tr, y_tr = X_all, y_all
        X_te, y_te = test[0], test[1]
        eval_split = "test"
    else:
        is_test = held_out_mask(args.features_dir, args.dataset, paths_all,
                                y_all, args.test_size)
        X_tr, y_tr = X_all[~is_test], y_all[~is_test]
        X_te, y_te = X_all[is_test], y_all[is_test]
        eval_split = f"held-out {int(args.test_size * 100)}% of train"

    # The original (small) 'val' split is reported as a secondary result.
    val = load(args.features_dir, args.dataset, args.model, "val")
    if eval_split == "test":
        val = None  # with a real test split, val is not needed here

    print(f"KNN reference set: {X_tr.shape}, test ({eval_split}): {X_te.shape}")

    # 2) Choose k with cross-validation on the reference set only.
    #    The test images are never used for this choice.
    print("Choosing k with 5-fold CV on the reference set:")
    best_k, cv_scores = choose_k(X_tr, y_tr)
    print(f"Best k = {best_k}\n")

    # 3) Fit KNN on the reference set, evaluate on the held-out images.
    knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance").fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    results = {"dataset": args.dataset, "model": args.model,
               "eval_split": eval_split, "n_reference": int(len(y_tr)),
               "n_test": int(len(y_te)), "best_k": best_k,
               "cv_scores": cv_scores, "knn": evaluate("KNN", y_te, y_pred)}
    if val is not None:
        results["knn_original_val"] = evaluate("KNN val", val[1], knn.predict(val[0]))

    os.makedirs(args.results_dir, exist_ok=True)
    prefix = os.path.join(args.results_dir, f"{args.dataset}_{args.model}")
    save_confusion(y_te, y_pred, class_names, f"{prefix}_confusion.png",
                   f"{args.model} + KNN (k={best_k})")

    # 4) Optional linear classifier on the same frozen features.
    if args.linear:
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_tr)
        results["linear"] = evaluate("Linear", y_te, clf.predict(X_te))
        if val is not None:
            results["linear_original_val"] = evaluate(
                "Lin. val", val[1], clf.predict(val[0]))

    with open(f"{prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results to {prefix}_results.json and confusion matrix png.")


if __name__ == "__main__":
    main()