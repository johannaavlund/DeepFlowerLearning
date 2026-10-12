"""
show_errors.py
--------------
Error analysis: find which test images KNN gets wrong, list the most
confused species pairs, and save a figure with example images.
 
Uses exactly the same reference/test split and k as knn_eval.py, so run
knn_eval.py for the model first.
 
Output (in results/):
    <dataset>_<model>_errors.csv        every misclassified test image
    <dataset>_<model>_top_pairs.csv     most confused (true, predicted) pairs
    <dataset>_<model>_examples.png/pdf  figure: one row per confused pair:
                                        misclassified images + a typical
                                        image of the class it was mistaken for
 
Usage:
    python show_errors.py --dataset flowers --model resnet50
    python show_errors.py --dataset flowers --model resnet50 --pairs 3 --per-pair 2
"""
 
import argparse
import csv
import json
import os
from collections import Counter
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
 
# Reuse the exact loading and splitting logic from knn_eval.py.
from knn_eval import held_out_mask, load
 
 
def get_split(args):
    """Return reference/test features, labels and paths, as in knn_eval.py."""
    X_all, y_all, class_names, paths_all = load(
        args.features_dir, args.dataset, args.model, "train")
    test = load(args.features_dir, args.dataset, args.model, "test")
    if test is not None:
        return (X_all, y_all, paths_all,
                test[0], test[1], test[3], class_names)
    # No official test split: use the saved held-out split (created by knn_eval.py).
    split_path = os.path.join(args.features_dir, f"{args.dataset}_split.npz")
    if not os.path.exists(split_path):
        raise FileNotFoundError("Run knn_eval.py first so the split exists.")
    is_test = held_out_mask(args.features_dir, args.dataset, paths_all, y_all, 0.2)
    return (X_all[~is_test], y_all[~is_test], paths_all[~is_test],
            X_all[is_test], y_all[is_test], paths_all[is_test], class_names)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pairs", type=int, default=4,
                        help="Number of confused pairs to show in the figure.")
    parser.add_argument("--per-pair", type=int, default=3,
                        help="Misclassified examples per pair in the figure.")
    args = parser.parse_args()
 
    X_ref, y_ref, p_ref, X_te, y_te, p_te, class_names = get_split(args)
    prefix = os.path.join(args.results_dir, f"{args.dataset}_{args.model}")
 
    # Use the same k that knn_eval.py selected by cross-validation.
    with open(f"{prefix}_results.json") as f:
        k = json.load(f)["best_k"]
 
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_ref, y_ref)
    y_pred = knn.predict(X_te)
    wrong = np.where(y_pred != y_te)[0]
    print(f"{args.model}, k={k}: {len(wrong)} of {len(y_te)} test images misclassified")
 
    # 1) Save every misclassified image to a CSV file.
    with open(f"{prefix}_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class", "predicted_class", "image_path"])
        for i in wrong:
            w.writerow([class_names[y_te[i]], class_names[y_pred[i]], p_te[i]])
 
    # 2) Count the most frequent (true, predicted) pairs.
    #    Share = errors / number of test images of the true class.
    class_counts = Counter(y_te.tolist())
    pair_counts = Counter((int(y_te[i]), int(y_pred[i])) for i in wrong)
    top = pair_counts.most_common()
    with open(f"{prefix}_top_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_class", "predicted_class", "count", "share_of_true_class"])
        for (t, p), n in top:
            w.writerow([class_names[t], class_names[p], n,
                        f"{n / class_counts[t]:.3f}"])
    print("\nMost confused pairs (true -> predicted):")
    for (t, p), n in top[:10]:
        print(f"  {class_names[t]:>20s} -> {class_names[p]:<20s}"
              f" {n:3d} images ({100 * n / class_counts[t]:.1f}% of {class_names[t]})")
 
    # 3) Figure: for each top pair, show misclassified images of the true class
    #    and, in the last column, a correctly classified image of the class it
    #    was confused with (for visual comparison).
    rng = np.random.default_rng(0)
    pairs = top[:args.pairs]
    if not pairs:
        print("No errors, no figure to make.")
        return
    ncols = args.per_pair + 1
    fig, axes = plt.subplots(len(pairs), ncols,
                             figsize=(2.2 * ncols, 2.4 * len(pairs)), squeeze=False)
    for row, ((t, p), _) in enumerate(pairs):
        idx = [i for i in wrong if y_te[i] == t and y_pred[i] == p]
        idx = rng.choice(idx, size=min(args.per_pair, len(idx)), replace=False)
        ref_idx = np.where(y_ref == p)[0]
        reference = p_ref[rng.choice(ref_idx)]
 
        for col in range(ncols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(idx):
                ax.imshow(Image.open(p_te[idx[col]]).convert("RGB"))
                ax.set_title(f"{class_names[t]}\n(predicted {class_names[p]})",
                             fontsize=8)
            elif col == ncols - 1:
                ax.imshow(Image.open(reference).convert("RGB"))
                ax.set_title(f"typical {class_names[p]}", fontsize=8,
                             fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{prefix}_examples.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {prefix}_errors.csv, {prefix}_top_pairs.csv and "
          f"{prefix}_examples.png/.pdf")
 
 
if __name__ == "__main__":
    main()
