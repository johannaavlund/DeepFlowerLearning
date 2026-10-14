"""
gray_eval.py
------------
Colour-dependence experiment: how much does KNN accuracy drop when the
TEST images are converted to grayscale, while the reference set stays in
colour? A model that relies mainly on colour should lose more accuracy.

The reference set, test images and k are exactly the same as in
knn_eval.py; only the colour of the test images differs.

Run first (for each model):
    python knn_eval.py        --dataset flowers --model resnet50
    python extract_features.py --data-root data/flowers --model resnet50 --grayscale

Then:
    python gray_eval.py --dataset flowers --models resnet50 dinov3
"""

import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from show_errors import get_split  # same reference/test split as knn_eval.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--models", nargs="+", default=["resnet50", "dinov3"])
    p.add_argument("--features-dir", default="features")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    rows = []
    for model in args.models:
        a = SimpleNamespace(features_dir=args.features_dir,
                            dataset=args.dataset, model=model)
        X_ref, y_ref, _, X_te, y_te, p_te, _ = get_split(a)

        # Grayscale features of the same test images, matched by file path.
        gray_path = os.path.join(args.features_dir,
                                 f"{args.dataset}_{model}_gray_test.npz")
        g = np.load(gray_path, allow_pickle=False)
        gray_index = {path: i for i, path in enumerate(g["paths"])}
        missing = [path for path in p_te if path not in gray_index]
        if missing:
            raise ValueError(f"{len(missing)} test images have no grayscale "
                             "features. Re-run extract_features.py --grayscale.")
        order = [gray_index[path] for path in p_te]
        X_gray = normalize(g["features"][order])
        assert (g["labels"][order] == y_te).all(), "Label mismatch"

        # Same k as selected by cross-validation in knn_eval.py.
        with open(os.path.join(args.results_dir,
                               f"{args.dataset}_{model}_results.json")) as f:
            k = json.load(f)["best_k"]
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_ref, y_ref)

        pred_col, pred_gray = knn.predict(X_te), knn.predict(X_gray)
        acc_col, acc_gray = accuracy_score(y_te, pred_col), accuracy_score(y_te, pred_gray)
        f1_gray = f1_score(y_te, pred_gray, average="macro")
        rows.append([model, acc_col, acc_gray, acc_col - acc_gray, f1_gray])
        print(f"{model:>9s}: colour {100*acc_col:5.1f} %  ->  grayscale "
              f"{100*acc_gray:5.1f} %   (drop {100*(acc_col-acc_gray):4.1f} pp, "
              f"gray macro-F1 {f1_gray:.3f})")

    out = os.path.join(args.results_dir, f"{args.dataset}_grayscale.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc_colour", "acc_grayscale", "drop", "macro_f1_grayscale"])
        w.writerows(rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()