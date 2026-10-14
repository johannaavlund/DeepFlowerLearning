"""
robustness_eval.py
------------------
Robustness experiment: how much does KNN accuracy drop when the TEST
images are perturbed (no colour, rotated, darker, blurred, cropped), while
the reference set stays unchanged?

The reference set, test images and k are exactly the same as in
knn_eval.py; only the test images are perturbed.

Run first (for each model):
    python knn_eval.py         --dataset flowers --model resnet50
    python extract_features.py --data-root data/flowers --model resnet50 \
        --perturb gray rotate dark blur crop

Then:
    python robustness_eval.py --dataset flowers --models resnet50 dinov3

Output: results/<dataset>_robustness.csv and a LaTeX table printed to the
terminal. Perturbations whose features are missing are skipped.
"""

import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from show_errors import get_split  # same reference/test split as knn_eval.py

PERTURBATIONS = ["gray", "rotate", "dark", "blur", "crop"]
LABELS = {"gray": "Gray", "rotate": "Rot.\\ 90", "dark": "Dark",
          "blur": "Blur", "crop": "Crop"}
MODEL_LABELS = {"resnet50": "ResNet-50", "dinov3": "DINOv3"}


def load_perturbed(features_dir, dataset, model, perturb, test_paths, y_test):
    """Load perturbed test features in the same order as the clean test set."""
    path = os.path.join(features_dir, f"{dataset}_{model}_{perturb}_test.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    index = {p: i for i, p in enumerate(d["paths"])}
    missing = [p for p in test_paths if p not in index]
    if missing:
        raise ValueError(f"{path}: {len(missing)} test images missing. "
                         "Re-run extract_features.py --perturb.")
    order = [index[p] for p in test_paths]
    assert (d["labels"][order] == y_test).all(), "Label mismatch"
    return normalize(d["features"][order])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--models", nargs="+", default=["resnet50", "dinov3"])
    p.add_argument("--features-dir", default="features")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    table = {}  # model -> {"clean": acc, perturb: acc, ...}
    for model in args.models:
        a = SimpleNamespace(features_dir=args.features_dir,
                            dataset=args.dataset, model=model)
        X_ref, y_ref, _, X_te, y_te, p_te, _ = get_split(a)

        # Same k as selected by cross-validation in knn_eval.py.
        with open(os.path.join(args.results_dir,
                               f"{args.dataset}_{model}_results.json")) as f:
            k = json.load(f)["best_k"]
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_ref, y_ref)

        accs = {"clean": accuracy_score(y_te, knn.predict(X_te))}
        for perturb in PERTURBATIONS:
            X = load_perturbed(args.features_dir, args.dataset, model, perturb, p_te, y_te)
            if X is not None:
                accs[perturb] = accuracy_score(y_te, knn.predict(X))
        table[model] = accs

        line = "  ".join(f"{name} {100*v:5.1f}" for name, v in accs.items())
        print(f"{model:>9s}: {line}")

    # Save CSV (accuracies in %).
    cols = ["clean"] + [c for c in PERTURBATIONS
                        if all(c in t for t in table.values())]
    out = os.path.join(args.results_dir, f"{args.dataset}_robustness.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols)
        for model, accs in table.items():
            w.writerow([model] + [f"{100*accs[c]:.1f}" for c in cols])
    print(f"\nSaved {out}")

    # Print a ready-to-paste LaTeX table body.
    print("\nLaTeX rows:")
    print("Backbone & Clean & " + " & ".join(LABELS[c] for c in cols[1:]) + " \\\\")
    for model, accs in table.items():
        vals = " & ".join(f"{100*accs[c]:.1f}" for c in cols)
        print(f"{MODEL_LABELS.get(model, model)} & {vals} \\\\")


if __name__ == "__main__":
    main()