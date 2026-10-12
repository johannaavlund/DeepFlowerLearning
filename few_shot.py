"""
few_shot.py
-----------
Few-shot experiment: how does KNN accuracy change when only n labelled
reference images per class are available?

For each n in N_SHOTS and each random seed, n images per class are drawn
from the reference set, a KNN classifier is built from them, and it is
evaluated on the SAME test set as knn_eval.py. Mean and standard deviation
over the seeds are reported.

Run knn_eval.py for each dataset/model first (it creates the test split).

Usage:
    # 1) Run the experiment (one dataset, one or more models)
    python few_shot.py --dataset flowers    --models resnet50 dinov3
    python few_shot.py --dataset flowers102 --models resnet50 dinov3

    # 2) Plot all datasets in one figure
    python few_shot.py --plot --datasets flowers flowers102 \
                       --labels Kaggle-14 Flowers-102
"""

import argparse
import csv
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from show_errors import get_split  # same reference/test split as knn_eval.py

N_SHOTS = [1, 2, 5, 10, 20, 50]   # images per class (skipped if too few exist)
N_SEEDS = 5                       # random draws per n
MODEL_LABELS = {"resnet50": "ResNet-50", "dinov3": "DINOv3"}


def run(dataset, models, features_dir, results_dir):
    rows = []
    for model in models:
        a = SimpleNamespace(features_dir=features_dir, dataset=dataset, model=model)
        X_ref, y_ref, _, X_te, y_te, _, _ = get_split(a)
        classes = np.unique(y_ref)
        min_per_class = min((y_ref == c).sum() for c in classes)

        for n in N_SHOTS:
            if n > min_per_class:
                continue  # not enough reference images for every class
            accs = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                # Draw n reference images from every class.
                idx = np.concatenate([
                    rng.choice(np.where(y_ref == c)[0], size=n, replace=False)
                    for c in classes])
                # k cannot exceed the number of images per class in a sensible way;
                # use k = min(5, n) (k=5 was selected by cross-validation earlier).
                k = min(5, n)
                knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
                knn.fit(X_ref[idx], y_ref[idx])
                accs.append(accuracy_score(y_te, knn.predict(X_te)))
            mean, std = float(np.mean(accs)), float(np.std(accs))
            rows.append([model, n, mean, std])
            print(f"{model:>9s}  n={n:<3d} accuracy = {100*mean:5.1f} ± {100*std:.1f} %")

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{dataset}_fewshot.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "shots", "accuracy_mean", "accuracy_std"])
        w.writerows(rows)
    print(f"\nSaved {path}")


def plot(datasets, labels, results_dir):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 2.8),
                             sharey=True, squeeze=False)
    for ax, dataset, label in zip(axes[0], datasets, labels):
        with open(os.path.join(results_dir, f"{dataset}_fewshot.csv")) as f:
            data = list(csv.DictReader(f))
        for model, style in (("resnet50", "o--"), ("dinov3", "s-")):
            r = [d for d in data if d["model"] == model]
            if not r:
                continue
            n = np.array([int(d["shots"]) for d in r])
            m = 100 * np.array([float(d["accuracy_mean"]) for d in r])
            s = 100 * np.array([float(d["accuracy_std"]) for d in r])
            ax.errorbar(n, m, yerr=s, fmt=style, capsize=3, markersize=4,
                        label=MODEL_LABELS.get(model, model))
        ax.set_xscale("log")
        ax.set_xticks(n)
        ax.set_xticklabels(n)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Reference images per class", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("Test accuracy (%)", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(results_dir, f"fewshot.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"Saved {results_dir}/fewshot.png and .pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--models", nargs="+", default=["resnet50", "dinov3"])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--features-dir", default="features")
    p.add_argument("--results-dir", default="results")
    a = p.parse_args()

    if a.plot:
        plot(a.datasets, a.labels or a.datasets, a.results_dir)
    else:
        run(a.dataset, a.models, a.features_dir, a.results_dir)


if __name__ == "__main__":
    main()