"""Lithography defect prediction from focus and dose.

Merges exposure and inspection data, trains three classifiers (MLP, Gaussian
Process, SVM) to predict defect / no defect from Focus [nm] and Dose [mJ],
prints metrics and saves decision-boundary plots.

Usage:
    python litho_defect_prediction.py \
        --exposure data/exposure_data.csv \
        --inspection data/inspection_data.csv \
        --out results
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works without a display; plots are saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MERGE_KEYS = ["Lot", "Wafer", "FieldX", "FieldY"]
SEED = 42


def find_col(df, keyword, fallback_idx):
    """Find a column whose name contains `keyword`; else use positional fallback."""
    for col in df.columns:
        if keyword in str(col).lower():
            return col
    return df.columns[fallback_idx]


def load_data(exposure_path, inspection_path):
    exposure = pd.read_csv(exposure_path).dropna()
    inspection = pd.read_csv(inspection_path).dropna()
    merged = pd.merge(exposure, inspection, how="inner", on=MERGE_KEYS)

    # Fallback indices (9, 10, 12) match the original notebook's layout.
    focus = find_col(merged, "focus", 9)
    dose = find_col(merged, "dose", 10)
    label = find_col(merged, "label", 12)
    print(f"Using columns -> focus: {focus!r}, dose: {dose!r}, label: {label!r}")

    X = merged[[focus, dose]].to_numpy(dtype=float)
    y = merged[label].to_numpy().astype(int)
    print(f"Samples: {len(y)} | defects: {int(y.sum())} | good: {int((y == 0).sum())}")
    return X, y


def build_models():
    return {
        "MLP": MLPClassifier(hidden_layer_sizes=(40, 40, 40), max_iter=1000, random_state=SEED),
        "GP": GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=SEED),
        "SVM": SVC(gamma=2, C=1, random_state=SEED),
    }


def scatter_points(ax, X, y):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=20, c="C0", edgecolor="k", linewidths=0.5, label="Good")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=20, c="C3", edgecolor="k", linewidths=0.5, label="Defect")
    ax.set_xlabel("Focus [nm]")
    ax.set_ylabel("Dose [mJ]")


def plot_boundary(ax, model, scaler, X, y, title):
    Xs = scaler.transform(X)
    xx, yy = np.meshgrid(
        np.linspace(Xs[:, 0].min() - 0.5, Xs[:, 0].max() + 0.5, 150),
        np.linspace(Xs[:, 1].min() - 0.5, Xs[:, 1].max() + 0.5, 150),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = model.predict(grid).reshape(xx.shape)
    grid_orig = scaler.inverse_transform(grid)  # back to real focus / dose units
    ax.contourf(
        grid_orig[:, 0].reshape(xx.shape),
        grid_orig[:, 1].reshape(xx.shape),
        z, cmap="RdYlBu_r", alpha=0.6,
    )
    scatter_points(ax, X, y)
    ax.set_title(title)
    ax.legend(loc=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exposure", default="data/exposure_data.csv")
    parser.add_argument("--inspection", default="data/inspection_data.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.exposure, args.inspection)

    # Raw data plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    scatter_points(ax, X, y)
    ax.set_title("Raw data")
    ax.legend(loc=4)
    fig.savefig(out / "raw_data.png", bbox_inches="tight")
    plt.close(fig)

    # Split first, then fit scaler on train only (no test-set leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    models = build_models()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5), dpi=100)
    rows = []
    for ax, (name, model) in zip(np.atleast_1d(axes), models.items()):
        print(f"\n=== {name} ===")
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)
        print(confusion_matrix(y_te, pred))
        print(classification_report(y_te, pred))
        rep = classification_report(y_te, pred, output_dict=True)
        rows.append({
            "model": name,
            "accuracy": rep["accuracy"],
            "defect_precision": rep["1"]["precision"],
            "defect_recall": rep["1"]["recall"],
            "defect_f1": rep["1"]["f1-score"],
        })
        plot_boundary(ax, model, scaler, X, y, name)

    fig.tight_layout()
    fig.savefig(out / "decision_boundaries.png", bbox_inches="tight")
    plt.close(fig)

    metrics = pd.DataFrame(rows).round(3)
    metrics.to_csv(out / "metrics.csv", index=False)
    print("\nSummary:\n", metrics.to_string(index=False))
    print(f"\nSaved plots and metrics to: {out.resolve()}")


if __name__ == "__main__":
    main()
