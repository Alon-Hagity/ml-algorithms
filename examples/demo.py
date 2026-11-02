"""Run every model on synthetic NumPy data and save result plots to assets/."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gmm import GaussianMixture1D  # noqa: E402
from kmeans import KMeans  # noqa: E402
from logistic_regression import LogisticRegression  # noqa: E402
from svm import LinearSVM  # noqa: E402

ASSETS = os.path.join(ROOT, "assets")
rng = np.random.default_rng(0)


def two_blobs(n=200, sep=2.5):
    X = np.vstack([rng.normal(-sep / 2, 1, (n, 2)), rng.normal(sep / 2, 1, (n, 2))])
    y = np.r_[np.zeros(n), np.ones(n)]
    return X, y


def plot_boundary(ax, model, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, edgecolors="k", linewidths=0.2)
    ax.set_title(title)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    X, y = two_blobs()

    logreg = LogisticRegression(learning_rate=0.1).fit(X, y)
    acc_lr = np.mean(logreg.predict(X) == y)
    svm = LinearSVM(C=1.0, learning_rate=0.01, max_iter=20000).fit(X, 2 * y - 1)
    acc_svm = np.mean(svm.predict(X) == 2 * y - 1)
    print(f"Logistic regression training accuracy: {acc_lr:.3f}")
    print(f"Linear SVM training accuracy:          {acc_svm:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_boundary(axes[0], logreg, X, y, f"Logistic regression (acc {acc_lr:.2f})")
    plot_boundary(axes[1], svm, X, 2 * y - 1, f"Linear SVM (acc {acc_svm:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "linear_classifiers.png"), dpi=120)
    plt.close(fig)

    centers = np.array([[0, 0], [5, 5], [0, 6]])
    Xc = np.vstack([rng.normal(c, 1.0, (150, 2)) for c in centers])
    km = KMeans(k=3, random_state=0).fit(Xc)
    print(f"K-means inertia (WCSS):                {km.inertia_:.1f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(Xc[:, 0], Xc[:, 1], c=km.labels_, cmap="viridis", s=10)
    ax.scatter(km.centroids_[:, 0], km.centroids_[:, 1], c="red", marker="x", s=120)
    ax.set_title("K-means (k=3)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "kmeans.png"), dpi=120)
    plt.close(fig)

    x1d = np.r_[rng.normal(-3, 1.0, 400), rng.normal(2, 0.7, 600)]
    gmm = GaussianMixture1D(k=2).fit(x1d)
    print("GMM weights:", np.round(gmm.weights_, 3), "means:", np.round(gmm.means_, 3),
          "sigmas:", np.round(gmm.sigmas_, 3))
    grid = np.linspace(-7, 6, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x1d, bins=50, density=True, alpha=0.4, label="data")
    ax.plot(grid, gmm.pdf(grid), "r", label="fitted GMM")
    ax.set_title("Gaussian mixture (EM, k=2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "gmm.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
