"""K-means clustering with a configurable L_p distance."""
import numpy as np


class KMeans:
    """K-means clustering (Lloyd's algorithm).

    Parameters
    ----------
    k : int
        Number of clusters.
    p : float
        Order of the L_p distance used for assignment (2 = Euclidean).
    max_iter : int
        Maximum number of assignment/update rounds.
    eps : float
        Stop when no centroid coordinate moves more than this value.
    random_state : int or None
        Seed for initialization and for re-seeding empty clusters.

    Attributes
    ----------
    centroids_ : ndarray of shape (k, n_features)
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each training sample.
    inertia_ : float
        Within-cluster sum of distances raised to the power p (WCSS for p=2).
    """

    def __init__(self, k=3, p=2, max_iter=100, eps=1e-8, random_state=None):
        self.k = k
        self.p = p
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _distances(self, X):
        """Return an array of shape (k, n_samples) of L_p distances to each centroid."""
        diff = np.abs(X[None, :, :] - self.centroids_[:, None, :])
        return np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)

    def fit(self, X):
        """Cluster X of shape (n_samples, n_features)."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.centroids_ = X[rng.choice(X.shape[0], self.k, replace=False)].copy()
        for _ in range(self.max_iter):
            self.labels_ = np.argmin(self._distances(X), axis=0)
            previous = self.centroids_.copy()
            for j in range(self.k):
                members = X[self.labels_ == j]
                if len(members) == 0:
                    self.centroids_[j] = X[rng.integers(X.shape[0])]
                else:
                    self.centroids_[j] = members.mean(axis=0)
            if np.max(np.abs(self.centroids_ - previous)) <= self.eps:
                break
        distances = self._distances(X)
        self.labels_ = np.argmin(distances, axis=0)
        self.inertia_ = float(np.sum(np.min(distances, axis=0) ** self.p))
        return self

    def predict(self, X):
        """Assign each sample in X to its nearest learned centroid."""
        return np.argmin(self._distances(np.asarray(X, dtype=float)), axis=0)
