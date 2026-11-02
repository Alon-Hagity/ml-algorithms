"""Soft-margin linear SVM trained by gradient descent on the regularized hinge loss."""
import numpy as np


class LinearSVM:
    """Linear support vector machine.

    Minimizes 0.5 * ||w||^2 + C * mean(max(0, 1 - y * (X @ w + b))).

    Parameters
    ----------
    C : float
        Weight of the hinge-loss (slack) term.
    learning_rate : float
        Gradient-descent step size.
    max_iter : int
        Maximum number of iterations.
    eps : float
        Stop when the change in loss falls below this value.
    random_state : int
        Seed for the random weight initialization.

    Attributes
    ----------
    w_ : ndarray of shape (n_features,)
    b_ : float
    loss_history_ : list of float
    """

    def __init__(self, C=1.0, learning_rate=0.05, max_iter=100000, eps=1e-10, random_state=42):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.loss_history_ = []

    def _loss_and_gradients(self, X, y):
        margins = 1 - y * self.decision_function(X)
        active = (margins > 0).astype(float)
        loss = 0.5 * np.sum(self.w_ ** 2) + self.C * np.mean(np.maximum(0, margins))
        dw = self.w_ - self.C * np.mean((active * y)[:, None] * X, axis=0)
        db = -self.C * np.mean(active * y)
        return loss, dw, db

    def fit(self, X, y):
        """Fit to X (n_samples, n_features) and labels y in {-1, 1} (0 is treated as -1)."""
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y).ravel() == 0, -1, np.asarray(y).ravel()).astype(float)
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.random(X.shape[1])
        self.b_ = float(rng.random())
        self.loss_history_ = []
        for _ in range(self.max_iter):
            loss, dw, db = self._loss_and_gradients(X, y)
            self.w_ -= self.learning_rate * dw
            self.b_ -= self.learning_rate * db
            if self.loss_history_ and abs(self.loss_history_[-1] - loss) < self.eps:
                break
            self.loss_history_.append(loss)
        return self

    def decision_function(self, X):
        """Return the signed distance score X @ w + b for each sample."""
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X):
        """Return predicted labels in {-1, 1}."""
        return np.where(self.decision_function(X) >= 0, 1, -1)
