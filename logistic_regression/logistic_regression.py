"""Binary logistic regression trained with batch gradient descent on the BCE loss."""
import numpy as np


class LogisticRegression:
    """Binary logistic regression classifier.

    Parameters
    ----------
    learning_rate : float
        Gradient-descent step size.
    max_iter : int
        Maximum number of gradient-descent iterations.
    eps : float
        Stop when the change in BCE loss between iterations falls below this value.
    random_state : int
        Seed for the small random weight initialization.

    Attributes
    ----------
    w_ : ndarray of shape (n_features + 1,)
        Learned weights; w_[0] is the bias.
    classes_ : ndarray of shape (2,)
        Original class labels, mapped internally to 0 and 1.
    loss_history_ : list of float
        BCE loss recorded after each iteration.
    """

    def __init__(self, learning_rate=0.1, max_iter=10000, eps=1e-6, random_state=1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        self.w_ = None
        self.classes_ = None
        self.loss_history_ = []

    @staticmethod
    def _add_bias(X):
        return np.column_stack((np.ones(X.shape[0]), X))

    def _proba(self, Xb):
        return 1.0 / (1.0 + np.exp(-(Xb @ self.w_)))

    def _bce(self, Xb, y01):
        p = np.clip(self._proba(Xb), 1e-12, 1 - 1e-12)
        return -np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p))

    def fit(self, X, y):
        """Fit the model to features X (n_samples, n_features) and binary labels y."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegression supports exactly two classes")
        y01 = (y == self.classes_[1]).astype(float)
        Xb = self._add_bias(X)
        rng = np.random.default_rng(self.random_state)
        self.w_ = 1e-6 * rng.standard_normal(Xb.shape[1])
        self.loss_history_ = []
        for i in range(self.max_iter):
            self.loss_history_.append(self._bce(Xb, y01))
            if i > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.eps:
                break
            grad = Xb.T @ (self._proba(Xb) - y01) / Xb.shape[0]
            self.w_ -= self.learning_rate * grad
        return self

    def predict_proba(self, X):
        """Return the probability of the positive class (classes_[1]) for each sample."""
        return self._proba(self._add_bias(np.asarray(X, dtype=float)))

    def predict(self, X, threshold=0.5):
        """Return predicted labels, using the same label values seen in fit."""
        return np.where(self.predict_proba(X) >= threshold, self.classes_[1], self.classes_[0])
