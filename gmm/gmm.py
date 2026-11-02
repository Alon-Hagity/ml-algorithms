"""One-dimensional Gaussian mixture model fitted with Expectation-Maximization."""
import numpy as np


def normal_pdf(x, mu, sigma):
    """Density of a univariate normal distribution N(mu, sigma^2) at x."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


class GaussianMixture1D:
    """Univariate Gaussian mixture model trained with EM.

    Parameters
    ----------
    k : int
        Number of Gaussian components.
    max_iter : int
        Maximum number of EM iterations.
    eps : float
        Stop when the change in negative log-likelihood falls below this value.
    random_state : int
        Seed for parameter initialization.

    Attributes
    ----------
    weights_, means_, sigmas_ : ndarray of shape (k,)
        Mixture weights, component means and standard deviations.
    loss_history_ : list of float
        Negative log-likelihood after each iteration.
    """

    _MIN_SIGMA = 1e-12

    def __init__(self, k=2, max_iter=1000, eps=1e-6, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        self.weights_ = self.means_ = self.sigmas_ = None
        self.loss_history_ = []

    def _init_params(self, x, rng):
        self.weights_ = np.full(self.k, 1.0 / self.k)
        if self.k == 1:
            self.means_ = np.array([x.mean()])
            self.sigmas_ = np.array([x.std()])
        else:
            self.means_ = rng.choice(x, self.k, replace=False).astype(float)
            self.sigmas_ = np.full(self.k, x.std())

    def _component_densities(self, x):
        return self.weights_ * normal_pdf(x[:, None], self.means_, self.sigmas_)

    def _expectation(self, x):
        dens = self._component_densities(x)
        return dens / dens.sum(axis=1, keepdims=True)

    def _maximization(self, x, resp):
        nk = resp.sum(axis=0)
        self.weights_ = nk / x.shape[0]
        self.means_ = (resp * x[:, None]).sum(axis=0) / nk
        var = (resp * (x[:, None] - self.means_) ** 2).sum(axis=0) / nk
        self.sigmas_ = np.maximum(np.sqrt(var), self._MIN_SIGMA)

    def _nll(self, x):
        return float(-np.sum(np.log(self._component_densities(x).sum(axis=1))))

    def fit(self, X):
        """Fit the mixture to 1-D data X (shape (n,) or (n, 1))."""
        x = np.asarray(X, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        self._init_params(x, rng)
        self.loss_history_ = []
        for _ in range(self.max_iter):
            self._maximization(x, self._expectation(x))
            self.loss_history_.append(self._nll(x))
            if len(self.loss_history_) > 1 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.eps:
                break
        return self

    def pdf(self, X):
        """Mixture density at each point of X."""
        return self._component_densities(np.asarray(X, dtype=float).ravel()).sum(axis=1)

    def predict_proba(self, X):
        """Posterior probability of each component for each point, shape (n, k)."""
        return self._expectation(np.asarray(X, dtype=float).ravel())

    def predict(self, X):
        """Most likely component index for each point."""
        return np.argmax(self.predict_proba(X), axis=1)
