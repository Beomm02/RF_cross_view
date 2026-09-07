from __future__ import annotations

import numpy as np


class MahalanobisDetector:
    def __init__(self, covariance: str = "ledoit_wolf", eps: float = 1e-6):
        self.covariance = covariance
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.fitted_covariance_method_: str | None = None

    def fit(self, features: np.ndarray) -> "MahalanobisDetector":
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must have shape [N, D]")
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        if self.covariance == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf  # type: ignore

                estimator = LedoitWolf().fit(x)
                self.precision_ = estimator.precision_
                self.fitted_covariance_method_ = "sklearn_ledoit_wolf"
                return self
            except ModuleNotFoundError:
                pass
        if self.covariance == "ledoit_wolf":
            cov = ledoit_wolf_covariance(centered, eps=self.eps)
            self.precision_ = np.linalg.pinv(cov)
            self.fitted_covariance_method_ = "analytic_ledoit_wolf"
            return self
        cov = shrinkage_covariance(centered, eps=self.eps)
        self.precision_ = np.linalg.pinv(cov)
        self.fitted_covariance_method_ = "diagonal_shrinkage_fallback"
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisDetector must be fitted before score")
        x = np.asarray(features, dtype=np.float64)
        delta = x - self.mean_
        squared = np.sum((delta @ self.precision_) * delta, axis=1)
        return np.sqrt(np.maximum(squared, 0.0))


def shrinkage_covariance(centered: np.ndarray, eps: float = 1e-6, shrinkage: float = 0.1) -> np.ndarray:
    x = np.asarray(centered, dtype=np.float64)
    n_samples = max(x.shape[0] - 1, 1)
    sample_cov = (x.T @ x) / n_samples
    diagonal = np.diag(np.diag(sample_cov))
    cov = (1.0 - float(shrinkage)) * sample_cov + float(shrinkage) * diagonal
    cov += np.eye(cov.shape[0], dtype=np.float64) * float(eps)
    return cov


def ledoit_wolf_covariance(centered: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(centered, dtype=np.float64)
    n_samples = max(x.shape[0], 1)
    n_features = x.shape[1]
    emp_cov = (x.T @ x) / n_samples
    mu = float(np.trace(emp_cov) / max(n_features, 1))
    target = np.eye(n_features, dtype=np.float64) * mu
    delta = float(np.sum((emp_cov - target) ** 2))
    if delta <= 0.0:
        cov = target
    else:
        squared_norms = np.sum(x * x, axis=1)
        phi = float(np.mean(squared_norms * squared_norms) - np.sum(emp_cov * emp_cov))
        beta = min(max(phi / n_samples, 0.0), delta)
        shrinkage = beta / delta
        cov = (1.0 - shrinkage) * emp_cov + shrinkage * target
    cov += np.eye(n_features, dtype=np.float64) * float(eps)
    return cov
