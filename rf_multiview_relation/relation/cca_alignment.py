from __future__ import annotations

import numpy as np


class Standardizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        values = np.asarray(x, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fitted before transform")
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.std_


class CCAAlignment:
    def __init__(self, n_components: int = 16, standardize: bool = True, regularization: float = 1e-6):
        self.n_components = int(n_components)
        self.standardize = bool(standardize)
        self.regularization = float(regularization)
        self.x_scaler = Standardizer()
        self.y_scaler = Standardizer()
        self.x_weights_: np.ndarray | None = None
        self.y_weights_: np.ndarray | None = None
        self.correlations_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CCAAlignment":
        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        if x_values.shape[0] != y_values.shape[0]:
            raise ValueError("x and y must have the same number of paired samples")
        if x_values.ndim != 2 or y_values.ndim != 2:
            raise ValueError("x and y must have shape [N, D]")
        if self.standardize:
            x_values = self.x_scaler.fit(x_values).transform(x_values)
            y_values = self.y_scaler.fit(y_values).transform(y_values)
        else:
            self.x_scaler.fit(np.zeros_like(x_values))
            self.y_scaler.fit(np.zeros_like(y_values))
        x_values = x_values - x_values.mean(axis=0, keepdims=True)
        y_values = y_values - y_values.mean(axis=0, keepdims=True)
        n = max(x_values.shape[0] - 1, 1)
        cxx = (x_values.T @ x_values) / n + np.eye(x_values.shape[1]) * self.regularization
        cyy = (y_values.T @ y_values) / n + np.eye(y_values.shape[1]) * self.regularization
        cxy = (x_values.T @ y_values) / n
        invsqrt_x = _inverse_sqrt(cxx)
        invsqrt_y = _inverse_sqrt(cyy)
        matrix = invsqrt_x @ cxy @ invsqrt_y
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        k = min(self.n_components, u.shape[1], vt.shape[0])
        self.x_weights_ = invsqrt_x @ u[:, :k]
        self.y_weights_ = invsqrt_y @ vt.T[:, :k]
        self.correlations_ = np.clip(singular_values[:k], 0.0, 1.0)
        return self

    def transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x_weights_ is None or self.y_weights_ is None:
            raise RuntimeError("CCAAlignment must be fitted before transform")
        x_values = self.x_scaler.transform(x) if self.standardize else np.asarray(x, dtype=np.float64)
        y_values = self.y_scaler.transform(y) if self.standardize else np.asarray(y, dtype=np.float64)
        return x_values @ self.x_weights_, y_values @ self.y_weights_


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    values = np.maximum(values, 1e-12)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T
