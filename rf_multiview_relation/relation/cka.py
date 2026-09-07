from __future__ import annotations

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_values = _center(np.asarray(x, dtype=np.float64))
    y_values = _center(np.asarray(y, dtype=np.float64))
    numerator = float(np.linalg.norm(y_values.T @ x_values, ord="fro") ** 2)
    x_norm = float(np.linalg.norm(x_values.T @ x_values, ord="fro"))
    y_norm = float(np.linalg.norm(y_values.T @ y_values, ord="fro"))
    denominator = x_norm * y_norm
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0, keepdims=True)
