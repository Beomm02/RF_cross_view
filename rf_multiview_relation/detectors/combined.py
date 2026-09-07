from __future__ import annotations

import numpy as np


EPS = 1e-8


class RobustScoreFusion:
    def __init__(self, alpha: float = 0.5):
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self.abs_median = 0.0
        self.abs_iqr = 1.0
        self.rel_median = 0.0
        self.rel_iqr = 1.0

    def fit(self, absolute_scores: np.ndarray, relation_scores: np.ndarray) -> "RobustScoreFusion":
        abs_scores = np.asarray(absolute_scores, dtype=np.float64)
        rel_scores = np.asarray(relation_scores, dtype=np.float64)
        self.abs_median, self.abs_iqr = _median_iqr(abs_scores)
        self.rel_median, self.rel_iqr = _median_iqr(rel_scores)
        return self

    def score(self, absolute_scores: np.ndarray, relation_scores: np.ndarray) -> np.ndarray:
        abs_scores = (np.asarray(absolute_scores, dtype=np.float64) - self.abs_median) / (self.abs_iqr + EPS)
        rel_scores = (np.asarray(relation_scores, dtype=np.float64) - self.rel_median) / (self.rel_iqr + EPS)
        return self.alpha * abs_scores + (1.0 - self.alpha) * rel_scores


def _median_iqr(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)), float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
