from __future__ import annotations

import numpy as np


def relation_only_features(relation_vector: np.ndarray) -> np.ndarray:
    return np.asarray(relation_vector, dtype=np.float64)
