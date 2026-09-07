from __future__ import annotations

import numpy as np


def residual_relation(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    mode: str = "signed_residual",
) -> np.ndarray:
    chunks = []
    for pair_name in ("iq_ap", "iq_stft", "ap_stft"):
        left, right = pairs[pair_name]
        diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
        if mode == "signed_residual":
            chunks.append(diff)
        elif mode == "absolute_residual":
            chunks.append(np.abs(diff))
        else:
            raise ValueError("residual mode must be 'signed_residual' or 'absolute_residual'")
    return np.concatenate(chunks, axis=1)


def distance_relation(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    mode: str,
) -> np.ndarray:
    values = []
    for pair_name in ("iq_ap", "iq_stft", "ap_stft"):
        left, right = pairs[pair_name]
        if mode == "cosine":
            values.append(cosine_distance(left, right)[:, np.newaxis])
        elif mode == "euclidean":
            values.append(euclidean_distance(left, right)[:, np.newaxis])
        else:
            raise ValueError("distance mode must be 'cosine' or 'euclidean'")
    return np.concatenate(values, axis=1)


def cosine_distance(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    numerator = np.sum(x * y, axis=1)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return 1.0 - numerator / np.maximum(denominator, eps)


def euclidean_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    return np.linalg.norm(x - y, axis=1) / np.sqrt(max(x.shape[1], 1))
