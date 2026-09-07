from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_iq_from_mat(mat_path: str | Path, key: str = "rxData") -> np.ndarray:
    path = Path(mat_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    mat = loadmat(path)
    if key not in mat:
        raise KeyError(f"'{key}' not found in {path}")

    rx = np.squeeze(mat[key])
    if not np.iscomplexobj(rx):
        raise ValueError(f"{path} -> '{key}' is not complex data")
    if rx.ndim != 1:
        rx = rx.reshape(-1)
    return np.stack([rx.real, rx.imag], axis=1).astype(np.float32)


def validate_iq(iq: np.ndarray, min_len: int) -> None:
    if not isinstance(iq, np.ndarray):
        raise TypeError("iq must be numpy.ndarray")
    if iq.ndim != 2 or iq.shape[1] != 2:
        raise ValueError("iq must have shape [N, 2]")
    if np.isnan(iq).any() or np.isinf(iq).any():
        raise ValueError("iq contains NaN/Inf")
    if iq.shape[0] < int(min_len):
        raise ValueError(f"iq length is smaller than required minimum ({int(min_len)})")


def normalize_iq(iq: np.ndarray, mode: str = "power") -> np.ndarray:
    x = np.asarray(iq, dtype=np.float32).copy()
    if mode == "none":
        return x
    if mode == "zscore":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        return (x - mean) / std
    if mode == "minmax":
        x_min = x.min(axis=0, keepdims=True)
        x_max = x.max(axis=0, keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-8)
    if mode == "power":
        power = np.mean(np.sum(x * x, axis=1), keepdims=True)
        return x / np.sqrt(power + 1e-8)
    if mode == "power_dc":
        x = x - x.mean(axis=0, keepdims=True)
        power = np.mean(np.sum(x * x, axis=1), keepdims=True)
        return x / np.sqrt(power + 1e-8)
    if mode == "dc_only":
        return x - x.mean(axis=0, keepdims=True)
    raise ValueError("mode must be 'none', 'zscore', 'minmax', 'power', 'power_dc', or 'dc_only'")
