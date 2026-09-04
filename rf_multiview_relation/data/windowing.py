from __future__ import annotations

import numpy as np


def count_possible_windows(n_samples: int, window_size: int, stride: int) -> int:
    n_samples = int(n_samples)
    window_size = int(window_size)
    stride = int(stride)
    if n_samples < window_size:
        return 0
    return 1 + (n_samples - window_size) // stride


def selected_window_indices(possible_windows: int, max_windows_per_file: int | None) -> np.ndarray:
    possible_windows = int(possible_windows)
    if possible_windows <= 0:
        return np.empty((0,), dtype=np.int64)
    if max_windows_per_file is None or possible_windows <= int(max_windows_per_file):
        return np.arange(possible_windows, dtype=np.int64)
    return np.linspace(0, possible_windows - 1, int(max_windows_per_file), dtype=np.int64)


def selected_window_count(
    n_samples: int,
    window_size: int,
    stride: int,
    max_windows_per_file: int | None,
) -> int:
    possible = count_possible_windows(n_samples, window_size, stride)
    return int(selected_window_indices(possible, max_windows_per_file).size)


def window_start_positions(
    n_samples: int,
    window_size: int,
    stride: int,
    max_windows_per_file: int | None,
) -> np.ndarray:
    possible = count_possible_windows(n_samples, window_size, stride)
    indices = selected_window_indices(possible, max_windows_per_file)
    return indices * int(stride)
