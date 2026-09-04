from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_2ND = PROJECT_ROOT / "code" / "2nd"
if str(CODE_2ND) not in sys.path:
    sys.path.insert(0, str(CODE_2ND))

from preprocessing import load_iq_from_mat, validate_iq  # noqa: E402
from rf_multiview_relation.data.representations import build_view  # noqa: E402
from rf_multiview_relation.data.windowing import window_start_positions  # noqa: E402


def read_manifest(path: str | Path, data_root: str | Path) -> list[Path]:
    manifest = Path(path)
    root = Path(data_root)
    files = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        candidate = Path(line)
        files.append(candidate if candidate.is_absolute() else root / candidate)
    return files


def split_manifest_path(output_dir: str | Path, split_name: str, seed: int) -> Path:
    split_dir = Path(output_dir) / "splits"
    matches = sorted(split_dir.glob(f"{split_name}_*_seed{int(seed)}.txt"))
    if not matches:
        raise FileNotFoundError(f"No split manifest found for {split_name} seed={seed} under {split_dir}")
    return matches[0]


def iter_view_batches(
    files: list[Path],
    view_name: str,
    config: dict,
    batch_size: int,
    rng: np.random.Generator,
    shuffle_files: bool = True,
    shuffle_windows: bool = True,
    max_files: int | None = None,
) -> Iterator[np.ndarray]:
    selected_files = list(files)
    if max_files is not None:
        selected_files = selected_files[: int(max_files)]
    file_indices = np.arange(len(selected_files))
    if shuffle_files:
        rng.shuffle(file_indices)

    data_cfg = config["data"]
    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg["stride"])
    max_windows = data_cfg.get("max_windows_per_file")
    max_windows = int(max_windows) if max_windows is not None else None
    mat_key = str(data_cfg["mat_key"])

    batch = []
    for file_idx in file_indices:
        path = selected_files[int(file_idx)]
        iq = load_iq_from_mat(str(path), key=mat_key)
        validate_iq(iq, min_len=window_size)
        starts = window_start_positions(iq.shape[0], window_size, stride, max_windows)
        if shuffle_windows:
            rng.shuffle(starts)
        for start in starts:
            window = iq[int(start) : int(start) + window_size]
            batch.append(build_view(window, view_name, config))
            if len(batch) == int(batch_size):
                yield np.stack(batch, axis=0).astype(np.float32)
                batch = []
    if batch:
        yield np.stack(batch, axis=0).astype(np.float32)


def count_selected_windows_for_files(files: list[Path], config: dict, max_files: int | None = None) -> int:
    selected_files = list(files)
    if max_files is not None:
        selected_files = selected_files[: int(max_files)]
    data_cfg = config["data"]
    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg["stride"])
    max_windows = data_cfg.get("max_windows_per_file")
    max_windows = int(max_windows) if max_windows is not None else None
    mat_key = str(data_cfg["mat_key"])
    total = 0
    for path in selected_files:
        iq = load_iq_from_mat(str(path), key=mat_key)
        validate_iq(iq, min_len=window_size)
        starts = window_start_positions(iq.shape[0], window_size, stride, max_windows)
        total += int(starts.size)
    return total
