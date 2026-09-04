from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def discover_sigmf_data_files(oracle_root: str | Path) -> list[Path]:
    root = Path(oracle_root)
    return sorted(root.rglob("*.sigmf-data"), key=lambda p: str(p).lower())


def discover_nonstandard_sigmf_like_files(oracle_root: str | Path) -> list[Path]:
    root = Path(oracle_root)
    paths = []
    for path in root.rglob("*"):
        if path.is_file() and ".sigmf-data" in path.name and path.suffix != ".sigmf-data":
            paths.append(path)
    return sorted(paths, key=lambda p: str(p).lower())


def read_sigmf_meta(data_path: str | Path) -> dict[str, Any]:
    data_path = Path(data_path)
    meta_path = data_path.with_suffix(".sigmf-meta")
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metadata_payload(meta: dict[str, Any]) -> dict[str, Any]:
    return meta.get("_metadata", meta) if isinstance(meta, dict) else {}


def meta_sample_count(meta: dict[str, Any]) -> int | None:
    payload = metadata_payload(meta)
    annotations = payload.get("annotations") or []
    if annotations:
        try:
            value = annotations[0].get("core:sample_count")
            return int(value) if value else None
        except (TypeError, ValueError):
            return None
    return None


def meta_sample_rate(meta: dict[str, Any]) -> float | None:
    payload = metadata_payload(meta)
    global_meta = payload.get("global") or {}
    for key in ("core:sample_rate", "sample_rate", "fs"):
        if key in global_meta:
            try:
                return float(global_meta[key])
            except (TypeError, ValueError):
                return None
    return None


def infer_sigmf_dtype_and_count(data_path: str | Path) -> tuple[np.dtype, int, int | None]:
    path = Path(data_path)
    size = path.stat().st_size
    meta_count = None
    try:
        meta_count = meta_sample_count(read_sigmf_meta(path))
    except (OSError, json.JSONDecodeError):
        meta_count = None

    complex128_bytes = np.dtype(np.complex128).itemsize
    complex64_bytes = np.dtype(np.complex64).itemsize

    if meta_count and size == meta_count * complex128_bytes:
        return np.dtype(np.complex128), int(meta_count), int(meta_count)
    if meta_count and size == meta_count * complex64_bytes:
        return np.dtype(np.complex64), int(meta_count), int(meta_count)
    if size % complex128_bytes == 0:
        return np.dtype(np.complex128), int(size // complex128_bytes), meta_count
    if size % complex64_bytes == 0:
        return np.dtype(np.complex64), int(size // complex64_bytes), meta_count
    raise ValueError(f"Unsupported SigMF byte length for {path}: {size}")


def sigmf_memmap(data_path: str | Path) -> np.memmap:
    dtype, count, _ = infer_sigmf_dtype_and_count(data_path)
    return np.memmap(data_path, dtype=dtype, mode="r", shape=(count,))


def oracle_file_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    distance_match = re.search(r"_(\d+ft)_run", path.name)
    if distance_match is None:
        distance_match = re.search(r"[\\/](\d+ft)[\\/]", str(path))
    device_match = re.search(r"X310_([^_]+)_\d+ft_run(\d+)", path.name)
    return {
        "oracle_device": device_match.group(1) if device_match else "unknown",
        "distance": distance_match.group(1) if distance_match else "unknown",
        "run": int(device_match.group(2)) if device_match else -1,
    }


def sigmf_metadata_summary(data_path: str | Path) -> dict[str, Any]:
    try:
        meta = read_sigmf_meta(data_path)
    except (OSError, json.JSONDecodeError):
        return {
            "meta_exists": Path(data_path).with_suffix(".sigmf-meta").exists(),
            "meta_readable": False,
            "meta_sample_count": None,
            "sample_rate": None,
        }

    return {
        "meta_exists": bool(meta),
        "meta_readable": True,
        "meta_sample_count": meta_sample_count(meta),
        "sample_rate": meta_sample_rate(meta),
    }
