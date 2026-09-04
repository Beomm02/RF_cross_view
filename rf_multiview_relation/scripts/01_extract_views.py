from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_2ND = PROJECT_ROOT / "code" / "2nd"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CODE_2ND) not in sys.path:
    sys.path.insert(0, str(CODE_2ND))

from preprocessing import load_iq_from_mat, validate_iq  # noqa: E402
from rf_multiview_relation.data.representations import build_all_views  # noqa: E402
from rf_multiview_relation.data.windowing import window_start_positions  # noqa: E402
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402
from rf_multiview_relation.utils.plotting import (  # noqa: E402
    save_ap_examples,
    save_iq_examples,
    save_stft_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 IQ/AP/STFT representation sanity extraction.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--split-manifest", default=None)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_manifest(path: Path, data_root: Path) -> list[Path]:
    files = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        candidate = Path(line)
        files.append(candidate if candidate.is_absolute() else data_root / candidate)
    return files


def default_train_manifest(output_dir: Path, seed: int) -> Path:
    candidates = sorted((output_dir / "splits").glob(f"tx1_train_*_seed{seed}.txt"))
    if candidates:
        return candidates[0]
    return output_dir / "splits" / f"tx1_train_320_seed{seed}.txt"


def stat_row(sample_index: int, file_path: Path, window_start: int, views: dict[str, np.ndarray], data_root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_index": sample_index,
        "file_id": file_path.name,
        "relative_path": file_path.relative_to(data_root).as_posix(),
        "window_start": int(window_start),
    }
    for name, array in views.items():
        row[f"{name}_shape"] = "x".join(str(dim) for dim in array.shape)
        row[f"{name}_min"] = float(np.min(array))
        row[f"{name}_max"] = float(np.max(array))
        row[f"{name}_mean"] = float(np.mean(array))
        row[f"{name}_std"] = float(np.std(array))
        row[f"{name}_nan_count"] = int(np.isnan(array).sum())
        row[f"{name}_inf_count"] = int(np.isinf(array).sum())
    return row


def load_example_files(args: argparse.Namespace, cfg: dict, data_root: Path, output_dir: Path) -> list[Path]:
    if args.split_manifest:
        manifest = resolve_project_path(args.split_manifest)
        if not manifest.exists():
            raise FileNotFoundError(f"split manifest not found: {manifest}")
        return read_manifest(manifest, data_root)

    manifest = default_train_manifest(output_dir, int(cfg["seed"]))
    if manifest.exists():
        return read_manifest(manifest, data_root)

    normal_device = str(cfg["data"]["normal_device"])
    fallback = sorted((data_root / normal_device).glob("*.mat"), key=lambda p: p.name.lower())
    print(f"[WARN] split manifest not found; falling back to sorted {normal_device} files")
    return fallback


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]
    data_root = resolve_project_path(args.data_root or paths_cfg["data_root"])
    output_dir = resolve_project_path(args.output_dir or paths_cfg["outputs"])
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    files = load_example_files(args, cfg, data_root, output_dir)
    if not files:
        raise FileNotFoundError("No Tx1 files available for representation examples")

    num_examples = min(int(args.num_examples), len(files))
    examples: dict[str, list[np.ndarray]] = {"iq": [], "ap": [], "stft": []}
    rows = []
    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg["stride"])
    max_windows = data_cfg.get("max_windows_per_file")
    max_windows = int(max_windows) if max_windows is not None else None

    for sample_index, file_path in enumerate(files[:num_examples]):
        iq = load_iq_from_mat(str(file_path), key=str(data_cfg["mat_key"]))
        validate_iq(iq, min_len=window_size)
        starts = window_start_positions(iq.shape[0], window_size, stride, max_windows)
        if starts.size == 0:
            raise ValueError(f"No window can be created from {file_path}")
        window_start = int(starts[0])
        window = iq[window_start : window_start + window_size]
        views = build_all_views(window, cfg)
        for name in examples:
            examples[name].append(views[name])
        rows.append(stat_row(sample_index, file_path, window_start, views, data_root))

    save_iq_examples(examples["iq"], figures_dir / "example_iq.png")
    save_ap_examples(examples["ap"], figures_dir / "example_ap.png")
    save_stft_examples(examples["stft"], figures_dir / "example_stft.png")
    write_csv(tables_dir / "representation_examples.csv", rows)

    first = rows[0]
    print(f"[REP] examples={num_examples}")
    print(f"[REP] IQ shape={first['iq_shape']} AP shape={first['ap_shape']} STFT shape={first['stft_shape']}")
    print(f"[REP] wrote {figures_dir / 'example_iq.png'}")
    print(f"[REP] wrote {figures_dir / 'example_ap.png'}")
    print(f"[REP] wrote {figures_dir / 'example_stft.png'}")
    print(f"[REP] wrote {tables_dir / 'representation_examples.csv'}")


if __name__ == "__main__":
    main()
