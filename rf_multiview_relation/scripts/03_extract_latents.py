from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_multiview_relation.data.dataset import read_manifest, split_manifest_path  # noqa: E402
from rf_multiview_relation.data.mat import load_iq_from_mat, validate_iq  # noqa: E402
from rf_multiview_relation.data.representations import build_all_views  # noqa: E402
from rf_multiview_relation.data.sigmf import discover_sigmf_data_files, sigmf_memmap  # noqa: E402
from rf_multiview_relation.data.windowing import window_start_positions  # noqa: E402
from rf_multiview_relation.pipeline import choose_device, encoder_config, load_encoders, resolve_project_path  # noqa: E402
from rf_multiview_relation.utils.config import load_config  # noqa: E402


DEFAULT_SPLITS = (
    "tx1_train",
    "tx1_calibration",
    "tx1_holdout",
    "tx2",
    "tx3",
    "tx4",
    "tx5",
    "tx6",
    "tx7",
    "tx8",
    "oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 latent extraction with frozen Tx1-trained encoders.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-files-per-split", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def maybe_with_run_name(path: Path, run_name: str) -> Path:
    return path / run_name if run_name else path


def relative_file_id(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def split_files(split_name: str, data_root: Path, output_dir: Path, config: dict) -> tuple[str, list[Path]]:
    lower = split_name.lower()
    seed = int(config["seed"])
    if lower in {"tx1_train", "tx1_calibration", "tx1_holdout"}:
        manifest = split_manifest_path(output_dir, lower, seed)
        return "Tx1", read_manifest(manifest, data_root)
    if lower.startswith("tx") and lower[2:].isdigit():
        device = f"Tx{int(lower[2:])}"
        return device, sorted((data_root / device).glob("*.mat"), key=lambda p: p.name.lower())
    if lower == "oracle":
        return "Oracle", discover_sigmf_data_files(data_root / str(config["data"]["oracle_dir"]))
    raise ValueError(f"Unsupported split: {split_name}")


def mat_windows(path: Path, config: dict) -> Iterator[tuple[int, int, np.ndarray]]:
    data_cfg = config["data"]
    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg["stride"])
    max_windows = data_cfg.get("max_windows_per_file")
    max_windows = int(max_windows) if max_windows is not None else None
    iq = load_iq_from_mat(path, key=str(data_cfg["mat_key"]))
    validate_iq(iq, min_len=window_size)
    starts = window_start_positions(iq.shape[0], window_size, stride, max_windows)
    for window_id, start in enumerate(starts):
        start = int(start)
        yield int(window_id), start, iq[start : start + window_size]


def oracle_windows(path: Path, config: dict) -> Iterator[tuple[int, int, np.ndarray]]:
    data_cfg = config["data"]
    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg["stride"])
    max_windows = data_cfg.get("max_windows_per_file")
    max_windows = int(max_windows) if max_windows is not None else None
    complex_samples = sigmf_memmap(path)
    starts = window_start_positions(complex_samples.shape[0], window_size, stride, max_windows)
    for window_id, start in enumerate(starts):
        start = int(start)
        segment = np.asarray(complex_samples[start : start + window_size])
        iq = np.stack([segment.real, segment.imag], axis=1).astype(np.float32)
        validate_iq(iq, min_len=window_size)
        yield int(window_id), start, iq


def encode_split(
    split_name: str,
    split_device: str,
    files: list[Path],
    data_root: Path,
    config: dict,
    encoders: dict[str, torch.nn.Module],
    batch_size: int,
    device: torch.device,
    output_path: Path,
    max_files: int | None,
) -> None:
    selected_files = files[: int(max_files)] if max_files is not None else files
    latent_chunks = {"iq": [], "ap": [], "stft": []}
    meta_file_ids: list[str] = []
    meta_window_ids: list[int] = []
    meta_window_starts: list[int] = []
    meta_devices: list[str] = []
    meta_splits: list[str] = []
    batch_views: dict[str, list[np.ndarray]] = {"iq": [], "ap": [], "stft": []}
    batch_file_ids: list[str] = []
    batch_window_ids: list[int] = []
    batch_window_starts: list[int] = []
    batch_devices: list[str] = []
    batch_splits: list[str] = []

    def flush() -> None:
        if not batch_file_ids:
            return
        with torch.no_grad():
            for view in ("iq", "ap", "stft"):
                tensor = torch.from_numpy(np.stack(batch_views[view], axis=0).astype(np.float32)).to(device)
                z = encoders[view](tensor).detach().cpu().numpy().astype(np.float32)
                latent_chunks[view].append(z)
        meta_file_ids.extend(batch_file_ids)
        meta_window_ids.extend(batch_window_ids)
        meta_window_starts.extend(batch_window_starts)
        meta_devices.extend(batch_devices)
        meta_splits.extend(batch_splits)
        for values in batch_views.values():
            values.clear()
        batch_file_ids.clear()
        batch_window_ids.clear()
        batch_window_starts.clear()
        batch_devices.clear()
        batch_splits.clear()

    for file_idx, path in enumerate(selected_files, start=1):
        iterator = oracle_windows(path, config) if split_device == "Oracle" else mat_windows(path, config)
        file_id = relative_file_id(path, data_root)
        for window_id, window_start, iq_window in iterator:
            views = build_all_views(iq_window, config)
            for view in ("iq", "ap", "stft"):
                batch_views[view].append(views[view])
            batch_file_ids.append(file_id)
            batch_window_ids.append(window_id)
            batch_window_starts.append(window_start)
            batch_devices.append(split_device)
            batch_splits.append(split_name)
            if len(batch_file_ids) >= batch_size:
                flush()
        if file_idx % 25 == 0 or file_idx == len(selected_files):
            print(f"[LATENT] split={split_name} files={file_idx}/{len(selected_files)}")
    flush()

    if not meta_file_ids:
        raise RuntimeError(f"No windows extracted for split={split_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        file_id=np.asarray(meta_file_ids, dtype=str),
        window_id=np.asarray(meta_window_ids, dtype=np.int64),
        window_start=np.asarray(meta_window_starts, dtype=np.int64),
        device=np.asarray(meta_devices, dtype=str),
        split=np.asarray(meta_splits, dtype=str),
        z_iq=np.concatenate(latent_chunks["iq"], axis=0),
        z_ap=np.concatenate(latent_chunks["ap"], axis=0),
        z_stft=np.concatenate(latent_chunks["stft"], axis=0),
    )
    print(
        f"[DONE] split={split_name} files={len(selected_files)} windows={len(meta_file_ids)} "
        f"wrote={output_path}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths_cfg = config["paths"]
    data_root = resolve_project_path(args.data_root or paths_cfg["data_root"], PROJECT_ROOT)
    output_dir = resolve_project_path(args.output_dir or paths_cfg["outputs"], PROJECT_ROOT)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir or (output_dir / "checkpoints"), PROJECT_ROOT)
    latent_dir = resolve_project_path(args.latent_dir or (output_dir / "latents"), PROJECT_ROOT)
    checkpoint_dir = maybe_with_run_name(checkpoint_dir, args.run_name)
    latent_dir = maybe_with_run_name(latent_dir, args.run_name)
    batch_size = int(args.batch_size or encoder_config(config)["batch_size"])
    device = choose_device(args.device)
    encoders = load_encoders(checkpoint_dir, config, device)

    print(
        f"[LATENT] device={device} splits={','.join(args.splits)} batch_size={batch_size} "
        f"checkpoint_dir={checkpoint_dir}"
    )
    for split_name in args.splits:
        split_device, files = split_files(split_name, data_root, output_dir, config)
        encode_split(
            split_name=split_name.lower(),
            split_device=split_device,
            files=files,
            data_root=data_root,
            config=config,
            encoders=encoders,
            batch_size=batch_size,
            device=device,
            output_path=latent_dir / f"{split_name.lower()}.npz",
            max_files=args.max_files_per_split,
        )


if __name__ == "__main__":
    main()
