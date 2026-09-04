from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_2ND = PROJECT_ROOT / "code" / "2nd"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CODE_2ND) not in sys.path:
    sys.path.insert(0, str(CODE_2ND))

from preprocessing import load_iq_from_mat, normalize_iq, validate_iq  # noqa: E402
from rf_multiview_relation.data.sigmf import (  # noqa: E402
    discover_nonstandard_sigmf_like_files,
    discover_sigmf_data_files,
    infer_sigmf_dtype_and_count,
    oracle_file_info,
    sigmf_metadata_summary,
)
from rf_multiview_relation.data.splits import (  # noqa: E402
    assert_disjoint,
    split_tx1_files,
    write_split_manifest,
)
from rf_multiview_relation.data.windowing import (  # noqa: E402
    count_possible_windows,
    selected_window_count,
)
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 dataset audit for Tx1-only multi-view relation study.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--full-value-check", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def mat_field_info(path: Path, mat_key: str) -> dict[str, Any]:
    fields = whosmat(path)
    for name, shape, matlab_class in fields:
        if name == mat_key:
            length = int(np.prod(shape))
            return {
                "mat_key_found": True,
                "shape": "x".join(str(int(v)) for v in shape),
                "length": length,
                "matlab_class": str(matlab_class),
            }
    return {
        "mat_key_found": False,
        "shape": "",
        "length": 0,
        "matlab_class": "",
    }


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    try:
        return str(item.item())
    except AttributeError:
        return str(item)


def mat_metadata(path: Path) -> dict[str, Any]:
    keys = ["fs_hw", "fc", "targetDevice", "bootSession"]
    try:
        mat = loadmat(path, variable_names=keys, squeeze_me=True, chars_as_strings=True)
    except Exception as exc:  # noqa: BLE001
        return {"metadata_error": str(exc)}
    return {key: scalar_text(mat[key]) for key in keys if key in mat}


def audit_value_files(
    files: list[Path],
    mat_key: str,
    window_size: int,
    norm_mode: str,
    full_value_check: bool,
    sample_count: int,
) -> dict[str, Any]:
    selected = files if full_value_check else files[: int(sample_count)]
    errors = []
    raw_nan_inf_files = []
    normalized_nan_inf_files = []
    dtypes = set()
    lengths = []

    for path in selected:
        try:
            iq = load_iq_from_mat(str(path), key=mat_key)
            dtypes.add(str(iq.dtype))
            lengths.append(int(iq.shape[0]))
            validate_iq(iq, min_len=window_size)
            if np.isnan(iq).any() or np.isinf(iq).any():
                raw_nan_inf_files.append(path.name)
            normalized = normalize_iq(iq, mode=norm_mode)
            if np.isnan(normalized).any() or np.isinf(normalized).any():
                normalized_nan_inf_files.append(path.name)
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": path.name, "error": str(exc)})

    return {
        "checked_files": len(selected),
        "full_value_check": bool(full_value_check),
        "iq_dtypes_after_load": sorted(dtypes),
        "min_loaded_length": min(lengths) if lengths else None,
        "max_loaded_length": max(lengths) if lengths else None,
        "raw_nan_inf_files": raw_nan_inf_files,
        "normalized_nan_inf_files": normalized_nan_inf_files,
        "errors": errors,
    }


def audit_mat_device(
    data_root: Path,
    device: str,
    mat_key: str,
    window_size: int,
    stride: int,
    max_windows_per_file: int | None,
    norm_mode: str,
    full_value_check: bool,
    value_check_files_per_device: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device_dir = data_root / device
    files = sorted(device_dir.glob("*.mat"), key=lambda p: p.name.lower())
    file_rows = []
    lengths = []
    possible_counts = []
    selected_counts = []
    matlab_classes = set()
    shapes = set()
    field_errors = []

    for path in files:
        try:
            field = mat_field_info(path, mat_key)
            length = int(field["length"])
            possible = count_possible_windows(length, window_size, stride)
            selected = selected_window_count(length, window_size, stride, max_windows_per_file)
            lengths.append(length)
            possible_counts.append(possible)
            selected_counts.append(selected)
            matlab_classes.add(field["matlab_class"])
            shapes.add(field["shape"])
            file_rows.append(
                {
                    "device": device,
                    "file_id": path.name,
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "mat_key_found": field["mat_key_found"],
                    "shape": field["shape"],
                    "length": length,
                    "matlab_class": field["matlab_class"],
                    "possible_windows": possible,
                    "selected_windows": selected,
                }
            )
        except Exception as exc:  # noqa: BLE001
            field_errors.append({"file": path.name, "error": str(exc)})

    metadata = mat_metadata(files[0]) if files else {}
    value_audit = audit_value_files(
        files=files,
        mat_key=mat_key,
        window_size=window_size,
        norm_mode=norm_mode,
        full_value_check=full_value_check,
        sample_count=value_check_files_per_device,
    )
    summary = {
        "device": device,
        "directory": device_dir.relative_to(PROJECT_ROOT).as_posix() if device_dir.exists() else str(device_dir),
        "file_count": len(files),
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "unique_shapes": sorted(shapes),
        "matlab_classes": sorted(matlab_classes),
        "min_possible_windows": min(possible_counts) if possible_counts else None,
        "max_possible_windows": max(possible_counts) if possible_counts else None,
        "min_selected_windows": min(selected_counts) if selected_counts else None,
        "max_selected_windows": max(selected_counts) if selected_counts else None,
        "total_selected_windows": int(sum(selected_counts)),
        "metadata_sample": metadata,
        "value_audit": value_audit,
        "field_errors": field_errors,
    }
    return summary, file_rows


def audit_oracle(
    data_root: Path,
    oracle_dir: str,
    window_size: int,
    stride: int,
    max_windows_per_file: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    oracle_root = data_root / oracle_dir
    files = discover_sigmf_data_files(oracle_root)
    nonstandard = discover_nonstandard_sigmf_like_files(oracle_root)
    zip_files = sorted(oracle_root.rglob("*.zip"), key=lambda p: str(p).lower())
    rows = []
    errors = []
    sample_rates = set()
    dtype_counts: dict[str, int] = {}
    distances: dict[str, int] = {}
    oracle_devices: dict[str, int] = {}
    possible_counts = []
    selected_counts = []

    for path in files:
        try:
            dtype, sample_count, meta_count = infer_sigmf_dtype_and_count(path)
            metadata = sigmf_metadata_summary(path)
            if metadata.get("sample_rate") is not None:
                sample_rates.add(str(metadata["sample_rate"]))
            info = oracle_file_info(path)
            distance = str(info["distance"])
            oracle_device = str(info["oracle_device"])
            distances[distance] = distances.get(distance, 0) + 1
            oracle_devices[oracle_device] = oracle_devices.get(oracle_device, 0) + 1
            dtype_name = str(dtype)
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            possible = count_possible_windows(sample_count, window_size, stride)
            selected = selected_window_count(sample_count, window_size, stride, max_windows_per_file)
            possible_counts.append(possible)
            selected_counts.append(selected)
            rows.append(
                {
                    "file_id": path.name,
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "oracle_device": oracle_device,
                    "distance": distance,
                    "run": info["run"],
                    "dtype": dtype_name,
                    "byte_size": path.stat().st_size,
                    "sample_count": sample_count,
                    "meta_sample_count": meta_count,
                    "sample_rate": metadata.get("sample_rate"),
                    "meta_exists": metadata.get("meta_exists"),
                    "meta_readable": metadata.get("meta_readable"),
                    "possible_windows": possible,
                    "selected_windows": selected,
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": path.name, "error": str(exc)})

    summary = {
        "directory": oracle_root.relative_to(PROJECT_ROOT).as_posix() if oracle_root.exists() else str(oracle_root),
        "standard_sigmf_data_count": len(files),
        "nonstandard_sigmf_like_count": len(nonstandard),
        "nonstandard_sigmf_like_files": [p.relative_to(data_root).as_posix() for p in nonstandard],
        "zip_count": len(zip_files),
        "dtype_counts": dtype_counts,
        "sample_rates": sorted(sample_rates),
        "distance_counts": dict(sorted(distances.items())),
        "oracle_device_counts": dict(sorted(oracle_devices.items())),
        "min_possible_windows": min(possible_counts) if possible_counts else None,
        "max_possible_windows": max(possible_counts) if possible_counts else None,
        "min_selected_windows": min(selected_counts) if selected_counts else None,
        "max_selected_windows": max(selected_counts) if selected_counts else None,
        "total_selected_windows": int(sum(selected_counts)),
        "errors": errors,
    }
    return summary, rows


def write_tx1_splits(
    tx1_files: list[Path],
    cfg: dict[str, Any],
    data_root: Path,
    splits_dir: Path,
) -> dict[str, Any]:
    seed = int(cfg["seed"])
    data_cfg = cfg["data"]
    groups = split_tx1_files(
        files=tx1_files,
        seed=seed,
        train_total=int(data_cfg["tx1_train_files"]),
        fit_count=int(data_cfg["tx1_fit_files"]),
        calibration_count=int(data_cfg["tx1_calibration_files"]),
        holdout_count=int(data_cfg["tx1_holdout_files"]),
    )
    write_split_manifest(groups["tx1_train"], splits_dir / f"tx1_train_{len(groups['tx1_train'])}_seed{seed}.txt", data_root)
    write_split_manifest(
        groups["tx1_calibration"],
        splits_dir / f"tx1_calibration_{len(groups['tx1_calibration'])}_seed{seed}.txt",
        data_root,
    )
    write_split_manifest(
        groups["tx1_holdout"],
        splits_dir / f"tx1_holdout_{len(groups['tx1_holdout'])}_seed{seed}.txt",
        data_root,
    )
    write_split_manifest(
        groups["tx1_train_all"],
        splits_dir / f"tx1_train_all_{len(groups['tx1_train_all'])}_seed{seed}.txt",
        data_root,
    )
    return {
        "seed": seed,
        "tx1_train": len(groups["tx1_train"]),
        "tx1_calibration": len(groups["tx1_calibration"]),
        "tx1_holdout": len(groups["tx1_holdout"]),
        "tx1_train_all": len(groups["tx1_train_all"]),
        "manifest_dir": splits_dir.relative_to(PROJECT_ROOT).as_posix(),
        "leakage_check": "passed",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]
    data_root = resolve_project_path(args.data_root or paths_cfg["data_root"])
    output_dir = resolve_project_path(args.output_dir or paths_cfg["outputs"])
    tables_dir = output_dir / "tables"
    splits_dir = output_dir / "splits"

    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg["stride"])
    max_windows_per_file = data_cfg.get("max_windows_per_file")
    if max_windows_per_file is not None:
        max_windows_per_file = int(max_windows_per_file)
    mat_key = str(data_cfg["mat_key"])
    normal_device = str(data_cfg["normal_device"])
    anomaly_devices = [str(device) for device in data_cfg["anomaly_devices"]]
    devices = [normal_device] + anomaly_devices

    device_summaries = []
    all_mat_file_rows = []
    tx_file_map: dict[str, list[Path]] = {}
    for device in devices:
        summary, rows = audit_mat_device(
            data_root=data_root,
            device=device,
            mat_key=mat_key,
            window_size=window_size,
            stride=stride,
            max_windows_per_file=max_windows_per_file,
            norm_mode=str(data_cfg["norm_mode"]),
            full_value_check=bool(args.full_value_check),
            value_check_files_per_device=int(data_cfg["value_check_files_per_device"]),
        )
        device_summaries.append(summary)
        all_mat_file_rows.extend(rows)
        tx_file_map[device] = [data_root / row["relative_path"] for row in rows]

    tx1_split_summary = write_tx1_splits(tx_file_map[normal_device], cfg, data_root, splits_dir)
    anomaly_groups = {device: tx_file_map[device] for device in anomaly_devices}
    assert_disjoint(
        {
            "tx1_train": tx_file_map[normal_device][:0]
            + [data_root / line.strip() for line in (splits_dir / f"tx1_train_{tx1_split_summary['tx1_train']}_seed{cfg['seed']}.txt").read_text(encoding="utf-8").splitlines() if line.strip()],
            "tx1_calibration": [data_root / line.strip() for line in (splits_dir / f"tx1_calibration_{tx1_split_summary['tx1_calibration']}_seed{cfg['seed']}.txt").read_text(encoding="utf-8").splitlines() if line.strip()],
            "tx1_holdout": [data_root / line.strip() for line in (splits_dir / f"tx1_holdout_{tx1_split_summary['tx1_holdout']}_seed{cfg['seed']}.txt").read_text(encoding="utf-8").splitlines() if line.strip()],
            **anomaly_groups,
        }
    )

    oracle_summary, oracle_rows = audit_oracle(
        data_root=data_root,
        oracle_dir=str(data_cfg["oracle_dir"]),
        window_size=window_size,
        stride=stride,
        max_windows_per_file=max_windows_per_file,
    )

    device_table_rows = []
    for summary in device_summaries:
        value_audit = summary["value_audit"]
        metadata = summary["metadata_sample"]
        device_table_rows.append(
            {
                "device": summary["device"],
                "file_count": summary["file_count"],
                "min_length": summary["min_length"],
                "max_length": summary["max_length"],
                "unique_shapes": ";".join(summary["unique_shapes"]),
                "matlab_classes": ";".join(summary["matlab_classes"]),
                "min_possible_windows": summary["min_possible_windows"],
                "max_possible_windows": summary["max_possible_windows"],
                "min_selected_windows": summary["min_selected_windows"],
                "max_selected_windows": summary["max_selected_windows"],
                "total_selected_windows": summary["total_selected_windows"],
                "sample_fs_hw": metadata.get("fs_hw", ""),
                "sample_fc": metadata.get("fc", ""),
                "sample_targetDevice": metadata.get("targetDevice", ""),
                "checked_files": value_audit["checked_files"],
                "raw_nan_inf_count": len(value_audit["raw_nan_inf_files"]),
                "normalized_nan_inf_count": len(value_audit["normalized_nan_inf_files"]),
                "value_error_count": len(value_audit["errors"]),
                "field_error_count": len(summary["field_errors"]),
            }
        )

    write_csv(tables_dir / "dataset_audit_devices.csv", device_table_rows)
    write_csv(tables_dir / "dataset_audit_mat_files.csv", all_mat_file_rows)
    write_csv(tables_dir / "dataset_audit_oracle.csv", oracle_rows)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(data_root),
        "config_path": str(Path(args.config).resolve()),
        "config": cfg,
        "mat_devices": device_summaries,
        "tx1_split": tx1_split_summary,
        "oracle": oracle_summary,
        "outputs": {
            "dataset_audit_json": str(output_dir / "dataset_audit.json"),
            "device_table": str(tables_dir / "dataset_audit_devices.csv"),
            "mat_file_table": str(tables_dir / "dataset_audit_mat_files.csv"),
            "oracle_table": str(tables_dir / "dataset_audit_oracle.csv"),
        },
    }
    write_json(output_dir / "dataset_audit.json", audit)

    print(f"[AUDIT] data_root={data_root}")
    for row in device_table_rows:
        print(
            "[AUDIT] "
            f"{row['device']}: files={row['file_count']} "
            f"length={row['min_length']}..{row['max_length']} "
            f"windows={row['min_possible_windows']}..{row['max_possible_windows']} "
            f"selected_per_file={row['min_selected_windows']}..{row['max_selected_windows']} "
            f"checked={row['checked_files']} value_errors={row['value_error_count']}"
        )
    print(
        "[AUDIT] Tx1 split: "
        f"train={tx1_split_summary['tx1_train']} "
        f"calibration={tx1_split_summary['tx1_calibration']} "
        f"holdout={tx1_split_summary['tx1_holdout']} leakage={tx1_split_summary['leakage_check']}"
    )
    print(
        "[AUDIT] Oracle: "
        f"standard_sigmf={oracle_summary['standard_sigmf_data_count']} "
        f"nonstandard_sigmf_like={oracle_summary['nonstandard_sigmf_like_count']} "
        f"zip={oracle_summary['zip_count']} "
        f"selected_windows_total={oracle_summary['total_selected_windows']}"
    )
    print(f"[DONE] wrote {output_dir / 'dataset_audit.json'}")


if __name__ == "__main__":
    main()
