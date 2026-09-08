from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_multiview_relation.data.mat import load_iq_from_mat, validate_iq  # noqa: E402
from rf_multiview_relation.data.representations import energy_normalize_iq_window, slope_channel  # noqa: E402
from rf_multiview_relation.data.sigmf import sigmf_memmap  # noqa: E402
from rf_multiview_relation.data.windowing import window_start_positions  # noqa: E402
from rf_multiview_relation.pipeline import (  # noqa: E402
    aggregate_file_scores,
    assert_no_file_leakage,
    assert_tx1_only_fit,
    fit_cca_models,
    load_latent_dir,
    resolve_project_path,
)
from rf_multiview_relation.relation.relation_features import cosine_distance, euclidean_distance  # noqa: E402
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402
from rf_multiview_relation.utils.metrics import binary_metrics, roc_auc_score  # noqa: E402


EVAL_SPLITS = ("tx1_holdout", "tx2", "tx3", "tx4", "tx5", "tx6", "tx7", "tx8", "oracle")
PAIRING_SPLITS = ("tx1_train", "tx1_calibration", *EVAL_SPLITS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AP-STFT CCA score differences by device.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--phase-max-files-per-split", type=int, default=120)
    parser.add_argument("--phase-max-windows-per-file", type=int, default=64)
    parser.add_argument("--skip-phase-slope", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


def maybe_with_run_name(path: Path, run_name: str) -> Path:
    return path / run_name if run_name else path


def percentile_name(config: dict[str, Any]) -> float:
    return float(config["file_scoring"]["percentile"])


def ap_stft_projections(latents: dict[str, np.ndarray], cca_models: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return cca_models["ap_stft"].transform(latents["ap"], latents["stft"])


def ap_stft_window_scores(
    latents: dict[str, np.ndarray],
    cca_models: dict[str, Any],
    metric: str = "cosine",
    stft_permutation: np.ndarray | None = None,
) -> np.ndarray:
    ap_proj, stft_proj = ap_stft_projections(latents, cca_models)
    if stft_permutation is not None:
        stft_proj = stft_proj[np.asarray(stft_permutation, dtype=np.int64)]
    if metric == "cosine":
        return cosine_distance(ap_proj, stft_proj)
    if metric == "l2":
        return euclidean_distance(ap_proj, stft_proj)
    if metric == "abs_mean":
        return np.mean(np.abs(ap_proj - stft_proj), axis=1)
    raise ValueError(f"Unsupported AP-STFT metric: {metric}")


def file_scores_for(
    latents: dict[str, np.ndarray],
    scores: np.ndarray,
    method: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    return aggregate_file_scores(
        latents["file_id"],
        latents["device"],
        latents["split"],
        scores,
        method,
        threshold=None,
        percentile=percentile_name(config),
    )


def values(rows: Iterable[dict[str, Any]], key: str = "file_score") -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def stats(prefix: str, data: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_iqr": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
    }


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diffs = x[:, np.newaxis] - y[np.newaxis, :]
    return float((np.sum(diffs > 0.0) - np.sum(diffs < 0.0)) / diffs.size)


def p_value(left: np.ndarray, right: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
    except ModuleNotFoundError:
        return float("nan")
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)


def device_diagnosis_rows(
    latent_sets: dict[str, dict[str, np.ndarray]],
    cca_models: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, latents in latent_sets.items():
        scores = ap_stft_window_scores(latents, cca_models, "cosine")
        by_split_rows[split] = file_scores_for(latents, scores, "ap_stft_cca_cosine_direct", config)

    normal_rows = by_split_rows["tx1_holdout"]
    normal_scores = values(normal_rows)
    output_rows = []
    for split in EVAL_SPLITS:
        if split not in by_split_rows:
            continue
        rows = by_split_rows[split]
        split_scores = values(rows)
        device = rows[0]["device"] if rows else split
        row = {
            "split": split,
            "device": device,
            **stats("score", split_scores),
            "tx1_holdout_median_delta": float(np.median(split_scores) - np.median(normal_scores)),
            "tx1_holdout_median_ratio": float(np.median(split_scores) / max(np.median(normal_scores), 1e-12)),
        }
        if split != "tx1_holdout":
            labels = np.concatenate([np.zeros(normal_scores.size, dtype=np.int64), np.ones(split_scores.size, dtype=np.int64)])
            merged = np.concatenate([normal_scores, split_scores])
            row.update(
                {
                    "AUROC_vs_tx1_holdout": float(roc_auc_score(labels, merged)),
                    "mannwhitney_p": p_value(normal_scores, split_scores),
                    "cliffs_delta_anomaly_minus_tx1": cliffs_delta(split_scores, normal_scores),
                }
            )
        output_rows.append(row)
    return output_rows, by_split_rows


def component_rows(
    latent_sets: dict[str, dict[str, np.ndarray]],
    cca_models: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    train_ap, train_stft = ap_stft_projections(latent_sets["tx1_train"], cca_models)
    train_abs = np.abs(train_ap - train_stft)
    train_mean = train_abs.mean(axis=0)
    train_std = train_abs.std(axis=0) + 1e-12

    normal_ap, normal_stft = ap_stft_projections(latent_sets["tx1_holdout"], cca_models)
    normal_abs = np.abs(normal_ap - normal_stft)
    normal_component_medians = component_file_medians(latent_sets["tx1_holdout"], normal_abs, config)
    normal_median = np.median(normal_component_medians, axis=0)

    rows = []
    for split in EVAL_SPLITS:
        if split not in latent_sets:
            continue
        latents = latent_sets[split]
        ap_proj, stft_proj = ap_stft_projections(latents, cca_models)
        residual = ap_proj - stft_proj
        abs_residual = np.abs(residual)
        component_medians = component_file_medians(latents, abs_residual, config)
        split_median = np.median(component_medians, axis=0)
        shift = split_median - normal_median
        abs_shift = np.abs(shift)
        total = float(np.sum(abs_shift)) + 1e-12
        signed_mean = np.mean(residual, axis=0)
        for idx in range(abs_residual.shape[1]):
            rows.append(
                {
                    "split": split,
                    "device": str(latents["device"][0]),
                    "component": idx + 1,
                    "train_abs_residual_mean": float(train_mean[idx]),
                    "train_abs_residual_std": float(train_std[idx]),
                    "tx1_holdout_file_median_abs_residual": float(normal_median[idx]),
                    "device_file_median_abs_residual": float(split_median[idx]),
                    "device_minus_tx1_median_abs_residual": float(shift[idx]),
                    "standardized_shift_vs_tx1_train": float((split_median[idx] - train_mean[idx]) / train_std[idx]),
                    "absolute_shift_share": float(abs_shift[idx] / total),
                    "device_signed_residual_mean": float(signed_mean[idx]),
                }
            )
    return rows


def component_summary_rows(component_shift_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    devices = []
    for row in component_shift_rows:
        device = str(row["device"])
        if device not in devices:
            devices.append(device)
    for device in devices:
        if device == "Tx1":
            continue
        device_rows = [row for row in component_shift_rows if str(row["device"]) == device]
        if not device_rows:
            continue
        total_shift = sum(abs(float(row["device_minus_tx1_median_abs_residual"])) for row in device_rows)
        top = sorted(device_rows, key=lambda row: float(row["absolute_shift_share"]), reverse=True)[:5]
        rows.append(
            {
                "device": device,
                "split": str(device_rows[0]["split"]),
                "total_abs_component_shift": float(total_shift),
                "top_components": ";".join(f"c{row['component']}" for row in top),
                "top_component_shares": ";".join(f"{float(row['absolute_shift_share']):.6f}" for row in top),
                "top_component_shifts": ";".join(f"{float(row['device_minus_tx1_median_abs_residual']):.6f}" for row in top),
            }
        )
    return rows


def component_file_medians(
    latents: dict[str, np.ndarray],
    component_scores: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    file_ids = latents["file_id"].astype(str)
    unique_ids = list(dict.fromkeys(file_ids.tolist()))
    percentile = percentile_name(config)
    rows = []
    for file_id in unique_ids:
        mask = file_ids == file_id
        if str(percentile).lower() == "mean":
            rows.append(np.mean(component_scores[mask], axis=0))
        else:
            rows.append(np.percentile(component_scores[mask], percentile, axis=0))
    return np.asarray(rows, dtype=np.float64)


def pairing_validation_rows(
    latent_sets: dict[str, dict[str, np.ndarray]],
    cca_models: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    rows = []
    for split in PAIRING_SPLITS:
        if split not in latent_sets:
            continue
        latents = latent_sets[split]
        true_scores = ap_stft_window_scores(latents, cca_models, "cosine")
        permutation = rng.permutation(true_scores.size)
        shuffled_scores = ap_stft_window_scores(latents, cca_models, "cosine", stft_permutation=permutation)
        true_rows = file_scores_for(latents, true_scores, "true_pair", config)
        shuffled_rows = file_scores_for(latents, shuffled_scores, "shuffled_pair", config)
        true_values = values(true_rows)
        shuffled_values = values(shuffled_rows)
        labels = np.concatenate([np.zeros(true_values.size, dtype=np.int64), np.ones(shuffled_values.size, dtype=np.int64)])
        merged = np.concatenate([true_values, shuffled_values])
        rows.append(
            {
                "split": split,
                "device": str(latents["device"][0]),
                **stats("true", true_values),
                **stats("shuffled", shuffled_values),
                "shuffled_minus_true_median": float(np.median(shuffled_values) - np.median(true_values)),
                "shuffled_over_true_median": float(np.median(shuffled_values) / max(np.median(true_values), 1e-12)),
                "paired_relation_auc_true_vs_shuffled": float(roc_auc_score(labels, merged)),
                "mannwhitney_p_true_vs_shuffled": p_value(true_values, shuffled_values),
                "cliffs_delta_shuffled_minus_true": cliffs_delta(shuffled_values, true_values),
            }
        )
    return rows


def rankdata(values_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(values_array, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(1, arr.size + 1)
    for value in np.unique(arr):
        mask = arr == value
        if np.sum(mask) > 1:
            ranks[mask] = float(np.mean(ranks[mask]))
    return ranks


def corr_or_nan(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    x = rankdata(left) if rank else np.asarray(left, dtype=np.float64)
    y = rankdata(right) if rank else np.asarray(right, dtype=np.float64)
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mat_phase_slope_rows(
    data_root: Path,
    file_ids: list[str],
    split: str,
    device: str,
    config: dict[str, Any],
    max_files: int,
    max_windows_per_file: int,
) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    rows = []
    for file_id in file_ids[: int(max_files)]:
        path = data_root / file_id
        if not path.exists():
            continue
        iq = load_iq_from_mat(path, key=str(data_cfg["mat_key"]))
        validate_iq(iq, min_len=int(data_cfg["window_size"]))
        starts = window_start_positions(
            iq.shape[0],
            int(data_cfg["window_size"]),
            int(data_cfg["stride"]),
            data_cfg.get("max_windows_per_file"),
        )
        if starts.size > int(max_windows_per_file):
            starts = starts[np.linspace(0, starts.size - 1, int(max_windows_per_file), dtype=np.int64)]
        slopes = []
        for start in starts:
            window = energy_normalize_iq_window(iq[int(start) : int(start) + int(data_cfg["window_size"])])
            phase = np.unwrap(np.arctan2(window[:, 1], window[:, 0]))
            slopes.append(float(slope_channel(phase, unit=True)[0]))
        slope_values = np.asarray(slopes, dtype=np.float64)
        rows.append(
            {
                "split": split,
                "device": device,
                "file_id": file_id,
                "checked_windows": int(slope_values.size),
                "phase_slope_unit_mean": float(np.mean(slope_values)),
                "phase_slope_unit_median": float(np.median(slope_values)),
                "phase_slope_unit_std": float(np.std(slope_values)),
                "phase_abs_slope_unit_mean": float(np.mean(np.abs(slope_values))),
                "phase_abs_slope_unit_p60": float(np.percentile(np.abs(slope_values), 60)),
                "phase_abs_slope_unit_p90": float(np.percentile(np.abs(slope_values), 90)),
            }
        )
    return rows


def oracle_phase_slope_rows(
    data_root: Path,
    file_ids: list[str],
    split: str,
    device: str,
    config: dict[str, Any],
    max_files: int,
    max_windows_per_file: int,
) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    rows = []
    for file_id in file_ids[: int(max_files)]:
        path = data_root / file_id
        if not path.exists():
            continue
        samples = sigmf_memmap(path)
        starts = window_start_positions(
            samples.shape[0],
            int(data_cfg["window_size"]),
            int(data_cfg["stride"]),
            data_cfg.get("max_windows_per_file"),
        )
        if starts.size > int(max_windows_per_file):
            starts = starts[np.linspace(0, starts.size - 1, int(max_windows_per_file), dtype=np.int64)]
        slopes = []
        for start in starts:
            segment = np.asarray(samples[int(start) : int(start) + int(data_cfg["window_size"])])
            iq = np.stack([segment.real, segment.imag], axis=1).astype(np.float32)
            window = energy_normalize_iq_window(iq)
            phase = np.unwrap(np.arctan2(window[:, 1], window[:, 0]))
            slopes.append(float(slope_channel(phase, unit=True)[0]))
        slope_values = np.asarray(slopes, dtype=np.float64)
        rows.append(
            {
                "split": split,
                "device": device,
                "file_id": file_id,
                "checked_windows": int(slope_values.size),
                "phase_slope_unit_mean": float(np.mean(slope_values)),
                "phase_slope_unit_median": float(np.median(slope_values)),
                "phase_slope_unit_std": float(np.std(slope_values)),
                "phase_abs_slope_unit_mean": float(np.mean(np.abs(slope_values))),
                "phase_abs_slope_unit_p60": float(np.percentile(np.abs(slope_values), 60)),
                "phase_abs_slope_unit_p90": float(np.percentile(np.abs(slope_values), 90)),
            }
        )
    return rows


def phase_slope_analysis_rows(
    latent_sets: dict[str, dict[str, np.ndarray]],
    file_rows_by_split: dict[str, list[dict[str, Any]]],
    data_root: Path,
    config: dict[str, Any],
    max_files: int,
    max_windows_per_file: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    file_score_by_id = {
        row["file_id"]: float(row["file_score"])
        for rows in file_rows_by_split.values()
        for row in rows
    }
    slope_rows = []
    for split in EVAL_SPLITS:
        if split not in latent_sets:
            continue
        latents = latent_sets[split]
        file_ids = sorted(set(latents["file_id"].astype(str).tolist()))
        device = str(latents["device"][0])
        if device == "Oracle":
            split_rows = oracle_phase_slope_rows(data_root, file_ids, split, device, config, max_files, max_windows_per_file)
        else:
            split_rows = mat_phase_slope_rows(data_root, file_ids, split, device, config, max_files, max_windows_per_file)
        for row in split_rows:
            row["ap_stft_cca_cosine_file_score"] = file_score_by_id.get(row["file_id"], float("nan"))
        slope_rows.extend(split_rows)

    tx1_values = np.asarray(
        [float(row["phase_abs_slope_unit_p60"]) for row in slope_rows if row["split"] == "tx1_holdout"],
        dtype=np.float64,
    )
    device_rows = []
    corr_rows = []
    for split in EVAL_SPLITS:
        rows = [row for row in slope_rows if row["split"] == split]
        if not rows:
            continue
        slope_values = np.asarray([float(row["phase_abs_slope_unit_p60"]) for row in rows], dtype=np.float64)
        score_values = np.asarray([float(row["ap_stft_cca_cosine_file_score"]) for row in rows], dtype=np.float64)
        valid = np.isfinite(score_values)
        device_rows.append(
            {
                "split": split,
                "device": rows[0]["device"],
                "files": len(rows),
                **stats("phase_abs_slope_unit_p60", slope_values),
                **stats("ap_stft_score", score_values[valid]),
                "phase_abs_slope_median_minus_tx1": float(np.median(slope_values) - np.median(tx1_values)) if tx1_values.size else float("nan"),
            }
        )
        corr_rows.append(
            {
                "split": split,
                "device": rows[0]["device"],
                "files": int(np.sum(valid)),
                "pearson_score_vs_abs_slope_p60": corr_or_nan(score_values[valid], slope_values[valid]),
                "spearman_score_vs_abs_slope_p60": corr_or_nan(score_values[valid], slope_values[valid], rank=True),
            }
        )
    all_valid_rows = [row for row in slope_rows if np.isfinite(float(row["ap_stft_cca_cosine_file_score"]))]
    all_scores = np.asarray([float(row["ap_stft_cca_cosine_file_score"]) for row in all_valid_rows], dtype=np.float64)
    all_slopes = np.asarray([float(row["phase_abs_slope_unit_p60"]) for row in all_valid_rows], dtype=np.float64)
    corr_rows.append(
        {
            "split": "all_evaluated",
            "device": "mixed",
            "files": len(all_valid_rows),
            "pearson_score_vs_abs_slope_p60": corr_or_nan(all_scores, all_slopes),
            "spearman_score_vs_abs_slope_p60": corr_or_nan(all_scores, all_slopes, rank=True),
        }
    )
    return slope_rows, device_rows, corr_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_project_path(args.output_dir or config["paths"]["outputs"], PROJECT_ROOT)
    data_root = resolve_project_path(config["paths"]["data_root"], PROJECT_ROOT)
    latent_dir = maybe_with_run_name(resolve_project_path(args.latent_dir or output_dir / "latents", PROJECT_ROOT), args.run_name)
    tables_dir = maybe_with_run_name(output_dir / "tables", args.run_name)
    tables_dir.mkdir(parents=True, exist_ok=True)

    latent_sets = load_latent_dir(latent_dir)
    assert_no_file_leakage(latent_sets)
    assert_tx1_only_fit(latent_sets["tx1_train"], "tx1_train")
    cca_models = fit_cca_models(latent_sets["tx1_train"], config)

    diagnosis_rows, file_rows_by_split = device_diagnosis_rows(latent_sets, cca_models, config)
    comp_rows = component_rows(latent_sets, cca_models, config)
    comp_summary_rows = component_summary_rows(comp_rows)
    pairing_rows = pairing_validation_rows(latent_sets, cca_models, config, int(args.shuffle_seed))

    write_csv(tables_dir / "ap_stft_device_diagnosis.csv", diagnosis_rows)
    write_csv(tables_dir / "ap_stft_component_shifts.csv", comp_rows)
    write_csv(tables_dir / "ap_stft_component_device_summary.csv", comp_summary_rows)
    write_csv(tables_dir / "ap_stft_pairing_validation.csv", pairing_rows)

    if not args.skip_phase_slope:
        slope_rows, slope_device_rows, slope_corr_rows = phase_slope_analysis_rows(
            latent_sets=latent_sets,
            file_rows_by_split=file_rows_by_split,
            data_root=data_root,
            config=config,
            max_files=int(args.phase_max_files_per_split),
            max_windows_per_file=int(args.phase_max_windows_per_file),
        )
        write_csv(tables_dir / "ap_stft_phase_slope_file_stats.csv", slope_rows)
        write_csv(tables_dir / "ap_stft_phase_slope_device_summary.csv", slope_device_rows)
        write_csv(tables_dir / "ap_stft_phase_slope_score_correlation.csv", slope_corr_rows)

    print(f"[DONE] wrote {tables_dir / 'ap_stft_device_diagnosis.csv'}")
    print(f"[DONE] wrote {tables_dir / 'ap_stft_component_shifts.csv'}")
    print(f"[DONE] wrote {tables_dir / 'ap_stft_component_device_summary.csv'}")
    print(f"[DONE] wrote {tables_dir / 'ap_stft_pairing_validation.csv'}")
    if not args.skip_phase_slope:
        print(f"[DONE] wrote {tables_dir / 'ap_stft_phase_slope_device_summary.csv'}")
        print(f"[DONE] wrote {tables_dir / 'ap_stft_phase_slope_score_correlation.csv'}")
    for row in diagnosis_rows:
        if row["split"] != "tx1_holdout":
            print(
                "[DIAG] "
                f"{row['device']} AUROC={float(row['AUROC_vs_tx1_holdout']):.4f} "
                f"median_delta={float(row['tx1_holdout_median_delta']):.4f} "
                f"delta={float(row['cliffs_delta_anomaly_minus_tx1']):.4f}"
            )
    for row in pairing_rows:
        if row["split"] in {"tx1_holdout", "tx2", "tx8", "oracle"}:
            print(
                "[PAIR] "
                f"{row['split']} gap={float(row['shuffled_minus_true_median']):.4f} "
                f"auc={float(row['paired_relation_auc_true_vs_shuffled']):.4f}"
            )


if __name__ == "__main__":
    main()
