from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_multiview_relation.pipeline import (  # noqa: E402
    PAIR_KEYS,
    aggregate_file_scores,
    assert_no_file_leakage,
    assert_tx1_only_fit,
    fit_cca_models,
    load_latent_dir,
    resolve_project_path,
    threshold_from_file_rows,
)
from rf_multiview_relation.relation.cka import linear_cka  # noqa: E402
from rf_multiview_relation.relation.relation_features import cosine_distance, euclidean_distance  # noqa: E402
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402
from rf_multiview_relation.utils.metrics import binary_metrics, roc_auc_score  # noqa: E402


EVAL_SPLITS = ("tx1_holdout", "tx2", "tx3", "tx4", "tx5", "tx6", "tx7", "tx8", "oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tx1-only relation screening followed by held-out anomaly evaluation.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-screen-samples", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def maybe_with_run_name(path: Path, run_name: str) -> Path:
    return path / run_name if run_name else path


def sample_latents(latents: dict[str, np.ndarray], max_samples: int | None, seed: int) -> dict[str, np.ndarray]:
    if max_samples is None or latents["iq"].shape[0] <= int(max_samples):
        return latents
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(latents["iq"].shape[0], size=int(max_samples), replace=False))
    sampled = dict(latents)
    for key in ("file_id", "window_id", "window_start", "device", "split", "iq", "ap", "stft"):
        sampled[key] = latents[key][indices]
    return sampled


def candidate_list() -> list[dict[str, str]]:
    rows = []
    for pair_name, _, _ in PAIR_KEYS:
        for source in ("raw", "cca"):
            for metric in ("cosine", "l2"):
                rows.append({"candidate": f"{pair_name}_{source}_{metric}", "pair": pair_name, "source": source, "metric": metric})
        rows.append({"candidate": f"{pair_name}_cca_abs_mean", "pair": pair_name, "source": "cca", "metric": "abs_mean"})
    return rows


def candidate_window_scores(
    latents: dict[str, np.ndarray],
    cca_models: dict[str, Any],
    candidate: dict[str, str],
) -> np.ndarray:
    pair_name = candidate["pair"]
    _, left_name, right_name = next(item for item in PAIR_KEYS if item[0] == pair_name)
    left = latents[left_name]
    right = latents[right_name]
    if candidate["source"] == "cca":
        left, right = cca_models[pair_name].transform(left, right)
    metric = candidate["metric"]
    if metric == "cosine":
        return cosine_distance(left, right)
    if metric == "l2":
        return euclidean_distance(left, right)
    if metric == "abs_mean":
        return np.mean(np.abs(left - right), axis=1)
    raise ValueError(f"Unsupported metric: {metric}")


def file_score_values(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["file_score"]) for row in rows], dtype=np.float64)


def median_iqr(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    return float(np.median(data)), float(np.percentile(data, 75) - np.percentile(data, 25))


def rank01(values: list[float], higher_is_better: bool = True) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 1:
        return [1.0]
    order = np.argsort(data)
    ranks = np.empty(data.size, dtype=np.float64)
    ranks[order] = np.arange(data.size, dtype=np.float64)
    for value in np.unique(data):
        mask = data == value
        if np.sum(mask) > 1:
            ranks[mask] = float(np.mean(ranks[mask]))
    normalized = ranks / max(data.size - 1, 1)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return normalized.tolist()


def add_selection_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cka_rank = rank01([float(row["tx1_train_cka"]) for row in rows], higher_is_better=True)
    cca_rank = rank01([float(row["tx1_train_cca_mean_corr"]) for row in rows], higher_is_better=True)
    shift_rank = rank01([float(row["median_shift_norm"]) for row in rows], higher_is_better=False)
    iqr_rank = rank01([float(row["iqr_ratio_log_abs"]) for row in rows], higher_is_better=False)
    for idx, row in enumerate(rows):
        score = 0.40 * cka_rank[idx] + 0.30 * cca_rank[idx] + 0.20 * shift_rank[idx] + 0.10 * iqr_rank[idx]
        row["screen_cka_rank"] = cka_rank[idx]
        row["screen_cca_rank"] = cca_rank[idx]
        row["screen_stability_shift_rank"] = shift_rank[idx]
        row["screen_stability_iqr_rank"] = iqr_rank[idx]
        row["tx1_only_screen_score"] = float(score)
    return sorted(rows, key=lambda item: float(item["tx1_only_screen_score"]), reverse=True)


def screening_rows(
    train_latents: dict[str, np.ndarray],
    calibration_latents: dict[str, np.ndarray],
    cca_models: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    percentile = float(config["file_scoring"]["percentile"])
    for candidate in candidate_list():
        pair_name = candidate["pair"]
        _, left_name, right_name = next(item for item in PAIR_KEYS if item[0] == pair_name)
        train_scores = candidate_window_scores(train_latents, cca_models, candidate)
        calibration_scores = candidate_window_scores(calibration_latents, cca_models, candidate)
        train_rows = aggregate_file_scores(
            train_latents["file_id"],
            train_latents["device"],
            train_latents["split"],
            train_scores,
            candidate["candidate"],
            threshold=None,
            percentile=percentile,
        )
        calibration_rows = aggregate_file_scores(
            calibration_latents["file_id"],
            calibration_latents["device"],
            calibration_latents["split"],
            calibration_scores,
            candidate["candidate"],
            threshold=None,
            percentile=percentile,
        )
        train_values = file_score_values(train_rows)
        calibration_values = file_score_values(calibration_rows)
        train_median, train_iqr = median_iqr(train_values)
        calibration_median, calibration_iqr = median_iqr(calibration_values)
        rows.append(
            {
                **candidate,
                "tx1_train_cka": float(linear_cka(train_latents[left_name], train_latents[right_name])),
                "tx1_train_cca_mean_corr": float(np.mean(cca_models[pair_name].correlations_))
                if candidate["source"] == "cca"
                else 0.0,
                "tx1_train_files": int(train_values.size),
                "tx1_calibration_files": int(calibration_values.size),
                "tx1_train_median": train_median,
                "tx1_train_iqr": train_iqr,
                "tx1_calibration_median": calibration_median,
                "tx1_calibration_iqr": calibration_iqr,
                "median_shift_norm": abs(calibration_median - train_median) / max(train_iqr, 1e-8),
                "iqr_ratio_log_abs": abs(np.log((calibration_iqr + 1e-8) / (train_iqr + 1e-8))),
            }
        )
    return add_selection_scores(rows)


def metric_row(rows: list[dict[str, Any]], candidate: str, evaluation: str, threshold: float) -> dict[str, Any] | None:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["file_score"]) for row in rows], dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return {
        "candidate": candidate,
        "evaluation": evaluation,
        "normal_files": int(np.sum(labels == 0)),
        "anomaly_files": int(np.sum(labels == 1)),
        "threshold": float(threshold),
        **binary_metrics(labels, scores, threshold),
    }


def cliffs_delta(anomaly: np.ndarray, normal: np.ndarray) -> float:
    x = np.asarray(anomaly, dtype=np.float64)
    y = np.asarray(normal, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diffs = x[:, np.newaxis] - y[np.newaxis, :]
    return float((np.sum(diffs > 0.0) - np.sum(diffs < 0.0)) / diffs.size)


def p_value(normal: np.ndarray, anomaly: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
    except ModuleNotFoundError:
        return float("nan")
    return float(mannwhitneyu(normal, anomaly, alternative="two-sided").pvalue)


def evaluate_candidates(
    selected: list[dict[str, Any]],
    latent_sets: dict[str, dict[str, np.ndarray]],
    cca_models: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    percentile = float(config["file_scoring"]["percentile"])
    threshold_percentile = float(config["threshold"]["percentile"])
    all_file_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []

    for candidate in selected:
        calibration = latent_sets["tx1_calibration"]
        calibration_scores = candidate_window_scores(calibration, cca_models, candidate)
        calibration_rows = aggregate_file_scores(
            calibration["file_id"],
            calibration["device"],
            calibration["split"],
            calibration_scores,
            candidate["candidate"],
            threshold=None,
            percentile=percentile,
        )
        threshold = threshold_from_file_rows(calibration_rows, threshold_percentile)
        file_rows = []
        for split in EVAL_SPLITS:
            if split not in latent_sets:
                continue
            latents = latent_sets[split]
            scores = candidate_window_scores(latents, cca_models, candidate)
            rows = aggregate_file_scores(
                latents["file_id"],
                latents["device"],
                latents["split"],
                scores,
                candidate["candidate"],
                threshold=threshold,
                percentile=percentile,
            )
            file_rows.extend(rows)
            all_file_rows.extend(rows)

        closed_splits = {"tx1_holdout"} | {f"tx{idx}" for idx in range(2, 9)}
        closed_rows = [row for row in file_rows if row["split"] in closed_splits]
        result = metric_row(closed_rows, candidate["candidate"], "closed_tx2_tx8", threshold)
        if result is not None:
            main_rows.append({**candidate, **result})
        oracle_rows = [row for row in file_rows if row["split"] in {"tx1_holdout", "oracle"}]
        result = metric_row(oracle_rows, candidate["candidate"], "external_oracle", threshold)
        if result is not None:
            main_rows.append({**candidate, **result})
        minimal_rows = [row for row in file_rows if row["split"] in {"tx1_holdout", "tx2"}]
        result = metric_row(minimal_rows, candidate["candidate"], "minimal_tx2", threshold)
        if result is not None:
            main_rows.append({**candidate, **result})

        normal = np.asarray(
            [float(row["file_score"]) for row in file_rows if row["split"] == "tx1_holdout"],
            dtype=np.float64,
        )
        for idx in range(2, 9):
            split = f"tx{idx}"
            rows = [row for row in file_rows if row["split"] in {"tx1_holdout", split}]
            result = metric_row(rows, candidate["candidate"], f"tx1_vs_tx{idx}", threshold)
            if result is not None:
                device_rows.append({**candidate, "anomaly": f"Tx{idx}", **result})
            anomaly = np.asarray(
                [float(row["file_score"]) for row in file_rows if row["split"] == split],
                dtype=np.float64,
            )
            if normal.size and anomaly.size:
                stat_rows.append(
                    {
                        **candidate,
                        "comparison": f"Tx1_holdout_vs_Tx{idx}",
                        "tx1_median_score": float(np.median(normal)),
                        "anomaly_median_score": float(np.median(anomaly)),
                        "p_value": p_value(normal, anomaly),
                        "cliffs_delta_anomaly_minus_tx1": cliffs_delta(anomaly, normal),
                        "directional_auroc": float(
                            roc_auc_score(
                                np.concatenate([np.zeros(normal.size, dtype=np.int64), np.ones(anomaly.size, dtype=np.int64)]),
                                np.concatenate([normal, anomaly]),
                            )
                        ),
                    }
                )
    return all_file_rows, main_rows, device_rows, stat_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_project_path(args.output_dir or config["paths"]["outputs"], PROJECT_ROOT)
    latent_dir = maybe_with_run_name(resolve_project_path(args.latent_dir or output_dir / "latents", PROJECT_ROOT), args.run_name)
    tables_dir = maybe_with_run_name(output_dir / "tables", args.run_name)
    scores_dir = maybe_with_run_name(output_dir / "scores", args.run_name)
    latent_sets = load_latent_dir(latent_dir)
    assert_no_file_leakage(latent_sets)

    train_latents = sample_latents(latent_sets["tx1_train"], args.max_screen_samples, int(config["seed"]))
    calibration_latents = latent_sets["tx1_calibration"]
    assert_tx1_only_fit(train_latents, "tx1_train")
    assert_tx1_only_fit(calibration_latents, "tx1_calibration")
    cca_models = fit_cca_models(train_latents, config)
    rows = screening_rows(train_latents, calibration_latents, cca_models, config)
    selected = rows[: int(args.top_k)]

    file_rows, main_rows, device_rows, stat_rows = evaluate_candidates(selected, latent_sets, cca_models, config)
    write_csv(tables_dir / "tx1_relation_screening.csv", rows)
    write_csv(tables_dir / "tx1_relation_screening_selected.csv", selected)
    write_csv(scores_dir / "tx1_screened_relation_file_scores.csv", file_rows)
    write_csv(tables_dir / "tx1_screened_relation_results.csv", main_rows)
    write_csv(tables_dir / "tx1_screened_relation_device_results.csv", device_rows)
    write_csv(tables_dir / "tx1_screened_relation_statistical_tests.csv", stat_rows)

    print(f"[DONE] wrote {tables_dir / 'tx1_relation_screening.csv'}")
    print(f"[DONE] wrote {tables_dir / 'tx1_screened_relation_results.csv'}")
    print("[SCREEN] top candidates from Tx1-only criteria:")
    for row in selected:
        print(
            f"[SCREEN] {row['candidate']} score={float(row['tx1_only_screen_score']):.4f} "
            f"cka={float(row['tx1_train_cka']):.4f} cca={float(row['tx1_train_cca_mean_corr']):.4f} "
            f"shift={float(row['median_shift_norm']):.4f}"
        )


if __name__ == "__main__":
    main()
