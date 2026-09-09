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
    aggregate_file_scores,
    assert_no_file_leakage,
    assert_tx1_only_fit,
    fit_cca_models,
    fit_mahalanobis,
    load_latent_dir,
    resolve_project_path,
    threshold_from_file_rows,
)
from rf_multiview_relation.relation.relation_features import cosine_distance  # noqa: E402
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402
from rf_multiview_relation.utils.metrics import binary_metrics  # noqa: E402


EPS = 1e-8
EVAL_SPLITS = ("tx1_holdout", "tx2", "tx3", "tx4", "tx5", "tx6", "tx7", "tx8", "oracle")
ALL_SCORE_SPLITS = ("tx1_calibration", *EVAL_SPLITS)

RELATION_METHOD = "ap_stft_cca_cosine_direct"
MARGINAL_METHODS = (
    "ap_latent_md",
    "stft_latent_md",
    "ap_cca_marginal_md",
    "stft_cca_marginal_md",
    "ap_stft_cca_marginal_sum",
)
CONTROL_METHODS = (
    *MARGINAL_METHODS,
    "ap_stft_raw_cosine",
    "ap_stft_cca_joint_md",
    "ap_stft_cca_residual_md",
    "ap_stft_cca_abs_residual_md",
)
DEFAULT_METHODS = (*CONTROL_METHODS, RELATION_METHOD)

DISPLAY_NAMES = {
    "ap_latent_md": "AP latent MD",
    "stft_latent_md": "STFT latent MD",
    "ap_cca_marginal_md": "AP CCA marginal MD",
    "stft_cca_marginal_md": "STFT CCA marginal MD",
    "ap_stft_cca_marginal_sum": "AP+STFT CCA marginal sum",
    "ap_stft_raw_cosine": "AP-STFT raw cosine",
    "ap_stft_cca_joint_md": "AP-STFT CCA joint MD",
    "ap_stft_cca_residual_md": "AP-STFT CCA residual MD",
    "ap_stft_cca_abs_residual_md": "AP-STFT CCA abs residual MD",
    "ap_stft_cca_cosine_direct": "AP-STFT CCA cosine direct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AP-STFT relation against marginal controls.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--methods", nargs="+", default=None)
    return parser.parse_args()


def maybe_with_run_name(path: Path, run_name: str) -> Path:
    return path / run_name if run_name else path


def percentile_value(config: dict[str, Any]) -> float:
    return float(config["file_scoring"]["percentile"])


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


def robust_scale(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    median, iqr = stats
    return (np.asarray(values, dtype=np.float64) - median) / (iqr + EPS)


def rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
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


def ap_stft_projection(latents: dict[str, np.ndarray], state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return state["cca_models"]["ap_stft"].transform(latents["ap"], latents["stft"])


def build_state(train_latents: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    assert_tx1_only_fit(train_latents, "tx1_train")
    cca_models = fit_cca_models(train_latents, config)
    ap_proj, stft_proj = cca_models["ap_stft"].transform(train_latents["ap"], train_latents["stft"])
    residual = ap_proj - stft_proj
    abs_residual = np.abs(residual)
    joint = np.concatenate([ap_proj, stft_proj], axis=1)

    detectors = {
        "ap_latent_md": fit_mahalanobis(train_latents["ap"], config),
        "stft_latent_md": fit_mahalanobis(train_latents["stft"], config),
        "ap_cca_marginal_md": fit_mahalanobis(ap_proj, config),
        "stft_cca_marginal_md": fit_mahalanobis(stft_proj, config),
        "ap_stft_cca_joint_md": fit_mahalanobis(joint, config),
        "ap_stft_cca_residual_md": fit_mahalanobis(residual, config),
        "ap_stft_cca_abs_residual_md": fit_mahalanobis(abs_residual, config),
    }
    ap_scores = detectors["ap_cca_marginal_md"].score(ap_proj)
    stft_scores = detectors["stft_cca_marginal_md"].score(stft_proj)
    return {
        "cca_models": cca_models,
        "detectors": detectors,
        "ap_marginal_stats": robust_stats(ap_scores),
        "stft_marginal_stats": robust_stats(stft_scores),
    }


def score_windows(method: str, latents: dict[str, np.ndarray], state: dict[str, Any]) -> np.ndarray:
    detectors = state["detectors"]
    if method == "ap_latent_md":
        return detectors[method].score(latents["ap"])
    if method == "stft_latent_md":
        return detectors[method].score(latents["stft"])
    if method == "ap_stft_raw_cosine":
        return cosine_distance(latents["ap"], latents["stft"])

    ap_proj, stft_proj = ap_stft_projection(latents, state)
    if method == "ap_cca_marginal_md":
        return detectors[method].score(ap_proj)
    if method == "stft_cca_marginal_md":
        return detectors[method].score(stft_proj)
    if method == "ap_stft_cca_marginal_sum":
        ap_scores = detectors["ap_cca_marginal_md"].score(ap_proj)
        stft_scores = detectors["stft_cca_marginal_md"].score(stft_proj)
        ap_scaled = robust_scale(ap_scores, state["ap_marginal_stats"])
        stft_scaled = robust_scale(stft_scores, state["stft_marginal_stats"])
        return 0.5 * (ap_scaled + stft_scaled)
    if method == "ap_stft_cca_joint_md":
        return detectors[method].score(np.concatenate([ap_proj, stft_proj], axis=1))
    if method == "ap_stft_cca_residual_md":
        return detectors[method].score(ap_proj - stft_proj)
    if method == "ap_stft_cca_abs_residual_md":
        return detectors[method].score(np.abs(ap_proj - stft_proj))
    if method == RELATION_METHOD:
        return cosine_distance(ap_proj, stft_proj)
    raise ValueError(f"Unsupported method: {method}")


def score_all_splits(
    latent_sets: dict[str, dict[str, np.ndarray]],
    state: dict[str, Any],
    methods: list[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    rows_by_method_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for split in ALL_SCORE_SPLITS:
        if split not in latent_sets:
            continue
        latents = latent_sets[split]
        for method in methods:
            scores = score_windows(method, latents, state)
            file_rows = aggregate_file_scores(
                latents["file_id"],
                latents["device"],
                latents["split"],
                scores,
                method,
                threshold=None,
                percentile=percentile_value(config),
            )
            rows_by_method_split[(method, split)] = file_rows
            rows.extend(file_rows)

    thresholds = {}
    percentile = float(config["threshold"]["percentile"])
    for method in methods:
        cal_rows = rows_by_method_split.get((method, "tx1_calibration"), [])
        thresholds[method] = threshold_from_file_rows(cal_rows, percentile)

    for row in rows:
        threshold = thresholds[str(row["method"])]
        row["threshold"] = float(threshold)
        row["prediction"] = int(float(row["file_score"]) > threshold)
    return rows, thresholds


def subset_rows(rows: list[dict[str, Any]], method: str, splits: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["method"]) == method and str(row["split"]) in splits]


def metrics_for_rows(
    rows: list[dict[str, Any]],
    method: str,
    evaluation: str,
    threshold: float,
) -> dict[str, Any] | None:
    if not rows:
        return None
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["file_score"]) for row in rows], dtype=np.float64)
    if np.unique(labels).size < 2:
        return None
    return {
        "evaluation": evaluation,
        "method": method,
        "method_label": DISPLAY_NAMES.get(method, method),
        "normal_files": int(np.sum(labels == 0)),
        "anomaly_files": int(np.sum(labels == 1)),
        "threshold": float(threshold),
        **binary_metrics(labels, scores, threshold),
    }


def build_metric_tables(
    file_rows: list[dict[str, Any]],
    methods: list[str],
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    closed_anomalies = {f"tx{idx}" for idx in range(2, 9)}
    evaluations = {
        "minimal_tx2": {"tx1_holdout", "tx2"},
        "closed_tx2_tx8": {"tx1_holdout", *closed_anomalies},
        "external_oracle": {"tx1_holdout", "oracle"},
    }
    for method in methods:
        threshold = float(thresholds[method])
        for evaluation, splits in evaluations.items():
            result = metrics_for_rows(subset_rows(file_rows, method, splits), method, evaluation, threshold)
            if result is not None:
                main_rows.append(result)
        for idx in range(2, 9):
            split = f"tx{idx}"
            result = metrics_for_rows(
                subset_rows(file_rows, method, {"tx1_holdout", split}),
                method,
                f"tx1_vs_tx{idx}",
                threshold,
            )
            if result is not None:
                result["anomaly"] = f"Tx{idx}"
                device_rows.append(result)
    return main_rows, device_rows


def score_map(rows: list[dict[str, Any]], method: str, splits: set[str]) -> dict[str, tuple[float, int, str]]:
    return {
        str(row["file_id"]): (float(row["file_score"]), int(row["label"]), str(row["split"]))
        for row in rows
        if str(row["method"]) == method and str(row["split"]) in splits
    }


def correlation_rows(file_rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    evaluations = {
        "closed_tx2_tx8": {"tx1_holdout"} | {f"tx{idx}" for idx in range(2, 9)},
        "external_oracle": {"tx1_holdout", "oracle"},
    }
    rows = []
    for evaluation, splits in evaluations.items():
        relation = score_map(file_rows, RELATION_METHOD, splits)
        for method in methods:
            if method == RELATION_METHOD:
                continue
            control = score_map(file_rows, method, splits)
            common = sorted(set(relation) & set(control))
            if not common:
                continue
            rel_scores = np.asarray([relation[file_id][0] for file_id in common], dtype=np.float64)
            ctrl_scores = np.asarray([control[file_id][0] for file_id in common], dtype=np.float64)
            labels = np.asarray([relation[file_id][1] for file_id in common], dtype=np.int64)
            rows.append(
                {
                    "evaluation": evaluation,
                    "control_method": method,
                    "control_label": DISPLAY_NAMES.get(method, method),
                    "files": len(common),
                    "normal_files": int(np.sum(labels == 0)),
                    "anomaly_files": int(np.sum(labels == 1)),
                    "pearson_with_relation": corr_or_nan(ctrl_scores, rel_scores),
                    "spearman_with_relation": corr_or_nan(ctrl_scores, rel_scores, rank=True),
                }
            )
    return rows


def gain_rows(main_rows: list[dict[str, Any]], device_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    all_rows = main_rows + device_rows
    evaluations = []
    for row in all_rows:
        evaluation = str(row["evaluation"])
        if evaluation not in evaluations:
            evaluations.append(evaluation)
    for evaluation in evaluations:
        eval_rows = [row for row in all_rows if str(row["evaluation"]) == evaluation]
        relation = next((row for row in eval_rows if str(row["method"]) == RELATION_METHOD), None)
        if relation is None:
            continue
        marginal = [row for row in eval_rows if str(row["method"]) in MARGINAL_METHODS]
        controls = [row for row in eval_rows if str(row["method"]) in CONTROL_METHODS]
        best_marginal = max(marginal, key=lambda row: float(row["AUROC"])) if marginal else None
        best_control = max(controls, key=lambda row: float(row["AUROC"])) if controls else None
        out = {
            "evaluation": evaluation,
            "relation_AUROC": float(relation["AUROC"]),
            "relation_F1": float(relation["F1"]),
        }
        if "anomaly" in relation:
            out["anomaly"] = relation["anomaly"]
        if best_marginal is not None:
            out.update(
                {
                    "best_marginal_method": best_marginal["method"],
                    "best_marginal_label": best_marginal["method_label"],
                    "best_marginal_AUROC": float(best_marginal["AUROC"]),
                    "relation_minus_best_marginal_AUROC": float(relation["AUROC"]) - float(best_marginal["AUROC"]),
                }
            )
        if best_control is not None:
            out.update(
                {
                    "best_control_method": best_control["method"],
                    "best_control_label": best_control["method_label"],
                    "best_control_AUROC": float(best_control["AUROC"]),
                    "relation_minus_best_control_AUROC": float(relation["AUROC"]) - float(best_control["AUROC"]),
                }
            )
        rows.append(out)
    return rows


def print_summary(main_rows: list[dict[str, Any]], gain_summary: list[dict[str, Any]]) -> None:
    for row in main_rows:
        if row["evaluation"] in {"closed_tx2_tx8", "external_oracle"}:
            print(
                "[RESULT] "
                f"{row['evaluation']} {row['method_label']} "
                f"AUROC={float(row['AUROC']):.4f} F1={float(row['F1']):.4f} "
                f"Recall={float(row['recall']):.4f}"
            )
    for row in gain_summary:
        if row["evaluation"] in {"closed_tx2_tx8", "external_oracle"}:
            print(
                "[GAIN] "
                f"{row['evaluation']} relation_minus_best_marginal_AUROC="
                f"{float(row['relation_minus_best_marginal_AUROC']):.4f} "
                f"best_marginal={row['best_marginal_label']}"
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_project_path(args.output_dir or config["paths"]["outputs"], PROJECT_ROOT)
    latent_dir = maybe_with_run_name(resolve_project_path(args.latent_dir or output_dir / "latents", PROJECT_ROOT), args.run_name)
    tables_dir = maybe_with_run_name(output_dir / "tables", args.run_name)
    tables_dir.mkdir(parents=True, exist_ok=True)

    methods = [str(method) for method in (args.methods or DEFAULT_METHODS)]
    latent_sets = load_latent_dir(latent_dir)
    assert_no_file_leakage(latent_sets)
    state = build_state(latent_sets["tx1_train"], config)

    file_rows, thresholds = score_all_splits(latent_sets, state, methods, config)
    main_rows, device_rows = build_metric_tables(file_rows, methods, thresholds)
    corr_rows = correlation_rows(file_rows, methods)
    gain_summary = gain_rows(main_rows, device_rows)

    write_csv(tables_dir / "ap_stft_marginal_control_file_scores.csv", file_rows)
    write_csv(tables_dir / "ap_stft_marginal_control_results.csv", main_rows)
    write_csv(tables_dir / "ap_stft_marginal_control_device_results.csv", device_rows)
    write_csv(tables_dir / "ap_stft_marginal_control_correlations.csv", corr_rows)
    write_csv(tables_dir / "ap_stft_marginal_control_gain.csv", gain_summary)

    print(f"[DONE] wrote {tables_dir / 'ap_stft_marginal_control_results.csv'}")
    print(f"[DONE] wrote {tables_dir / 'ap_stft_marginal_control_device_results.csv'}")
    print(f"[DONE] wrote {tables_dir / 'ap_stft_marginal_control_gain.csv'}")
    print_summary(main_rows, gain_summary)


if __name__ == "__main__":
    main()
