from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_multiview_relation.pipeline import (  # noqa: E402
    aggregate_file_scores,
    load_latent_dir,
    load_pickle,
    resolve_project_path,
    score_windows_for_method,
)
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402
from rf_multiview_relation.utils.metrics import binary_metrics  # noqa: E402


DEFAULT_EVAL_SPLITS = ("tx1_holdout", "tx2", "tx3", "tx4", "tx5", "tx6", "tx7", "tx8", "oracle")
DISPLAY_NAMES = {
    "iq": "IQ-only",
    "ap": "AP-only",
    "stft": "STFT-only",
    "concat": "Concat",
    "raw_relation": "Raw Relation",
    "cca_relation": "Proposed CCA Relation",
    "cca_compact_relation": "CCA Compact Relation",
    "ap_stft_cca_cosine": "AP-STFT CCA Cosine",
    "ap_stft_cca_l2": "AP-STFT CCA L2",
    "ap_stft_cca_abs_mean": "AP-STFT CCA Abs Mean",
    "ap_stft_cca_cosine_direct": "AP-STFT CCA Cosine Direct",
    "ap_stft_cca_l2_direct": "AP-STFT CCA L2 Direct",
    "ap_stft_cca_abs_mean_direct": "AP-STFT CCA Abs Mean Direct",
    "absolute_plus_relation": "Concat + Relation",
    "score_fusion": "Score Fusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 evaluate Tx1-only RF anomaly detectors.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--eval-splits", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--stat-method", default="cca_relation")
    parser.add_argument("--write-window-scores", action="store_true")
    return parser.parse_args()


def maybe_with_run_name(path: Path, run_name: str) -> Path:
    return path / run_name if run_name else path


def score_splits(
    latent_sets: dict[str, dict[str, np.ndarray]],
    eval_splits: list[str],
    artifact: dict[str, Any],
    methods: list[str],
    config: dict[str, Any],
    scores_dir: Path,
    write_window_scores: bool,
) -> list[dict[str, Any]]:
    file_rows: list[dict[str, Any]] = []
    window_handle = None
    writer = None
    if write_window_scores:
        scores_dir.mkdir(parents=True, exist_ok=True)
        window_handle = (scores_dir / "window_scores.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            window_handle,
            fieldnames=["method", "split", "device", "file_id", "window_id", "window_score"],
        )
        writer.writeheader()

    try:
        for split_name in eval_splits:
            if split_name not in latent_sets:
                print(f"[EVAL] skip missing split={split_name}")
                continue
            latents = latent_sets[split_name]
            for method in methods:
                scores = score_windows_for_method(
                    method,
                    latents,
                    artifact["detectors"],
                    artifact["cca_models"],
                    artifact["relation_mode"],
                    fusion=artifact.get("score_fusion"),
                )
                threshold = float(artifact["thresholds"][method])
                rows = aggregate_file_scores(
                    latents["file_id"],
                    latents["device"],
                    latents["split"],
                    scores,
                    method,
                    threshold=threshold,
                    percentile=float(config["file_scoring"]["percentile"]),
                )
                file_rows.extend(rows)
                if writer is not None:
                    for idx, score in enumerate(scores):
                        writer.writerow(
                            {
                                "method": method,
                                "split": str(latents["split"][idx]),
                                "device": str(latents["device"][idx]),
                                "file_id": str(latents["file_id"][idx]),
                                "window_id": int(latents["window_id"][idx]),
                                "window_score": float(score),
                            }
                        )
                print(f"[EVAL] split={split_name} method={method} files={len(rows)}")
    finally:
        if window_handle is not None:
            window_handle.close()
    return file_rows


def subset_rows(rows: list[dict[str, Any]], method: str, splits: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["method"] == method and row["split"] in splits]


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
    metrics = binary_metrics(labels, scores, threshold)
    return {
        "evaluation": evaluation,
        "method": method,
        "method_label": DISPLAY_NAMES.get(method, method),
        "normal_files": int(np.sum(labels == 0)),
        "anomaly_files": int(np.sum(labels == 1)),
        "threshold": threshold,
        **metrics,
    }


def build_metric_tables(
    file_rows: list[dict[str, Any]],
    methods: list[str],
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    closed_anomalies = {f"tx{idx}" for idx in range(2, 9)}
    for method in methods:
        threshold = float(thresholds[method])
        closed_rows = subset_rows(file_rows, method, {"tx1_holdout"} | closed_anomalies)
        result = metrics_for_rows(closed_rows, method, "closed_tx2_tx8", threshold)
        if result is not None:
            main_rows.append(result)
        minimal_rows = subset_rows(file_rows, method, {"tx1_holdout", "tx2"})
        result = metrics_for_rows(minimal_rows, method, "minimal_tx2", threshold)
        if result is not None:
            main_rows.append(result)
        oracle_rows = subset_rows(file_rows, method, {"tx1_holdout", "oracle"})
        result = metrics_for_rows(oracle_rows, method, "external_oracle", threshold)
        if result is not None:
            main_rows.append(result)
        for anomaly_idx in range(2, 9):
            split_name = f"tx{anomaly_idx}"
            rows = subset_rows(file_rows, method, {"tx1_holdout", split_name})
            result = metrics_for_rows(rows, method, f"tx1_vs_tx{anomaly_idx}", threshold)
            if result is not None:
                result["anomaly"] = f"Tx{anomaly_idx}"
                device_rows.append(result)
    return main_rows, device_rows


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diffs = x[:, np.newaxis] - y[np.newaxis, :]
    return float((np.sum(diffs > 0.0) - np.sum(diffs < 0.0)) / diffs.size)


def statistical_tests(file_rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    try:
        from scipy.stats import mannwhitneyu
    except ModuleNotFoundError:
        mannwhitneyu = None
    normal = np.asarray(
        [
            float(row["file_score"])
            for row in file_rows
            if row["method"] == method and row["split"] == "tx1_holdout"
        ],
        dtype=np.float64,
    )
    rows = []
    for anomaly_idx in range(2, 9):
        split_name = f"tx{anomaly_idx}"
        anomaly = np.asarray(
            [
                float(row["file_score"])
                for row in file_rows
                if row["method"] == method and row["split"] == split_name
            ],
            dtype=np.float64,
        )
        if normal.size == 0 or anomaly.size == 0:
            continue
        p_value = float("nan")
        if mannwhitneyu is not None:
            p_value = float(mannwhitneyu(normal, anomaly, alternative="two-sided").pvalue)
        rows.append(
            {
                "method": method,
                "comparison": f"Tx1_holdout_vs_Tx{anomaly_idx}",
                "tx1_median_score": float(np.median(normal)),
                "anomaly_median_score": float(np.median(anomaly)),
                "p_value": p_value,
                "cliffs_delta_anomaly_minus_tx1": cliffs_delta(anomaly, normal),
            }
        )
    return rows


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)
    for value in np.unique(values):
        mask = values == value
        if np.sum(mask) > 1:
            ranks[mask] = float(np.mean(ranks[mask]))
    return ranks


def corr_or_nan(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    x = rankdata(left) if rank else np.asarray(left, dtype=np.float64)
    y = rankdata(right) if rank else np.asarray(right, dtype=np.float64)
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def complementarity_rows(file_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    evaluations = {
        "minimal_tx2": {"tx1_holdout", "tx2"},
        "closed_tx2_tx8": {"tx1_holdout"} | {f"tx{idx}" for idx in range(2, 9)},
    }
    for evaluation, splits in evaluations.items():
        concat_map = {
            row["file_id"]: float(row["file_score"])
            for row in file_rows
            if row["method"] == "concat" and row["split"] in splits
        }
        relation_map = {
            row["file_id"]: float(row["file_score"])
            for row in file_rows
            if row["method"] == "cca_relation" and row["split"] in splits
        }
        common = sorted(set(concat_map) & set(relation_map))
        if not common:
            continue
        concat_scores = np.asarray([concat_map[key] for key in common], dtype=np.float64)
        relation_scores = np.asarray([relation_map[key] for key in common], dtype=np.float64)
        rows.append(
            {
                "evaluation": evaluation,
                "files": len(common),
                "pearson_corr_concat_relation": corr_or_nan(concat_scores, relation_scores),
                "spearman_corr_concat_relation": corr_or_nan(concat_scores, relation_scores, rank=True),
            }
        )
    return rows


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    thresholds = np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]
    tpr = []
    fpr = []
    for threshold in thresholds:
        pred = s > threshold
        tp = np.sum((y == 1) & pred)
        fp = np.sum((y == 0) & pred)
        tn = np.sum((y == 0) & ~pred)
        fn = np.sum((y == 1) & ~pred)
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
    return np.asarray(fpr), np.asarray(tpr)


def save_figures(file_rows: list[dict[str, Any]], methods: list[str], figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)

    relation_rows = [row for row in file_rows if row["method"] == "cca_relation"]
    labels = ["Tx1"] + [f"Tx{idx}" for idx in range(2, 9)]
    splits = ["tx1_holdout"] + [f"tx{idx}" for idx in range(2, 9)]
    box_data = [
        [float(row["file_score"]) for row in relation_rows if row["split"] == split]
        for split in splits
    ]
    if any(box_data):
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
        ax.boxplot(box_data, labels=labels, showfliers=False)
        ax.set_ylabel("Mahalanobis relation score")
        ax.set_title("CCA Relation Score Distribution")
        fig.tight_layout()
        fig.savefig(figures_dir / "score_distribution.png")
        plt.close(fig)

    closed_splits = {"tx1_holdout"} | {f"tx{idx}" for idx in range(2, 9)}
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=150)
    plotted = False
    for method in methods:
        rows = subset_rows(file_rows, method, closed_splits)
        if not rows:
            continue
        labels_arr = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        if np.unique(labels_arr).size < 2:
            continue
        scores = np.asarray([float(row["file_score"]) for row in rows], dtype=np.float64)
        fpr, tpr = roc_points(labels_arr, scores)
        ax.plot(fpr, tpr, label=DISPLAY_NAMES.get(method, method))
        plotted = True
    if plotted:
        ax.plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Closed Dataset ROC")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "roc_curve.png")
    plt.close(fig)

    oracle_data = [
        [float(row["file_score"]) for row in relation_rows if row["split"] == "tx1_holdout"],
        [float(row["file_score"]) for row in relation_rows if row["split"] == "oracle"],
    ]
    if all(oracle_data):
        fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=150)
        ax.boxplot(oracle_data, labels=["Tx1 holdout", "Oracle"], showfliers=False)
        ax.set_ylabel("Mahalanobis relation score")
        ax.set_title("External Oracle Score")
        fig.tight_layout()
        fig.savefig(figures_dir / "oracle_distribution.png")
        plt.close(fig)

    concat_map = {
        row["file_id"]: (float(row["file_score"]), int(row["label"]))
        for row in file_rows
        if row["method"] == "concat" and row["split"] in closed_splits
    }
    relation_map = {
        row["file_id"]: float(row["file_score"])
        for row in file_rows
        if row["method"] == "cca_relation" and row["split"] in closed_splits
    }
    common = sorted(set(concat_map) & set(relation_map))
    if common:
        x = np.asarray([concat_map[key][0] for key in common], dtype=np.float64)
        y = np.asarray([relation_map[key] for key in common], dtype=np.float64)
        labels_arr = np.asarray([concat_map[key][1] for key in common], dtype=np.int64)
        fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=150)
        ax.scatter(x[labels_arr == 0], y[labels_arr == 0], s=14, alpha=0.7, label="Tx1")
        ax.scatter(x[labels_arr == 1], y[labels_arr == 1], s=14, alpha=0.35, label="Anomaly")
        ax.set_xlabel("Concat score")
        ax.set_ylabel("CCA relation score")
        ax.set_title("Absolute vs Relation Scores")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "absolute_vs_relation_score.png")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths_cfg = config["paths"]
    output_dir = resolve_project_path(args.output_dir or paths_cfg["outputs"], PROJECT_ROOT)
    latent_dir = maybe_with_run_name(resolve_project_path(args.latent_dir or (output_dir / "latents"), PROJECT_ROOT), args.run_name)
    artifact_dir = maybe_with_run_name(resolve_project_path(args.artifact_dir or (output_dir / "artifacts"), PROJECT_ROOT), args.run_name)
    scores_dir = maybe_with_run_name(output_dir / "scores", args.run_name)
    tables_dir = maybe_with_run_name(output_dir / "tables", args.run_name)
    figures_dir = maybe_with_run_name(output_dir / "figures", args.run_name)

    artifact = load_pickle(artifact_dir / "relation_detectors.pkl")
    latent_sets = load_latent_dir(latent_dir)
    methods = [str(method) for method in (args.methods or artifact["methods"])]
    eval_splits = [split.lower() for split in (args.eval_splits or DEFAULT_EVAL_SPLITS)]

    file_rows = score_splits(
        latent_sets=latent_sets,
        eval_splits=eval_splits,
        artifact=artifact,
        methods=methods,
        config=config,
        scores_dir=scores_dir,
        write_window_scores=bool(args.write_window_scores),
    )
    scores_dir.mkdir(parents=True, exist_ok=True)
    write_csv(scores_dir / "file_scores.csv", file_rows)
    main_rows, device_rows = build_metric_tables(file_rows, methods, artifact["thresholds"])
    write_csv(tables_dir / "main_results.csv", main_rows)
    write_csv(tables_dir / "device_results.csv", device_rows)
    write_csv(tables_dir / "statistical_tests.csv", statistical_tests(file_rows, args.stat_method))
    write_csv(tables_dir / "score_complementarity.csv", complementarity_rows(file_rows))
    save_figures(file_rows, methods, figures_dir)

    print(f"[DONE] wrote {scores_dir / 'file_scores.csv'}")
    print(f"[DONE] wrote {tables_dir / 'main_results.csv'}")
    print(f"[DONE] wrote {tables_dir / 'device_results.csv'}")
    print(f"[DONE] wrote figures under {figures_dir}")


if __name__ == "__main__":
    main()
