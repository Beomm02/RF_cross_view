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
    assert_tx1_only_fit,
    fit_cca_models,
    load_latent_dir,
    resolve_project_path,
)
from rf_multiview_relation.relation.relation_features import cosine_distance, euclidean_distance  # noqa: E402
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402
from rf_multiview_relation.utils.metrics import roc_auc_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose which multi-view relation pairs separate anomalies.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--eval-splits", nargs="+", default=["tx1_holdout", "tx2"])
    return parser.parse_args()


def maybe_with_run_name(path: Path, run_name: str) -> Path:
    return path / run_name if run_name else path


def pair_window_scores(latents: dict[str, np.ndarray], cca_models: dict[str, Any]) -> dict[str, np.ndarray]:
    scores = {}
    for pair_name, left, right in PAIR_KEYS:
        raw_left = latents[left]
        raw_right = latents[right]
        cca_left, cca_right = cca_models[pair_name].transform(raw_left, raw_right)
        scores[f"{pair_name}_raw_cosine"] = cosine_distance(raw_left, raw_right)
        scores[f"{pair_name}_raw_l2"] = euclidean_distance(raw_left, raw_right)
        scores[f"{pair_name}_cca_cosine"] = cosine_distance(cca_left, cca_right)
        scores[f"{pair_name}_cca_l2"] = euclidean_distance(cca_left, cca_right)
        scores[f"{pair_name}_cca_abs_mean"] = np.mean(np.abs(cca_left - cca_right), axis=1)
    return scores


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths_cfg = config["paths"]
    output_dir = resolve_project_path(args.output_dir or paths_cfg["outputs"], PROJECT_ROOT)
    latent_dir = maybe_with_run_name(resolve_project_path(args.latent_dir or (output_dir / "latents"), PROJECT_ROOT), args.run_name)
    tables_dir = maybe_with_run_name(output_dir / "tables", args.run_name)

    latent_sets = load_latent_dir(latent_dir)
    train_latents = latent_sets["tx1_train"]
    assert_tx1_only_fit(train_latents, "tx1_train")
    cca_models = fit_cca_models(train_latents, config)

    file_rows = []
    for split in [value.lower() for value in args.eval_splits]:
        if split not in latent_sets:
            print(f"[PAIR] skip missing split={split}")
            continue
        latents = latent_sets[split]
        for metric, window_scores in pair_window_scores(latents, cca_models).items():
            file_rows.extend(
                aggregate_file_scores(
                    latents["file_id"],
                    latents["device"],
                    latents["split"],
                    window_scores,
                    metric,
                    threshold=None,
                    percentile=float(config["file_scoring"]["percentile"]),
                )
            )

    summary_rows = []
    metrics = sorted({str(row["method"]) for row in file_rows})
    normal_by_metric = {
        metric: np.asarray(
            [
                float(row["file_score"])
                for row in file_rows
                if row["method"] == metric and row["split"] == "tx1_holdout"
            ],
            dtype=np.float64,
        )
        for metric in metrics
    }
    for split in [value.lower() for value in args.eval_splits if value.lower() != "tx1_holdout"]:
        for metric in metrics:
            normal = normal_by_metric[metric]
            anomaly = np.asarray(
                [
                    float(row["file_score"])
                    for row in file_rows
                    if row["method"] == metric and row["split"] == split
                ],
                dtype=np.float64,
            )
            if normal.size == 0 or anomaly.size == 0:
                continue
            scores = np.concatenate([normal, anomaly])
            labels = np.concatenate([np.zeros(normal.size, dtype=np.int64), np.ones(anomaly.size, dtype=np.int64)])
            summary_rows.append(
                {
                    "comparison": f"tx1_holdout_vs_{split}",
                    "metric": metric,
                    "normal_files": int(normal.size),
                    "anomaly_files": int(anomaly.size),
                    "tx1_median": float(np.median(normal)),
                    "anomaly_median": float(np.median(anomaly)),
                    "median_shift": float(np.median(anomaly) - np.median(normal)),
                    "auroc": float(roc_auc_score(labels, scores)),
                    "p_value": p_value(normal, anomaly),
                    "cliffs_delta_anomaly_minus_tx1": cliffs_delta(anomaly, normal),
                }
            )

    write_csv(tables_dir / "relation_pair_file_scores.csv", file_rows)
    write_csv(tables_dir / "relation_pair_diagnostics.csv", summary_rows)
    print(f"[DONE] wrote {tables_dir / 'relation_pair_file_scores.csv'}")
    print(f"[DONE] wrote {tables_dir / 'relation_pair_diagnostics.csv'}")
    for row in sorted(summary_rows, key=lambda r: float(r["auroc"]), reverse=True)[:8]:
        print(
            f"[PAIR] {row['metric']} AUROC={float(row['auroc']):.4f} "
            f"shift={float(row['median_shift']):.6f} delta={float(row['cliffs_delta_anomaly_minus_tx1']):.4f}"
        )


if __name__ == "__main__":
    main()
