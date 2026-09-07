from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_multiview_relation.detectors.combined import RobustScoreFusion  # noqa: E402
from rf_multiview_relation.pipeline import (  # noqa: E402
    MAIN_METHODS,
    aggregate_file_scores,
    assert_no_file_leakage,
    assert_tx1_only_fit,
    feature_matrix_for_method,
    fit_cca_models,
    fit_mahalanobis,
    load_latent_dir,
    resolve_project_path,
    save_pickle,
    score_windows_for_method,
    threshold_from_file_rows,
)
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 fit Tx1-only relation and baseline Mahalanobis models.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--methods", nargs="+", default=list(MAIN_METHODS))
    return parser.parse_args()


def maybe_with_run_name(path: Path, run_name: str) -> Path:
    return path / run_name if run_name else path


def fit_detectors(
    methods: list[str],
    train_latents: dict[str, np.ndarray],
    config: dict[str, Any],
    cca_models: dict[str, Any],
    relation_mode: str,
) -> dict[str, Any]:
    detectors: dict[str, Any] = {}
    for method in methods:
        if method == "score_fusion" or method.endswith("_direct"):
            continue
        features = feature_matrix_for_method(method, train_latents, cca_models, relation_mode=relation_mode)
        detectors[method] = fit_mahalanobis(features, config)
        print(
            f"[FIT] method={method} samples={features.shape[0]} dim={features.shape[1]} "
            f"covariance={detectors[method].fitted_covariance_method_}"
        )
    return detectors


def fit_score_fusion(
    methods: list[str],
    detectors: dict[str, Any],
    calibration_latents: dict[str, np.ndarray],
    cca_models: dict[str, Any],
    config: dict[str, Any],
    relation_mode: str,
) -> RobustScoreFusion | None:
    if "score_fusion" not in methods:
        return None
    required = {"concat", "cca_relation"}
    missing = required - set(detectors)
    if missing:
        raise ValueError(f"score_fusion requires fitted detectors: {sorted(missing)}")
    concat_scores = score_windows_for_method("concat", calibration_latents, detectors, cca_models, relation_mode)
    relation_scores = score_windows_for_method("cca_relation", calibration_latents, detectors, cca_models, relation_mode)
    fusion = RobustScoreFusion(alpha=float(config["score_fusion"]["alpha"]))
    fusion.fit(concat_scores, relation_scores)
    print(f"[FIT] method=score_fusion alpha={fusion.alpha}")
    return fusion


def calibration_thresholds(
    methods: list[str],
    detectors: dict[str, Any],
    calibration_latents: dict[str, np.ndarray],
    cca_models: dict[str, Any],
    fusion: RobustScoreFusion | None,
    config: dict[str, Any],
    relation_mode: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    thresholds: dict[str, float] = {}
    threshold_rows: list[dict[str, Any]] = []
    file_percentile = float(config["file_scoring"]["percentile"])
    threshold_percentile = float(config["threshold"]["percentile"])
    for method in methods:
        scores = score_windows_for_method(method, calibration_latents, detectors, cca_models, relation_mode, fusion=fusion)
        rows = aggregate_file_scores(
            calibration_latents["file_id"],
            calibration_latents["device"],
            calibration_latents["split"],
            scores,
            method,
            threshold=None,
            percentile=file_percentile,
        )
        threshold = threshold_from_file_rows(rows, threshold_percentile)
        thresholds[method] = threshold
        threshold_rows.append(
            {
                "method": method,
                "source": "tx1_calibration",
                "calibration_files": len(rows),
                "file_score_percentile": file_percentile,
                "threshold_percentile": threshold_percentile,
                "threshold": threshold,
            }
        )
        print(f"[THRESHOLD] method={method} tau={threshold:.6f} source=Tx1_calibration")
    return thresholds, threshold_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths_cfg = config["paths"]
    output_dir = resolve_project_path(args.output_dir or paths_cfg["outputs"], PROJECT_ROOT)
    latent_dir = maybe_with_run_name(resolve_project_path(args.latent_dir or (output_dir / "latents"), PROJECT_ROOT), args.run_name)
    artifact_dir = maybe_with_run_name(resolve_project_path(args.artifact_dir or (output_dir / "artifacts"), PROJECT_ROOT), args.run_name)
    tables_dir = maybe_with_run_name(output_dir / "tables", args.run_name)
    methods = [str(method) for method in args.methods]
    relation_mode = str(config["relation"]["mode"])

    latent_sets = load_latent_dir(latent_dir)
    assert_no_file_leakage(latent_sets)
    train_latents = latent_sets["tx1_train"]
    calibration_latents = latent_sets["tx1_calibration"]
    assert_tx1_only_fit(train_latents, "tx1_train")
    assert_tx1_only_fit(calibration_latents, "tx1_calibration")

    cca_models = fit_cca_models(train_latents, config)
    detectors = fit_detectors(methods, train_latents, config, cca_models, relation_mode)
    fusion = fit_score_fusion(methods, detectors, calibration_latents, cca_models, config, relation_mode)
    thresholds, threshold_rows = calibration_thresholds(
        methods,
        detectors,
        calibration_latents,
        cca_models,
        fusion,
        config,
        relation_mode,
    )

    payload = {
        "config": config,
        "methods": methods,
        "relation_mode": relation_mode,
        "cca_models": cca_models,
        "detectors": detectors,
        "score_fusion": fusion,
        "thresholds": thresholds,
    }
    save_pickle(artifact_dir / "relation_detectors.pkl", payload)
    write_csv(tables_dir / "detector_thresholds.csv", threshold_rows)
    print(f"[DONE] wrote {artifact_dir / 'relation_detectors.pkl'}")
    print(f"[DONE] wrote {tables_dir / 'detector_thresholds.csv'}")


if __name__ == "__main__":
    main()
