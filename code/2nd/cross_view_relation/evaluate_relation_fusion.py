import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_relation_scoring import (  # noqa: E402
    build_pair_features,
    evaluate_scores,
    fit_rep_transforms,
    load_feature_matrices,
    parse_pairs,
    score_pca_residual,
    score_zdist,
    standardize_apply,
    standardize_fit,
    write_csv,
)
from representation_screening import pca_fit  # noqa: E402


DEFAULT_COMPONENTS = [
    "single:cyclo:pca_residual",
    "relation:ap:cyclo:pca_residual",
    "relation:iq:cyclo:pca_residual",
    "relation:iq:bispectrum:zdist",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse selected cross-view scores using Tx1-train empirical-rank calibration."
    )
    parser.add_argument("--manifest-dir", default="cross_view_relation/manifests")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--normal-manifest", default=None)
    parser.add_argument("--abnormal-manifests", nargs="*", default=None)
    parser.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    parser.add_argument("--output-dir", default="results/cross_view_relation/relation_fusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mat-key", default="rxData")
    parser.add_argument("--norm-mode", default="power")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-windows-per-file", type=int, default=16)
    parser.add_argument("--max-files-train", type=int, default=128)
    parser.add_argument("--max-files-normal", type=int, default=80)
    parser.add_argument("--max-files-anomaly-per-device", type=int, default=80)
    parser.add_argument("--fft-bins", type=int, default=64)
    parser.add_argument("--stft-nperseg", type=int, default=128)
    parser.add_argument("--stft-noverlap", type=int, default=96)
    parser.add_argument("--stft-freq-bins", type=int, default=16)
    parser.add_argument("--stft-time-bins", type=int, default=8)
    parser.add_argument("--cepstral-coeffs", type=int, default=32)
    parser.add_argument("--cyclo-shifts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--bispectrum-bins", type=int, default=12)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--relation-pca-components", type=int, default=16)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-quantiles", type=float, nargs="+", default=None)
    return parser.parse_args()


def parse_components(items):
    reps = set()
    relation_pairs = []
    parsed = []
    for item in items:
        parts = item.split(":")
        if len(parts) == 3 and parts[0] == "single":
            _, rep, method = parts
            reps.add(rep)
            parsed.append({"kind": "single", "label": item, "rep": rep, "method": method})
        elif len(parts) == 4 and parts[0] == "relation":
            _, rep_a, rep_b, method = parts
            reps.update([rep_a, rep_b])
            relation_pairs.append(f"{rep_a}:{rep_b}")
            parsed.append(
                {
                    "kind": "relation",
                    "label": item,
                    "pair": (rep_a, rep_b),
                    "method": method,
                }
            )
        else:
            raise ValueError(f"invalid component: {item}")
    parse_pairs(relation_pairs)
    return parsed, sorted(reps)


def empirical_rank(train_scores, scores):
    train_scores = np.sort(np.asarray(train_scores, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    return np.searchsorted(train_scores, scores, side="right") / max(len(train_scores), 1)


def single_component_scores(rep, method, train, normal, anomaly, args):
    mean, std = standardize_fit(train[rep])
    train_z = standardize_apply(train[rep], mean, std)
    normal_z = standardize_apply(normal[rep], mean, std)
    anomaly_z = {tx: standardize_apply(by_rep[rep], mean, std) for tx, by_rep in anomaly.items()}
    pca = pca_fit(train_z, args.pca_components)
    if method == "zdist":
        return (
            score_zdist(train_z),
            score_zdist(normal_z),
            {tx: score_zdist(z) for tx, z in anomaly_z.items()},
        )
    if method == "pca_residual":
        return (
            score_pca_residual(train_z, pca),
            score_pca_residual(normal_z, pca),
            {tx: score_pca_residual(z, pca) for tx, z in anomaly_z.items()},
        )
    raise ValueError(f"unknown single method: {method}")


def relation_component_scores(pair, method, train, normal, anomaly, rep_models, args):
    train_rel = build_pair_features(train, rep_models, pair)
    normal_rel = build_pair_features(normal, rep_models, pair)
    anomaly_rel = {tx: build_pair_features(by_rep, rep_models, pair) for tx, by_rep in anomaly.items()}
    mean, std = standardize_fit(train_rel)
    train_z = standardize_apply(train_rel, mean, std)
    normal_z = standardize_apply(normal_rel, mean, std)
    anomaly_z = {tx: standardize_apply(x, mean, std) for tx, x in anomaly_rel.items()}
    pca = pca_fit(train_z, args.relation_pca_components)
    if method == "zdist":
        return (
            score_zdist(train_z),
            score_zdist(normal_z),
            {tx: score_zdist(z) for tx, z in anomaly_z.items()},
        )
    if method == "pca_residual":
        return (
            score_pca_residual(train_z, pca),
            score_pca_residual(normal_z, pca),
            {tx: score_pca_residual(z, pca) for tx, z in anomaly_z.items()},
        )
    raise ValueError(f"unknown relation method: {method}")


def stack_component_ranks(component_outputs):
    train_cols = []
    normal_cols = []
    anomaly_cols = {}
    for _, train_scores, normal_scores, anomaly_scores in component_outputs:
        train_cols.append(empirical_rank(train_scores, train_scores))
        normal_cols.append(empirical_rank(train_scores, normal_scores))
        for tx, scores in anomaly_scores.items():
            anomaly_cols.setdefault(tx, []).append(empirical_rank(train_scores, scores))
    train_rank = np.column_stack(train_cols)
    normal_rank = np.column_stack(normal_cols)
    anomaly_rank = {tx: np.column_stack(cols) for tx, cols in anomaly_cols.items()}
    return train_rank, normal_rank, anomaly_rank


def fusion_methods(train_rank, normal_rank, anomaly_rank):
    return {
        "rank_mean": (
            np.mean(train_rank, axis=1),
            np.mean(normal_rank, axis=1),
            {tx: np.mean(x, axis=1) for tx, x in anomaly_rank.items()},
        ),
        "rank_max": (
            np.max(train_rank, axis=1),
            np.max(normal_rank, axis=1),
            {tx: np.max(x, axis=1) for tx, x in anomaly_rank.items()},
        ),
        "rank_top2_mean": (
            np.mean(np.sort(train_rank, axis=1)[:, -2:], axis=1),
            np.mean(np.sort(normal_rank, axis=1)[:, -2:], axis=1),
            {tx: np.mean(np.sort(x, axis=1)[:, -2:], axis=1) for tx, x in anomaly_rank.items()},
        ),
        "rank_cyclo_weighted": (
            0.4 * train_rank[:, 0] + 0.2 * np.sum(train_rank[:, 1:], axis=1),
            0.4 * normal_rank[:, 0] + 0.2 * np.sum(normal_rank[:, 1:], axis=1),
            {tx: 0.4 * x[:, 0] + 0.2 * np.sum(x[:, 1:], axis=1) for tx, x in anomaly_rank.items()},
        ),
    }


def main():
    args = parse_args()
    start = time.time()
    threshold_quantiles = args.threshold_quantiles or [args.threshold_quantile]
    components, reps = parse_components(args.components)
    args.pairs = [
        f"{component['pair'][0]}:{component['pair'][1]}"
        for component in components
        if component["kind"] == "relation"
    ]
    train, normal, anomaly, config = load_feature_matrices(args, reps)
    rep_models = fit_rep_transforms(train, reps, args.pca_components)

    component_outputs = []
    component_rows = []
    for component in components:
        if component["kind"] == "single":
            scores = single_component_scores(
                component["rep"],
                component["method"],
                train,
                normal,
                anomaly,
                args,
            )
        else:
            scores = relation_component_scores(
                component["pair"],
                component["method"],
                train,
                normal,
                anomaly,
                rep_models,
                args,
            )
        train_scores, normal_scores, anomaly_scores = scores
        for threshold_quantile in threshold_quantiles:
            threshold, metrics, _ = evaluate_scores(
                train_scores,
                normal_scores,
                anomaly_scores,
                threshold_quantile,
            )
            component_rows.append(
                {
                    "component": component["label"],
                    "threshold_quantile": threshold_quantile,
                    "threshold": threshold,
                    **metrics,
                }
            )
        component_outputs.append((component["label"], train_scores, normal_scores, anomaly_scores))

    train_rank, normal_rank, anomaly_rank = stack_component_ranks(component_outputs)
    metric_rows = []
    per_tx_rows = []
    for method, (train_scores, normal_scores, anomaly_scores) in fusion_methods(
        train_rank,
        normal_rank,
        anomaly_rank,
    ).items():
        for threshold_quantile in threshold_quantiles:
            threshold, metrics, tx_metrics = evaluate_scores(
                train_scores,
                normal_scores,
                anomaly_scores,
                threshold_quantile,
            )
            metric_rows.append(
                {
                    "fusion": method,
                    "components": "|".join(label for label, *_ in component_outputs),
                    "threshold_quantile": threshold_quantile,
                    "threshold": threshold,
                    **metrics,
                }
            )
            for tx, row in tx_metrics.items():
                per_tx_rows.append(
                    {
                        "fusion": method,
                        "tx": tx,
                        "threshold_quantile": threshold_quantile,
                        "threshold": threshold,
                        **row,
                    }
                )

    output_dir = Path(args.output_dir).resolve()
    write_csv(output_dir / "fusion_metrics.csv", metric_rows)
    write_csv(output_dir / "fusion_per_tx_metrics.csv", per_tx_rows)
    write_csv(output_dir / "component_metrics.csv", component_rows)
    run_config = vars(args).copy()
    run_config.update(config)
    run_config["representations_used"] = reps
    run_config["threshold_quantiles_effective"] = threshold_quantiles
    run_config["elapsed_seconds"] = time.time() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("[fusion] component metrics")
    for row in sorted(component_rows, key=lambda r: (r["auc"], r["f1"]), reverse=True):
        print(
            f"  q={row['threshold_quantile']:.2f} {row['component']:42s} auc={row['auc']:.6f} "
            f"f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}"
        )
    print("[fusion] fusion metrics")
    for row in sorted(metric_rows, key=lambda r: (r["auc"], r["f1"]), reverse=True):
        print(
            f"  q={row['threshold_quantile']:.2f} {row['fusion']:20s} auc={row['auc']:.6f} "
            f"f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}"
        )
    print(f"[fusion] wrote: {output_dir}")


if __name__ == "__main__":
    main()
