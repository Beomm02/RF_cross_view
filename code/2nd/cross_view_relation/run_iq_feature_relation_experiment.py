import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_relation_scoring import build_pair_features, fit_rep_transforms, write_csv  # noqa: E402
from representation_screening import (  # noqa: E402
    DEFAULT_REPRESENTATIONS,
    binary_metrics,
    file_features,
    pca_fit,
    relation_diagnostics,
    score_pca_residual,
    score_zdist,
    standardize_apply,
    standardize_fit,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one full IQ-derived feature-relation experiment: fresh split, "
            "feature extraction, CKA/CCA diagnostics, one-class scoring, and rank fusion."
        )
    )
    parser.add_argument("--data-root", default="../../data")
    parser.add_argument("--output-dir", default="results/cross_view_relation/iq_feature_relation_full_seed42_w16")
    parser.add_argument("--feature-cache", default="artifacts/cross_view_relation/iq_feature_relation_full_seed42_w16.npz")
    parser.add_argument("--manifest-dir", default="cross_view_relation/manifests_iq_feature_relation_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-tx", default="Tx1")
    parser.add_argument("--anomaly-txs", nargs="+", default=["Tx2", "Tx3", "Tx4", "Tx5", "Tx6", "Tx7", "Tx8"])
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--max-anomaly-files-per-device", type=int, default=0)
    parser.add_argument("--representations", nargs="+", default=DEFAULT_REPRESENTATIONS, choices=DEFAULT_REPRESENTATIONS)
    parser.add_argument("--mat-key", default="rxData")
    parser.add_argument("--norm-mode", default="power")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-windows-per-file", type=int, default=16)
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
    parser.add_argument("--threshold-quantiles", type=float, nargs="+", default=[0.90, 0.95, 0.97])
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--low-cka-fraction", type=float, default=0.25)
    parser.add_argument("--high-cka-fraction", type=float, default=0.25)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def write_manifest(path: Path, files: list[Path], relative_to: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in files:
            f.write(str(item.resolve().relative_to(relative_to.resolve()).as_posix()) + "\n")


def make_split(args):
    data_root = Path(args.data_root).resolve()
    tx1_files = sorted((data_root / args.normal_tx).glob("*.mat"))
    expected = args.train_count + args.holdout_count
    if len(tx1_files) < expected:
        raise ValueError(f"{args.normal_tx} has {len(tx1_files)} files, expected at least {expected}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(tx1_files))
    train_files = sorted(tx1_files[int(i)].resolve() for i in order[: args.train_count])
    holdout_files = sorted(tx1_files[int(i)].resolve() for i in order[args.train_count : expected])

    anomaly_files = {}
    for tx in args.anomaly_txs:
        files = sorted((data_root / tx).glob("*.mat"))
        if args.max_anomaly_files_per_device and len(files) > args.max_anomaly_files_per_device:
            indices = np.sort(rng.choice(len(files), size=args.max_anomaly_files_per_device, replace=False))
            files = [files[int(i)] for i in indices]
        anomaly_files[tx] = [p.resolve() for p in files]

    manifest_dir = Path(args.manifest_dir).resolve()
    write_manifest(manifest_dir / "tx1_train_400_seed42.txt", train_files, data_root)
    write_manifest(manifest_dir / "tx1_holdout_100_seed42.txt", holdout_files, data_root)
    for tx, files in anomaly_files.items():
        write_manifest(manifest_dir / f"{tx.lower()}_all_seed42.txt", files, data_root)

    split_rows = [
        {"split": "train", "tx": args.normal_tx, "count": len(train_files)},
        {"split": "holdout", "tx": args.normal_tx, "count": len(holdout_files)},
    ]
    split_rows.extend({"split": "anomaly", "tx": tx, "count": len(files)} for tx, files in anomaly_files.items())
    return data_root, manifest_dir, train_files, holdout_files, anomaly_files, split_rows


def extract_or_load_features(args, train_files, holdout_files, anomaly_files):
    cache_path = Path(args.feature_cache).resolve()
    if cache_path.exists() and not args.force_extract:
        print(f"[pipeline] loading feature cache: {cache_path}", flush=True)
        loaded = np.load(cache_path, allow_pickle=False)
        train = {rep: loaded[f"train__{rep}"] for rep in args.representations}
        holdout = {rep: loaded[f"holdout__{rep}"] for rep in args.representations}
        anomaly = {
            tx: {rep: loaded[f"{tx}__{rep}"] for rep in args.representations}
            for tx in anomaly_files
        }
        return train, holdout, anomaly, str(cache_path)

    jobs = [("train", args.normal_tx, p) for p in train_files]
    jobs += [("holdout", args.normal_tx, p) for p in holdout_files]
    for tx, files in anomaly_files.items():
        jobs += [("anomaly", tx, p) for p in files]

    print(
        f"[pipeline] extracting features: files={len(jobs)} train={len(train_files)} "
        f"holdout={len(holdout_files)} anomaly={sum(len(v) for v in anomaly_files.values())}",
        flush=True,
    )
    start = time.time()
    train = {rep: [] for rep in args.representations}
    holdout = {rep: [] for rep in args.representations}
    anomaly = {tx: {rep: [] for rep in args.representations} for tx in anomaly_files}

    for idx, (split, tx, path) in enumerate(jobs, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(jobs):
            print(f"[pipeline] feature {idx}/{len(jobs)} elapsed={time.time() - start:.1f}s {split}:{tx}:{path.name}", flush=True)
        features = file_features(path, args)
        for rep, values in features.items():
            if split == "train":
                train[rep].append(values)
            elif split == "holdout":
                holdout[rep].append(values)
            else:
                anomaly[tx][rep].append(values)

    train = {rep: np.vstack(values) for rep, values in train.items()}
    holdout = {rep: np.vstack(values) for rep, values in holdout.items()}
    anomaly = {
        tx: {rep: np.vstack(values) for rep, values in by_rep.items()}
        for tx, by_rep in anomaly.items()
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for rep in args.representations:
        payload[f"train__{rep}"] = train[rep]
        payload[f"holdout__{rep}"] = holdout[rep]
        for tx in anomaly:
            payload[f"{tx}__{rep}"] = anomaly[tx][rep]
    np.savez_compressed(cache_path, **payload)
    print(f"[pipeline] saved local feature cache: {cache_path}", flush=True)
    return train, holdout, anomaly, str(cache_path)


def fit_shrinkage_mahalanobis(train_z: np.ndarray, shrinkage: float) -> dict:
    mu = np.mean(train_z, axis=0)
    centered = train_z - mu
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    diag = np.diag(np.diag(cov))
    target = diag
    if np.all(np.diag(target) < 1e-12):
        target = np.eye(cov.shape[0], dtype=np.float64)
    cov_shrink = (1.0 - shrinkage) * cov + shrinkage * target
    cov_shrink = cov_shrink + np.eye(cov_shrink.shape[0]) * 1e-6
    inv_cov = np.linalg.pinv(cov_shrink)
    return {"mu": mu, "inv_cov": inv_cov}


def score_shrinkage_mahalanobis(z: np.ndarray, model: dict) -> np.ndarray:
    centered = z - model["mu"]
    return np.einsum("ij,jk,ik->i", centered, model["inv_cov"], centered)


def evaluate_thresholds(train_scores, holdout_scores, anomaly_scores_by_tx, threshold_quantiles):
    rows = []
    per_tx_rows = []
    all_anomaly = np.concatenate(list(anomaly_scores_by_tx.values()))
    y_all = np.concatenate(
        [np.zeros(len(holdout_scores), dtype=np.int64), np.ones(len(all_anomaly), dtype=np.int64)]
    )
    scores_all = np.concatenate([holdout_scores, all_anomaly])
    for q in threshold_quantiles:
        threshold = float(np.quantile(train_scores, q))
        metrics = binary_metrics(y_all, (scores_all > threshold).astype(np.int64), scores_all)
        rows.append({"threshold_quantile": q, "threshold": threshold, **metrics})
        for tx, tx_scores in anomaly_scores_by_tx.items():
            y_tx = np.concatenate(
                [np.zeros(len(holdout_scores), dtype=np.int64), np.ones(len(tx_scores), dtype=np.int64)]
            )
            scores_tx = np.concatenate([holdout_scores, tx_scores])
            tx_metrics = binary_metrics(y_tx, (scores_tx > threshold).astype(np.int64), scores_tx)
            per_tx_rows.append({"tx": tx, "threshold_quantile": q, "threshold": threshold, **tx_metrics})
    return rows, per_tx_rows


def score_single_representations(train, holdout, anomaly, args):
    metric_rows = []
    per_tx_rows = []
    score_outputs = {}
    for rep in args.representations:
        mean, std = standardize_fit(train[rep])
        train_z = standardize_apply(train[rep], mean, std)
        holdout_z = standardize_apply(holdout[rep], mean, std)
        anomaly_z = {tx: standardize_apply(by_rep[rep], mean, std) for tx, by_rep in anomaly.items()}
        pca = pca_fit(train_z, args.pca_components)
        shrink = fit_shrinkage_mahalanobis(train_z, args.shrinkage)
        methods = {
            "zdist": (
                score_zdist(train_z),
                score_zdist(holdout_z),
                {tx: score_zdist(z) for tx, z in anomaly_z.items()},
            ),
            "pca_residual": (
                score_pca_residual(train_z, pca),
                score_pca_residual(holdout_z, pca),
                {tx: score_pca_residual(z, pca) for tx, z in anomaly_z.items()},
            ),
            "shrinkage_mahalanobis": (
                score_shrinkage_mahalanobis(train_z, shrink),
                score_shrinkage_mahalanobis(holdout_z, shrink),
                {tx: score_shrinkage_mahalanobis(z, shrink) for tx, z in anomaly_z.items()},
            ),
        }
        for method, scores in methods.items():
            label = f"{rep}:{method}"
            score_outputs[label] = scores
            rows, tx_rows = evaluate_thresholds(*scores, args.threshold_quantiles)
            for row in rows:
                metric_rows.append({"representation": rep, "method": method, **row})
            for row in tx_rows:
                per_tx_rows.append({"representation": rep, "method": method, **row})
    return metric_rows, per_tx_rows, score_outputs


def score_relation_pairs(train, holdout, anomaly, args):
    rep_models = fit_rep_transforms(train, args.representations, args.pca_components)
    pairs = list(itertools.combinations(args.representations, 2))
    metric_rows = []
    per_tx_rows = []
    score_outputs = {}
    for pair in pairs:
        pair_name = f"{pair[0]}:{pair[1]}"
        train_rel = build_pair_features(train, rep_models, pair)
        holdout_rel = build_pair_features(holdout, rep_models, pair)
        anomaly_rel = {tx: build_pair_features(by_rep, rep_models, pair) for tx, by_rep in anomaly.items()}
        mean, std = standardize_fit(train_rel)
        train_z = standardize_apply(train_rel, mean, std)
        holdout_z = standardize_apply(holdout_rel, mean, std)
        anomaly_z = {tx: standardize_apply(x, mean, std) for tx, x in anomaly_rel.items()}
        pca = pca_fit(train_z, args.relation_pca_components)
        shrink = fit_shrinkage_mahalanobis(train_z, args.shrinkage)
        methods = {
            "relation_zdist": (
                score_zdist(train_z),
                score_zdist(holdout_z),
                {tx: score_zdist(z) for tx, z in anomaly_z.items()},
            ),
            "relation_pca_residual": (
                score_pca_residual(train_z, pca),
                score_pca_residual(holdout_z, pca),
                {tx: score_pca_residual(z, pca) for tx, z in anomaly_z.items()},
            ),
            "relation_shrinkage_mahalanobis": (
                score_shrinkage_mahalanobis(train_z, shrink),
                score_shrinkage_mahalanobis(holdout_z, shrink),
                {tx: score_shrinkage_mahalanobis(z, shrink) for tx, z in anomaly_z.items()},
            ),
        }
        for method, scores in methods.items():
            label = f"{pair_name}:{method}"
            score_outputs[label] = scores
            rows, tx_rows = evaluate_thresholds(*scores, args.threshold_quantiles)
            for row in rows:
                metric_rows.append({"pair": pair_name, "method": method, **row})
            for row in tx_rows:
                per_tx_rows.append({"pair": pair_name, "method": method, **row})
    return metric_rows, per_tx_rows, score_outputs


def empirical_rank(train_scores, scores):
    train_scores = np.sort(np.asarray(train_scores, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    return np.searchsorted(train_scores, scores, side="right") / max(len(train_scores), 1)


def fuse_score_outputs(selected_labels, score_outputs):
    train_cols = []
    holdout_cols = []
    anomaly_cols = {}
    for label in selected_labels:
        train_scores, holdout_scores, anomaly_scores = score_outputs[label]
        train_cols.append(empirical_rank(train_scores, train_scores))
        holdout_cols.append(empirical_rank(train_scores, holdout_scores))
        for tx, tx_scores in anomaly_scores.items():
            anomaly_cols.setdefault(tx, []).append(empirical_rank(train_scores, tx_scores))
    train_rank = np.column_stack(train_cols)
    holdout_rank = np.column_stack(holdout_cols)
    anomaly_rank = {tx: np.column_stack(cols) for tx, cols in anomaly_cols.items()}
    return (
        np.mean(train_rank, axis=1),
        np.mean(holdout_rank, axis=1),
        {tx: np.mean(x, axis=1) for tx, x in anomaly_rank.items()},
    )


def score_fusions(single_outputs, relation_outputs, relation_rows, args):
    cka_ranked = sorted(relation_rows, key=lambda r: float(r["linear_cka"]))
    group_size = max(1, int(round(len(cka_ranked) * args.low_cka_fraction)))
    low_pairs = {f"{row['rep_a']}:{row['rep_b']}" for row in cka_ranked[:group_size]}
    high_pairs = {f"{row['rep_a']}:{row['rep_b']}" for row in cka_ranked[-group_size:]}

    single_pca = [label for label in single_outputs if label.endswith(":pca_residual")]
    relation_pca = [label for label in relation_outputs if label.endswith(":relation_pca_residual")]
    relation_mahal = [label for label in relation_outputs if label.endswith(":relation_shrinkage_mahalanobis")]
    low_relation_pca = [label for label in relation_pca if ":".join(label.split(":")[:2]) in low_pairs]
    high_relation_pca = [label for label in relation_pca if ":".join(label.split(":")[:2]) in high_pairs]
    low_relation_mahal = [label for label in relation_mahal if ":".join(label.split(":")[:2]) in low_pairs]

    fusion_defs = {
        "single_pca_rank_mean": single_pca,
        "all_relation_pca_rank_mean": relation_pca,
        "low_cka_relation_pca_rank_mean": low_relation_pca,
        "high_cka_relation_pca_rank_mean": high_relation_pca,
        "low_cka_relation_mahal_rank_mean": low_relation_mahal,
    }
    all_outputs = {**single_outputs, **relation_outputs}
    metric_rows = []
    per_tx_rows = []
    for fusion_name, labels in fusion_defs.items():
        if not labels:
            continue
        scores = fuse_score_outputs(labels, all_outputs)
        rows, tx_rows = evaluate_thresholds(*scores, args.threshold_quantiles)
        for row in rows:
            metric_rows.append(
                {
                    "fusion": fusion_name,
                    "component_count": len(labels),
                    "components": "|".join(labels),
                    **row,
                }
            )
        for row in tx_rows:
            per_tx_rows.append(
                {
                    "fusion": fusion_name,
                    "component_count": len(labels),
                    **row,
                }
            )
    selection_rows = []
    for group_name, pairs in [("low_cka", low_pairs), ("high_cka", high_pairs)]:
        for pair in sorted(pairs):
            selection_rows.append({"group": group_name, "pair": pair})
    return metric_rows, per_tx_rows, selection_rows


def best_rows(rows, primary="auc", n=10):
    return sorted(rows, key=lambda r: (float(r[primary]), float(r.get("f1", 0.0))), reverse=True)[:n]


def main():
    args = parse_args()
    started = time.time()
    data_root, manifest_dir, train_files, holdout_files, anomaly_files, split_rows = make_split(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "split_summary.csv", split_rows)
    train, holdout, anomaly, cache_path = extract_or_load_features(args, train_files, holdout_files, anomaly_files)

    print("[pipeline] running CKA/CCA diagnostics", flush=True)
    relation_rows = relation_diagnostics(train, args)
    write_csv(output_dir / "relation_diagnostics.csv", relation_rows)

    print("[pipeline] scoring single representations", flush=True)
    single_rows, single_tx_rows, single_outputs = score_single_representations(train, holdout, anomaly, args)
    write_csv(output_dir / "single_representation_metrics.csv", single_rows)
    write_csv(output_dir / "single_representation_per_tx_metrics.csv", single_tx_rows)

    print("[pipeline] scoring relation pairs", flush=True)
    relation_score_rows, relation_tx_rows, relation_outputs = score_relation_pairs(train, holdout, anomaly, args)
    write_csv(output_dir / "relation_pair_metrics.csv", relation_score_rows)
    write_csv(output_dir / "relation_pair_per_tx_metrics.csv", relation_tx_rows)

    print("[pipeline] scoring train-rank fusions", flush=True)
    fusion_rows, fusion_tx_rows, selection_rows = score_fusions(
        single_outputs,
        relation_outputs,
        relation_rows,
        args,
    )
    write_csv(output_dir / "fusion_metrics.csv", fusion_rows)
    write_csv(output_dir / "fusion_per_tx_metrics.csv", fusion_tx_rows)
    write_csv(output_dir / "cka_selected_pairs.csv", selection_rows)

    summary = {
        "data_root_name": data_root.name,
        "manifest_dir": str(manifest_dir),
        "feature_cache": str(cache_path),
        "elapsed_seconds": time.time() - started,
        "split": {
            "train": len(train_files),
            "holdout": len(holdout_files),
            "anomaly_total": sum(len(v) for v in anomaly_files.values()),
            "anomaly_by_tx": {tx: len(files) for tx, files in anomaly_files.items()},
        },
        "representations": args.representations,
        "best_single": best_rows(single_rows, n=5),
        "best_relation_pair": best_rows(relation_score_rows, n=10),
        "best_fusion": best_rows(fusion_rows, n=5),
        "lowest_cka_pairs": sorted(relation_rows, key=lambda r: float(r["linear_cka"]))[:5],
        "highest_cka_pairs": sorted(relation_rows, key=lambda r: float(r["linear_cka"]), reverse=True)[:5],
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    run_config = vars(args).copy()
    run_config.update(
        {
            "data_root": str(data_root),
            "manifest_dir": str(manifest_dir),
            "output_dir": str(output_dir),
            "feature_cache": str(cache_path),
            "elapsed_seconds": time.time() - started,
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("[pipeline] best single")
    for row in summary["best_single"]:
        print(
            f"  {row['representation']:12s} {row['method']:24s} q={row['threshold_quantile']:.2f} "
            f"auc={row['auc']:.6f} f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}",
            flush=True,
        )
    print("[pipeline] best relation")
    for row in summary["best_relation_pair"][:5]:
        print(
            f"  {row['pair']:18s} {row['method']:30s} q={row['threshold_quantile']:.2f} "
            f"auc={row['auc']:.6f} f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}",
            flush=True,
        )
    print("[pipeline] best fusion")
    for row in summary["best_fusion"]:
        print(
            f"  {row['fusion']:34s} q={row['threshold_quantile']:.2f} "
            f"auc={row['auc']:.6f} f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}",
            flush=True,
        )
    print(f"[pipeline] wrote: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
