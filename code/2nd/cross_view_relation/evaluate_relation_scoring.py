import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from representation_screening import (  # noqa: E402
    DEFAULT_REPRESENTATIONS,
    binary_metrics,
    file_features,
    pca_fit,
    read_manifest,
    sample_files,
    standardize_apply,
    standardize_fit,
)


DEFAULT_PAIRS = [
    "ap:cyclo",
    "iq:bispectrum",
    "iq:cyclo",
    "fft:stft",
    "fft:cepstral",
    "stft:cepstral",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Tx1-train-fitted cross-view relation scores."
    )
    parser.add_argument("--manifest-dir", default="cross_view_relation/manifests")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--normal-manifest", default=None)
    parser.add_argument("--abnormal-manifests", nargs="*", default=None)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", default="results/cross_view_relation/relation_scoring")
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
    return parser.parse_args()


def resolve_default_manifests(args):
    manifest_dir = Path(args.manifest_dir).resolve()
    train = Path(args.train_manifest).resolve() if args.train_manifest else manifest_dir / "tx1_train_80_seed42_rebased.txt"
    normal = Path(args.normal_manifest).resolve() if args.normal_manifest else manifest_dir / "tx1_test_20_seed42_rebased.txt"
    if args.abnormal_manifests:
        abnormal = [Path(p).resolve() for p in args.abnormal_manifests]
    else:
        abnormal = sorted(manifest_dir.glob("tx*_all_rebased.txt"))
        abnormal = [p for p in abnormal if p.name.lower() != "tx1_all_rebased.txt"]
    return train, normal, abnormal


def parse_pairs(pairs):
    parsed = []
    reps = set()
    for item in pairs:
        if ":" not in item:
            raise ValueError(f"pair must use rep_a:rep_b form: {item}")
        a, b = item.split(":", 1)
        if a not in DEFAULT_REPRESENTATIONS or b not in DEFAULT_REPRESENTATIONS:
            raise ValueError(f"unknown representation pair: {item}")
        parsed.append((a, b))
        reps.add(a)
        reps.add(b)
    return parsed, sorted(reps)


def load_feature_matrices(args, representations):
    train_manifest, normal_manifest, abnormal_manifests = resolve_default_manifests(args)
    train_files = sample_files(read_manifest(train_manifest), args.max_files_train, args.seed + 1)
    normal_files = sample_files(read_manifest(normal_manifest), args.max_files_normal, args.seed + 2)
    anomaly_files_by_tx = {}
    for idx, manifest in enumerate(abnormal_manifests):
        tx = manifest.name.split("_")[0].upper().replace("TX", "Tx")
        anomaly_files_by_tx[tx] = sample_files(
            read_manifest(manifest),
            args.max_files_anomaly_per_device,
            args.seed + 100 + idx,
        )

    jobs = [("train", "Tx1", p) for p in train_files]
    jobs += [("normal", "Tx1", p) for p in normal_files]
    for tx, files in anomaly_files_by_tx.items():
        jobs += [("anomaly", tx, p) for p in files]

    args.representations = representations
    matrices = {split: {rep: [] for rep in representations} for split in ["train", "normal"]}
    anomaly = {tx: {rep: [] for rep in representations} for tx in anomaly_files_by_tx}

    start = time.time()
    print(
        f"[relation] files={len(jobs)} train={len(train_files)} "
        f"normal={len(normal_files)} anomaly={sum(len(v) for v in anomaly_files_by_tx.values())}"
    )
    print(f"[relation] pairs={','.join(args.pairs)}")
    for idx, (split, tx, path) in enumerate(jobs, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(jobs):
            print(f"[relation] {idx}/{len(jobs)} elapsed={time.time() - start:.1f}s {split}:{tx}:{path.name}")
        feats = file_features(path, args)
        for rep, values in feats.items():
            if split == "anomaly":
                anomaly[tx][rep].append(values)
            else:
                matrices[split][rep].append(values)

    train = {rep: np.vstack(rows) for rep, rows in matrices["train"].items()}
    normal = {rep: np.vstack(rows) for rep, rows in matrices["normal"].items()}
    anomaly = {tx: {rep: np.vstack(rows) for rep, rows in by_rep.items()} for tx, by_rep in anomaly.items()}
    config = {
        "train_manifest": str(train_manifest),
        "normal_manifest": str(normal_manifest),
        "abnormal_manifests": [str(p) for p in abnormal_manifests],
        "train_files": len(train_files),
        "normal_files": len(normal_files),
        "anomaly_files_by_tx": {tx: len(files) for tx, files in anomaly_files_by_tx.items()},
    }
    return train, normal, anomaly, config


def fit_rep_transforms(train, reps, components):
    models = {}
    for rep in reps:
        mean, std = standardize_fit(train[rep])
        z = standardize_apply(train[rep], mean, std)
        pca = pca_fit(z, components)
        models[rep] = {"mean": mean, "std": std, "pca": pca}
    return models


def transform_rep(x, model):
    z = standardize_apply(x, model["mean"], model["std"])
    return z @ model["pca"].T


def relation_features(a, b):
    eps = 1e-12
    diff = np.abs(a - b)
    prod = a * b
    cosine = np.sum(a * b, axis=1, keepdims=True) / (
        np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True) + eps
    )
    return np.concatenate([diff, prod, 1.0 - cosine], axis=1)


def build_pair_features(mats, rep_models, pair):
    a, b = pair
    za = transform_rep(mats[a], rep_models[a])
    zb = transform_rep(mats[b], rep_models[b])
    return relation_features(za, zb)


def score_zdist(z):
    return np.mean(z**2, axis=1)


def score_pca_residual(z, components):
    centered = z - np.mean(z, axis=0, keepdims=True)
    proj = centered @ components.T
    recon = proj @ components
    return np.mean((centered - recon) ** 2, axis=1)


def evaluate_scores(train_scores, normal_scores, anomaly_scores_by_tx, threshold_quantile):
    threshold = float(np.quantile(train_scores, threshold_quantile))
    all_anom = np.concatenate(list(anomaly_scores_by_tx.values()))
    y = np.concatenate([np.zeros(len(normal_scores), dtype=np.int64), np.ones(len(all_anom), dtype=np.int64)])
    scores = np.concatenate([normal_scores, all_anom])
    overall = binary_metrics(y, (scores > threshold).astype(np.int64), scores)
    per_tx = {}
    for tx, tx_scores in anomaly_scores_by_tx.items():
        tx_y = np.concatenate([np.zeros(len(normal_scores), dtype=np.int64), np.ones(len(tx_scores), dtype=np.int64)])
        tx_all = np.concatenate([normal_scores, tx_scores])
        per_tx[tx] = binary_metrics(tx_y, (tx_all > threshold).astype(np.int64), tx_all)
    return threshold, overall, per_tx


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    start = time.time()
    pairs, reps = parse_pairs(args.pairs)
    train, normal, anomaly, config = load_feature_matrices(args, reps)
    rep_models = fit_rep_transforms(train, reps, args.pca_components)

    metric_rows = []
    per_tx_rows = []
    for pair in pairs:
        label = f"{pair[0]}:{pair[1]}"
        train_rel = build_pair_features(train, rep_models, pair)
        normal_rel = build_pair_features(normal, rep_models, pair)
        anomaly_rel = {tx: build_pair_features(by_rep, rep_models, pair) for tx, by_rep in anomaly.items()}

        mean, std = standardize_fit(train_rel)
        train_z = standardize_apply(train_rel, mean, std)
        normal_z = standardize_apply(normal_rel, mean, std)
        anomaly_z = {tx: standardize_apply(x, mean, std) for tx, x in anomaly_rel.items()}
        rel_pca = pca_fit(train_z, args.relation_pca_components)

        methods = {
            "relation_zdist": (
                score_zdist(train_z),
                score_zdist(normal_z),
                {tx: score_zdist(z) for tx, z in anomaly_z.items()},
            ),
            "relation_pca_residual": (
                score_pca_residual(train_z, rel_pca),
                score_pca_residual(normal_z, rel_pca),
                {tx: score_pca_residual(z, rel_pca) for tx, z in anomaly_z.items()},
            ),
        }
        for method, (train_scores, normal_scores, anomaly_scores) in methods.items():
            threshold, metrics, tx_metrics = evaluate_scores(
                train_scores,
                normal_scores,
                anomaly_scores,
                args.threshold_quantile,
            )
            metric_rows.append(
                {
                    "pair": label,
                    "method": method,
                    "relation_dim": int(train_rel.shape[1]),
                    "pca_components": args.pca_components,
                    "relation_pca_components": int(rel_pca.shape[0]),
                    "threshold_quantile": args.threshold_quantile,
                    "threshold": threshold,
                    **metrics,
                }
            )
            for tx, row in tx_metrics.items():
                per_tx_rows.append(
                    {
                        "pair": label,
                        "method": method,
                        "tx": tx,
                        "threshold": threshold,
                        **row,
                    }
                )

    output_dir = Path(args.output_dir).resolve()
    write_csv(output_dir / "relation_metrics.csv", metric_rows)
    write_csv(output_dir / "relation_per_tx_metrics.csv", per_tx_rows)
    run_config = vars(args).copy()
    run_config.update(config)
    run_config["representations_used"] = reps
    run_config["elapsed_seconds"] = time.time() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("[relation] top metrics")
    for row in sorted(metric_rows, key=lambda r: (r["auc"], r["f1"]), reverse=True)[:8]:
        print(
            f"  {row['pair']:18s} {row['method']:22s} "
            f"auc={row['auc']:.6f} f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}"
        )
    print(f"[relation] wrote: {output_dir}")


if __name__ == "__main__":
    main()
