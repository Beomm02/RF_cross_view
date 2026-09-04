import argparse
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
    score_zdist,
    standardize_apply,
    standardize_fit,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a known-normal Tx1-Tx4 representation-relation experiment. "
            "The fitted relation model only sees Tx1-Tx4 train files; Tx1-Tx4 holdout "
            "files test normal-relation stability and Tx5-Tx8 files test unseen anomalies."
        )
    )
    parser.add_argument("--data-root", default="../../data")
    parser.add_argument("--output-dir", default="results/cross_view_relation/known4_relation_generalization_seed42_w16")
    parser.add_argument("--manifest-dir", default="cross_view_relation/manifests_known4_relation_seed42")
    parser.add_argument("--feature-cache", default="artifacts/cross_view_relation/known4_relation_generalization_seed42_w16.npz")
    parser.add_argument("--source-feature-cache", default="artifacts/cross_view_relation/iq_feature_relation_full_seed42_w16.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--known-txs", nargs="+", default=["Tx1", "Tx2", "Tx3", "Tx4"])
    parser.add_argument("--unknown-txs", nargs="+", default=["Tx5", "Tx6", "Tx7", "Tx8"])
    parser.add_argument("--train-count-per-known-tx", type=int, default=400)
    parser.add_argument("--known-test-count-per-tx", type=int, default=100)
    parser.add_argument("--max-unknown-files-per-device", type=int, default=0)
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


def tx_sort_key(tx: str) -> tuple[int, str]:
    digits = "".join(ch for ch in tx if ch.isdigit())
    return (int(digits) if digits else 10_000, tx)


def write_manifest(path: Path, files: list[Path], relative_to: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in files:
            f.write(str(item.resolve().relative_to(relative_to.resolve()).as_posix()) + "\n")


def split_tx_files(files: list[Path], tx: str, args) -> tuple[np.ndarray, np.ndarray]:
    expected = args.train_count_per_known_tx + args.known_test_count_per_tx
    if len(files) < expected:
        raise ValueError(f"{tx} has {len(files)} files, expected at least {expected}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(files))
    train_idx = np.sort(order[: args.train_count_per_known_tx])
    test_idx = np.sort(order[args.train_count_per_known_tx : expected])
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def make_split(args):
    data_root = Path(args.data_root).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()

    split = {"known_train": {}, "known_test": {}, "unknown_test": {}}
    split_indices = {"known_train": {}, "known_test": {}, "unknown_test": {}}
    split_rows = []

    for tx in sorted(args.known_txs, key=tx_sort_key):
        files = sorted((data_root / tx).glob("*.mat"))
        train_idx, test_idx = split_tx_files(files, tx, args)
        train_files = [files[int(i)].resolve() for i in train_idx]
        test_files = [files[int(i)].resolve() for i in test_idx]
        split["known_train"][tx] = train_files
        split["known_test"][tx] = test_files
        split_indices["known_train"][tx] = train_idx
        split_indices["known_test"][tx] = test_idx
        write_manifest(
            manifest_dir / f"{tx.lower()}_train_{args.train_count_per_known_tx}_seed{args.seed}.txt",
            train_files,
            data_root,
        )
        write_manifest(
            manifest_dir / f"{tx.lower()}_known_test_{args.known_test_count_per_tx}_seed{args.seed}.txt",
            test_files,
            data_root,
        )
        split_rows.append({"split": "known_train", "tx": tx, "count": len(train_files)})
        split_rows.append({"split": "known_test", "tx": tx, "count": len(test_files)})

    rng = np.random.default_rng(args.seed + 9000)
    for tx in sorted(args.unknown_txs, key=tx_sort_key):
        files = sorted((data_root / tx).glob("*.mat"))
        indices = np.arange(len(files), dtype=np.int64)
        if args.max_unknown_files_per_device and len(files) > args.max_unknown_files_per_device:
            indices = np.sort(rng.choice(indices, size=args.max_unknown_files_per_device, replace=False))
        selected = [files[int(i)].resolve() for i in indices]
        split["unknown_test"][tx] = selected
        split_indices["unknown_test"][tx] = indices
        suffix = "all" if not args.max_unknown_files_per_device else str(args.max_unknown_files_per_device)
        write_manifest(manifest_dir / f"{tx.lower()}_unknown_{suffix}_seed{args.seed}.txt", selected, data_root)
        split_rows.append({"split": "unknown_test", "tx": tx, "count": len(selected)})

    return data_root, manifest_dir, split, split_indices, split_rows


def load_from_previous_cache(args, split_indices):
    source_path = Path(args.source_feature_cache).resolve()
    if not source_path.exists() or args.force_extract:
        return None
    can_reuse_tx1_split = (
        args.seed == 42
        and args.train_count_per_known_tx == 400
        and args.known_test_count_per_tx == 100
        and "Tx1" in args.known_txs
    )
    if not can_reuse_tx1_split:
        print("[known4] source cache has Tx1 split-only arrays; extracting instead.", flush=True)
        return None

    loaded = np.load(source_path, allow_pickle=False)
    required = {f"train__{rep}" for rep in args.representations}
    required.update(f"holdout__{rep}" for rep in args.representations)
    for tx in list(args.known_txs) + list(args.unknown_txs):
        if tx == "Tx1":
            continue
        required.update(f"{tx}__{rep}" for rep in args.representations)
    missing = sorted(k for k in required if k not in loaded.files)
    if missing:
        print(f"[known4] source cache missing keys; extracting instead. first_missing={missing[:3]}", flush=True)
        return None

    print(f"[known4] loading reusable feature cache: {source_path}", flush=True)
    train_by_tx = {}
    known_by_tx = {}
    unknown_by_tx = {}
    for tx in sorted(args.known_txs, key=tx_sort_key):
        if tx == "Tx1":
            train_by_tx[tx] = {rep: loaded[f"train__{rep}"] for rep in args.representations}
            known_by_tx[tx] = {rep: loaded[f"holdout__{rep}"] for rep in args.representations}
        else:
            train_idx = split_indices["known_train"][tx]
            test_idx = split_indices["known_test"][tx]
            train_by_tx[tx] = {rep: loaded[f"{tx}__{rep}"][train_idx] for rep in args.representations}
            known_by_tx[tx] = {rep: loaded[f"{tx}__{rep}"][test_idx] for rep in args.representations}
    for tx in sorted(args.unknown_txs, key=tx_sort_key):
        indices = split_indices["unknown_test"][tx]
        unknown_by_tx[tx] = {rep: loaded[f"{tx}__{rep}"][indices] for rep in args.representations}
    return train_by_tx, known_by_tx, unknown_by_tx, str(source_path)


def extract_features(args, split):
    cache_path = Path(args.feature_cache).resolve()
    if cache_path.exists() and not args.force_extract:
        print(f"[known4] loading experiment feature cache: {cache_path}", flush=True)
        loaded = np.load(cache_path, allow_pickle=False)
        train_by_tx = {
            tx: {rep: loaded[f"known_train__{tx}__{rep}"] for rep in args.representations}
            for tx in args.known_txs
        }
        known_by_tx = {
            tx: {rep: loaded[f"known_test__{tx}__{rep}"] for rep in args.representations}
            for tx in args.known_txs
        }
        unknown_by_tx = {
            tx: {rep: loaded[f"unknown_test__{tx}__{rep}"] for rep in args.representations}
            for tx in args.unknown_txs
        }
        return train_by_tx, known_by_tx, unknown_by_tx, str(cache_path)

    jobs = []
    for split_name in ["known_train", "known_test", "unknown_test"]:
        for tx, files in split[split_name].items():
            jobs.extend((split_name, tx, path) for path in files)

    print(f"[known4] extracting features from {len(jobs)} files", flush=True)
    start = time.time()
    matrices = {
        split_name: {
            tx: {rep: [] for rep in args.representations}
            for tx in split[split_name]
        }
        for split_name in split
    }
    for idx, (split_name, tx, path) in enumerate(jobs, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(jobs):
            elapsed = time.time() - start
            print(f"[known4] feature {idx}/{len(jobs)} elapsed={elapsed:.1f}s {split_name}:{tx}:{path.name}", flush=True)
        features = file_features(path, args)
        for rep, values in features.items():
            matrices[split_name][tx][rep].append(values)

    train_by_tx = {
        tx: {rep: np.vstack(values) for rep, values in by_rep.items()}
        for tx, by_rep in matrices["known_train"].items()
    }
    known_by_tx = {
        tx: {rep: np.vstack(values) for rep, values in by_rep.items()}
        for tx, by_rep in matrices["known_test"].items()
    }
    unknown_by_tx = {
        tx: {rep: np.vstack(values) for rep, values in by_rep.items()}
        for tx, by_rep in matrices["unknown_test"].items()
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for tx, by_rep in train_by_tx.items():
        for rep, values in by_rep.items():
            payload[f"known_train__{tx}__{rep}"] = values
    for tx, by_rep in known_by_tx.items():
        for rep, values in by_rep.items():
            payload[f"known_test__{tx}__{rep}"] = values
    for tx, by_rep in unknown_by_tx.items():
        for rep, values in by_rep.items():
            payload[f"unknown_test__{tx}__{rep}"] = values
    np.savez_compressed(cache_path, **payload)
    print(f"[known4] saved local experiment cache: {cache_path}", flush=True)
    return train_by_tx, known_by_tx, unknown_by_tx, str(cache_path)


def stack_by_rep(by_tx, reps):
    return {
        rep: np.vstack([by_tx[tx][rep] for tx in sorted(by_tx, key=tx_sort_key)])
        for rep in reps
    }


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


def fit_pca_residual_model(train_z: np.ndarray, components: int) -> dict:
    center = np.mean(train_z, axis=0, keepdims=True)
    pca = pca_fit(train_z, components)
    return {"center": center, "components": pca}


def score_pca_residual_fixed(z: np.ndarray, model: dict) -> np.ndarray:
    centered = z - model["center"]
    components = model["components"]
    proj = centered @ components.T
    recon = proj @ components
    return np.mean((centered - recon) ** 2, axis=1)


def summarize_scores(train_scores, known_scores, unknown_scores_by_tx):
    unknown_scores = np.concatenate(list(unknown_scores_by_tx.values()))
    return {
        "train_score_mean": float(np.mean(train_scores)),
        "known_score_mean": float(np.mean(known_scores)),
        "unknown_score_mean": float(np.mean(unknown_scores)),
        "known_unknown_mean_gap": float(np.mean(unknown_scores) - np.mean(known_scores)),
        "known_score_p95": float(np.quantile(known_scores, 0.95)),
        "unknown_score_p50": float(np.quantile(unknown_scores, 0.50)),
    }


def evaluate_thresholds(train_scores, known_scores_by_tx, unknown_scores_by_tx, threshold_quantiles):
    known_scores = np.concatenate(list(known_scores_by_tx.values()))
    unknown_scores = np.concatenate(list(unknown_scores_by_tx.values()))
    y_all = np.concatenate(
        [np.zeros(len(known_scores), dtype=np.int64), np.ones(len(unknown_scores), dtype=np.int64)]
    )
    scores_all = np.concatenate([known_scores, unknown_scores])
    rows = []
    known_rows = []
    unknown_rows = []
    score_summary = summarize_scores(train_scores, known_scores, unknown_scores_by_tx)
    for q in threshold_quantiles:
        threshold = float(np.quantile(train_scores, q))
        metrics = binary_metrics(y_all, (scores_all > threshold).astype(np.int64), scores_all)
        rows.append({"threshold_quantile": q, "threshold": threshold, **score_summary, **metrics})
        for tx, scores in known_scores_by_tx.items():
            false_rejects = int((scores > threshold).sum())
            known_rows.append(
                {
                    "tx": tx,
                    "threshold_quantile": q,
                    "threshold": threshold,
                    "known_files": int(len(scores)),
                    "false_rejects": false_rejects,
                    "known_accepts": int(len(scores) - false_rejects),
                    "false_reject_rate": false_rejects / max(len(scores), 1),
                    "score_mean": float(np.mean(scores)),
                    "score_p95": float(np.quantile(scores, 0.95)),
                }
            )
        for tx, scores in unknown_scores_by_tx.items():
            y_tx = np.concatenate(
                [np.zeros(len(known_scores), dtype=np.int64), np.ones(len(scores), dtype=np.int64)]
            )
            scores_tx = np.concatenate([known_scores, scores])
            tx_metrics = binary_metrics(y_tx, (scores_tx > threshold).astype(np.int64), scores_tx)
            unknown_rows.append({"tx": tx, "threshold_quantile": q, "threshold": threshold, **tx_metrics})
    return rows, known_rows, unknown_rows


def score_single_representations(train, known_by_tx, unknown_by_tx, args):
    metric_rows = []
    known_tx_rows = []
    unknown_tx_rows = []
    score_outputs = {}
    for rep in args.representations:
        mean, std = standardize_fit(train[rep])
        train_z = standardize_apply(train[rep], mean, std)
        known_z = {tx: standardize_apply(by_rep[rep], mean, std) for tx, by_rep in known_by_tx.items()}
        unknown_z = {tx: standardize_apply(by_rep[rep], mean, std) for tx, by_rep in unknown_by_tx.items()}
        pca = fit_pca_residual_model(train_z, args.pca_components)
        shrink = fit_shrinkage_mahalanobis(train_z, args.shrinkage)
        methods = {
            "zdist": (
                score_zdist(train_z),
                {tx: score_zdist(z) for tx, z in known_z.items()},
                {tx: score_zdist(z) for tx, z in unknown_z.items()},
            ),
            "pca_residual": (
                score_pca_residual_fixed(train_z, pca),
                {tx: score_pca_residual_fixed(z, pca) for tx, z in known_z.items()},
                {tx: score_pca_residual_fixed(z, pca) for tx, z in unknown_z.items()},
            ),
            "shrinkage_mahalanobis": (
                score_shrinkage_mahalanobis(train_z, shrink),
                {tx: score_shrinkage_mahalanobis(z, shrink) for tx, z in known_z.items()},
                {tx: score_shrinkage_mahalanobis(z, shrink) for tx, z in unknown_z.items()},
            ),
        }
        for method, scores in methods.items():
            label = f"{rep}:{method}"
            score_outputs[label] = scores
            rows, known_rows, unknown_rows = evaluate_thresholds(*scores, args.threshold_quantiles)
            for row in rows:
                metric_rows.append({"representation": rep, "method": method, **row})
            for row in known_rows:
                known_tx_rows.append({"representation": rep, "method": method, **row})
            for row in unknown_rows:
                unknown_tx_rows.append({"representation": rep, "method": method, **row})
    return metric_rows, known_tx_rows, unknown_tx_rows, score_outputs


def score_relation_pairs(train, known_by_tx, unknown_by_tx, args):
    rep_models = fit_rep_transforms(train, args.representations, args.pca_components)
    pairs = list(itertools.combinations(args.representations, 2))
    metric_rows = []
    known_tx_rows = []
    unknown_tx_rows = []
    score_outputs = {}
    for pair in pairs:
        pair_name = f"{pair[0]}:{pair[1]}"
        train_rel = build_pair_features(train, rep_models, pair)
        known_rel = {tx: build_pair_features(by_rep, rep_models, pair) for tx, by_rep in known_by_tx.items()}
        unknown_rel = {tx: build_pair_features(by_rep, rep_models, pair) for tx, by_rep in unknown_by_tx.items()}
        mean, std = standardize_fit(train_rel)
        train_z = standardize_apply(train_rel, mean, std)
        known_z = {tx: standardize_apply(x, mean, std) for tx, x in known_rel.items()}
        unknown_z = {tx: standardize_apply(x, mean, std) for tx, x in unknown_rel.items()}
        pca = fit_pca_residual_model(train_z, args.relation_pca_components)
        shrink = fit_shrinkage_mahalanobis(train_z, args.shrinkage)
        methods = {
            "relation_zdist": (
                score_zdist(train_z),
                {tx: score_zdist(z) for tx, z in known_z.items()},
                {tx: score_zdist(z) for tx, z in unknown_z.items()},
            ),
            "relation_pca_residual": (
                score_pca_residual_fixed(train_z, pca),
                {tx: score_pca_residual_fixed(z, pca) for tx, z in known_z.items()},
                {tx: score_pca_residual_fixed(z, pca) for tx, z in unknown_z.items()},
            ),
            "relation_shrinkage_mahalanobis": (
                score_shrinkage_mahalanobis(train_z, shrink),
                {tx: score_shrinkage_mahalanobis(z, shrink) for tx, z in known_z.items()},
                {tx: score_shrinkage_mahalanobis(z, shrink) for tx, z in unknown_z.items()},
            ),
        }
        for method, scores in methods.items():
            label = f"{pair_name}:{method}"
            score_outputs[label] = scores
            rows, known_rows, unknown_rows = evaluate_thresholds(*scores, args.threshold_quantiles)
            for row in rows:
                metric_rows.append({"pair": pair_name, "method": method, "relation_dim": int(train_rel.shape[1]), **row})
            for row in known_rows:
                known_tx_rows.append({"pair": pair_name, "method": method, **row})
            for row in unknown_rows:
                unknown_tx_rows.append({"pair": pair_name, "method": method, **row})
    return metric_rows, known_tx_rows, unknown_tx_rows, score_outputs


def empirical_rank(train_scores, scores):
    train_scores = np.sort(np.asarray(train_scores, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    return np.searchsorted(train_scores, scores, side="right") / max(len(train_scores), 1)


def fuse_score_outputs(selected_labels, score_outputs):
    train_cols = []
    known_cols = {}
    unknown_cols = {}
    for label in selected_labels:
        train_scores, known_scores_by_tx, unknown_scores_by_tx = score_outputs[label]
        train_cols.append(empirical_rank(train_scores, train_scores))
        for tx, tx_scores in known_scores_by_tx.items():
            known_cols.setdefault(tx, []).append(empirical_rank(train_scores, tx_scores))
        for tx, tx_scores in unknown_scores_by_tx.items():
            unknown_cols.setdefault(tx, []).append(empirical_rank(train_scores, tx_scores))
    train_rank = np.column_stack(train_cols)
    known_rank = {tx: np.column_stack(cols) for tx, cols in known_cols.items()}
    unknown_rank = {tx: np.column_stack(cols) for tx, cols in unknown_cols.items()}
    return (
        np.mean(train_rank, axis=1),
        {tx: np.mean(x, axis=1) for tx, x in known_rank.items()},
        {tx: np.mean(x, axis=1) for tx, x in unknown_rank.items()},
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
    known_tx_rows = []
    unknown_tx_rows = []
    for fusion_name, labels in fusion_defs.items():
        if not labels:
            continue
        scores = fuse_score_outputs(labels, all_outputs)
        rows, known_rows, unknown_rows = evaluate_thresholds(*scores, args.threshold_quantiles)
        for row in rows:
            metric_rows.append(
                {
                    "fusion": fusion_name,
                    "component_count": len(labels),
                    "components": "|".join(labels),
                    **row,
                }
            )
        for row in known_rows:
            known_tx_rows.append({"fusion": fusion_name, "component_count": len(labels), **row})
        for row in unknown_rows:
            unknown_tx_rows.append({"fusion": fusion_name, "component_count": len(labels), **row})

    selection_rows = []
    for group_name, pairs in [("low_cka", low_pairs), ("high_cka", high_pairs)]:
        for pair in sorted(pairs):
            selection_rows.append({"group": group_name, "pair": pair})
    return metric_rows, known_tx_rows, unknown_tx_rows, selection_rows


def relation_stability(train_by_tx, pooled_relation_rows, args):
    by_tx_rows = []
    for tx in sorted(train_by_tx, key=tx_sort_key):
        rows = relation_diagnostics(train_by_tx[tx], args)
        for row in rows:
            by_tx_rows.append({"tx": tx, **row})

    pooled = {f"{row['rep_a']}:{row['rep_b']}": row for row in pooled_relation_rows}
    grouped = {}
    for row in by_tx_rows:
        key = f"{row['rep_a']}:{row['rep_b']}"
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, rows in sorted(grouped.items()):
        cka_values = np.asarray([float(row["linear_cka"]) for row in rows], dtype=np.float64)
        split_values = np.asarray([float(row["split_cca_mean5"]) for row in rows], dtype=np.float64)
        pooled_row = pooled[key]
        summary_rows.append(
            {
                "pair": key,
                "pooled_linear_cka": pooled_row["linear_cka"],
                "pooled_split_cca_mean5": pooled_row["split_cca_mean5"],
                "known_tx_linear_cka_mean": float(np.mean(cka_values)),
                "known_tx_linear_cka_std": float(np.std(cka_values)),
                "known_tx_linear_cka_min": float(np.min(cka_values)),
                "known_tx_linear_cka_max": float(np.max(cka_values)),
                "known_tx_split_cca_mean5_mean": float(np.mean(split_values)),
                "known_tx_split_cca_mean5_std": float(np.std(split_values)),
            }
        )
    return by_tx_rows, summary_rows


def best_rows(rows, primary="auc", n=10):
    return sorted(rows, key=lambda r: (float(r[primary]), float(r.get("f1", 0.0))), reverse=True)[:n]


def main():
    args = parse_args()
    started = time.time()
    data_root, manifest_dir, split, split_indices, split_rows = make_split(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "split_summary.csv", split_rows)

    loaded = load_from_previous_cache(args, split_indices)
    if loaded is None:
        train_by_tx, known_by_tx, unknown_by_tx, cache_path = extract_features(args, split)
    else:
        train_by_tx, known_by_tx, unknown_by_tx, cache_path = loaded

    train = stack_by_rep(train_by_tx, args.representations)
    known_test = stack_by_rep(known_by_tx, args.representations)
    unknown_test = stack_by_rep(unknown_by_tx, args.representations)

    print("[known4] running pooled normal CKA/CCA diagnostics", flush=True)
    pooled_relation_rows = relation_diagnostics(train, args)
    write_csv(output_dir / "relation_diagnostics_pooled.csv", pooled_relation_rows)

    print("[known4] running per-known-Tx relation stability diagnostics", flush=True)
    by_tx_relation_rows, stability_rows = relation_stability(train_by_tx, pooled_relation_rows, args)
    write_csv(output_dir / "relation_diagnostics_by_known_tx.csv", by_tx_relation_rows)
    write_csv(output_dir / "relation_stability_summary.csv", stability_rows)

    print("[known4] scoring single representations", flush=True)
    single_rows, single_known_rows, single_unknown_rows, single_outputs = score_single_representations(
        train,
        known_by_tx,
        unknown_by_tx,
        args,
    )
    write_csv(output_dir / "single_representation_metrics.csv", single_rows)
    write_csv(output_dir / "single_representation_known_tx_metrics.csv", single_known_rows)
    write_csv(output_dir / "single_representation_unknown_tx_metrics.csv", single_unknown_rows)

    print("[known4] scoring relation pairs", flush=True)
    relation_rows, relation_known_rows, relation_unknown_rows, relation_outputs = score_relation_pairs(
        train,
        known_by_tx,
        unknown_by_tx,
        args,
    )
    write_csv(output_dir / "relation_pair_metrics.csv", relation_rows)
    write_csv(output_dir / "relation_pair_known_tx_metrics.csv", relation_known_rows)
    write_csv(output_dir / "relation_pair_unknown_tx_metrics.csv", relation_unknown_rows)

    print("[known4] scoring train-rank fusions", flush=True)
    fusion_rows, fusion_known_rows, fusion_unknown_rows, selection_rows = score_fusions(
        single_outputs,
        relation_outputs,
        pooled_relation_rows,
        args,
    )
    write_csv(output_dir / "fusion_metrics.csv", fusion_rows)
    write_csv(output_dir / "fusion_known_tx_metrics.csv", fusion_known_rows)
    write_csv(output_dir / "fusion_unknown_tx_metrics.csv", fusion_unknown_rows)
    write_csv(output_dir / "cka_selected_pairs.csv", selection_rows)

    summary = {
        "data_root_name": data_root.name,
        "manifest_dir": str(manifest_dir),
        "feature_cache_used": str(cache_path),
        "elapsed_seconds": time.time() - started,
        "known_train_total": int(sum(len(v) for v in split["known_train"].values())),
        "known_test_total": int(sum(len(v) for v in split["known_test"].values())),
        "unknown_test_total": int(sum(len(v) for v in split["unknown_test"].values())),
        "known_txs": args.known_txs,
        "unknown_txs": args.unknown_txs,
        "representations": args.representations,
        "lowest_pooled_cka_pairs": sorted(pooled_relation_rows, key=lambda r: float(r["linear_cka"]))[:5],
        "highest_pooled_cka_pairs": sorted(pooled_relation_rows, key=lambda r: float(r["linear_cka"]), reverse=True)[:5],
        "best_single": best_rows(single_rows, n=5),
        "best_relation_pair": best_rows(relation_rows, n=10),
        "best_fusion": best_rows(fusion_rows, n=5),
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    run_config = vars(args).copy()
    run_config.update(
        {
            "data_root": str(data_root),
            "manifest_dir": str(manifest_dir),
            "output_dir": str(output_dir),
            "feature_cache_used": str(cache_path),
            "elapsed_seconds": time.time() - started,
            "pooled_train_shape_by_rep": {rep: list(values.shape) for rep, values in train.items()},
            "pooled_known_shape_by_rep": {rep: list(values.shape) for rep, values in known_test.items()},
            "pooled_unknown_shape_by_rep": {rep: list(values.shape) for rep, values in unknown_test.items()},
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("[known4] best single")
    for row in summary["best_single"]:
        print(
            f"  {row['representation']:12s} {row['method']:24s} q={row['threshold_quantile']:.2f} "
            f"auc={row['auc']:.6f} f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}",
            flush=True,
        )
    print("[known4] best relation")
    for row in summary["best_relation_pair"][:5]:
        print(
            f"  {row['pair']:18s} {row['method']:30s} q={row['threshold_quantile']:.2f} "
            f"auc={row['auc']:.6f} f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}",
            flush=True,
        )
    print("[known4] best fusion")
    for row in summary["best_fusion"]:
        print(
            f"  {row['fusion']:34s} q={row['threshold_quantile']:.2f} "
            f"auc={row['auc']:.6f} f1={row['f1']:.6f} fp={row['fp']} fn={row['fn']}",
            flush=True,
        )
    print(f"[known4] wrote: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
