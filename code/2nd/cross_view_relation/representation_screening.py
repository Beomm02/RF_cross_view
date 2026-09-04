import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fft import dct


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from preprocessing import load_iq_from_mat, normalize_iq, validate_iq  # noqa: E402


DEFAULT_REPRESENTATIONS = [
    "iq",
    "ap",
    "fft",
    "stft",
    "cepstral",
    "cyclo",
    "hos",
    "bispectrum",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exploratory representation screening and Tx1-train CCA/CKA diagnostics."
    )
    parser.add_argument("--manifest-dir", default="cross_view_relation/manifests")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--normal-manifest", default=None)
    parser.add_argument("--abnormal-manifests", nargs="*", default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=DEFAULT_REPRESENTATIONS,
        choices=DEFAULT_REPRESENTATIONS,
    )
    parser.add_argument(
        "--output-dir",
        default="results/cross_view_relation/representation_screening",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mat-key", default="rxData")
    parser.add_argument("--norm-mode", default="power")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-windows-per-file", type=int, default=16)
    parser.add_argument("--max-files-train", type=int, default=64)
    parser.add_argument("--max-files-normal", type=int, default=64)
    parser.add_argument("--max-files-anomaly-per-device", type=int, default=64)
    parser.add_argument("--fft-bins", type=int, default=64)
    parser.add_argument("--stft-nperseg", type=int, default=128)
    parser.add_argument("--stft-noverlap", type=int, default=96)
    parser.add_argument("--stft-freq-bins", type=int, default=16)
    parser.add_argument("--stft-time-bins", type=int, default=8)
    parser.add_argument("--cepstral-coeffs", type=int, default=32)
    parser.add_argument(
        "--cyclo-shifts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--bispectrum-bins", type=int, default=12)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    return parser.parse_args()


def read_manifest(path: Path) -> list[Path]:
    files = []
    base = path.parent
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            item = Path(line)
            if not item.is_absolute():
                item = (base / item).resolve()
            if item.suffix.lower() == ".mat":
                files.append(item)
    return files


def sample_files(files: list[Path], max_count: int | None, seed: int) -> list[Path]:
    files = list(files)
    if max_count is None or max_count <= 0 or len(files) <= max_count:
        return files
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(files), size=max_count, replace=False))
    return [files[int(i)] for i in indices]


def robust_stats(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(np.mean(values))
    std = float(np.std(values) + 1e-12)
    centered = values - mean
    skew = float(np.mean(centered**3) / (std**3))
    kurt = float(np.mean(centered**4) / (std**4))
    qs = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return np.asarray([mean, std, skew, kurt, *qs], dtype=np.float64)


def block_reduce_1d(x: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    chunks = np.array_split(values, bins)
    return np.asarray([float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in chunks])


def block_reduce_2d(x: np.ndarray, freq_bins: int, time_bins: int) -> np.ndarray:
    rows = np.array_split(np.asarray(x, dtype=np.float64), freq_bins, axis=0)
    pooled = []
    for row_block in rows:
        cols = np.array_split(row_block, time_bins, axis=1)
        for block in cols:
            pooled.append(float(np.mean(block)) if block.size else 0.0)
    return np.asarray(pooled, dtype=np.float64)


def spectral_shape_features(logmag: np.ndarray) -> np.ndarray:
    mag = np.expm1(np.asarray(logmag, dtype=np.float64))
    mag = np.maximum(mag, 0.0)
    total = float(np.sum(mag) + 1e-12)
    p = mag / total
    idx = np.linspace(-0.5, 0.5, len(p), endpoint=False)
    centroid = float(np.sum(idx * p))
    spread = float(np.sqrt(np.sum(((idx - centroid) ** 2) * p)))
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / math.log(max(len(p), 2)))
    flatness = float(np.exp(np.mean(np.log(mag + 1e-12))) / (np.mean(mag) + 1e-12))
    cdf = np.cumsum(p)
    rolloff_idx = int(np.searchsorted(cdf, 0.95, side="left"))
    rolloff = float(idx[min(rolloff_idx, len(idx) - 1)])
    peak_idx = int(np.argmax(mag))
    peak_loc = float(idx[peak_idx])
    peak_rel = float(mag[peak_idx] / (np.mean(mag) + 1e-12))
    bands = np.asarray([np.sum(chunk) / total for chunk in np.array_split(mag, 4)], dtype=np.float64)
    return np.asarray(
        [centroid, spread, entropy, flatness, rolloff, peak_loc, peak_rel, *bands],
        dtype=np.float64,
    )


def complex_window(window: np.ndarray) -> np.ndarray:
    return window[:, 0].astype(np.float64) + 1j * window[:, 1].astype(np.float64)


def iq_features(window: np.ndarray, args) -> np.ndarray:
    i = window[:, 0]
    q = window[:, 1]
    power = i**2 + q**2
    corr = float(np.corrcoef(i, q)[0, 1]) if np.std(i) > 1e-12 and np.std(q) > 1e-12 else 0.0
    cross = np.asarray(
        [
            corr,
            float(np.mean(i * q)),
            float(np.mean(power)),
            float(np.std(power)),
        ],
        dtype=np.float64,
    )
    return np.concatenate([robust_stats(i), robust_stats(q), robust_stats(power), cross])


def ap_features(window: np.ndarray, args) -> np.ndarray:
    x = complex_window(window)
    amp = np.abs(x)
    phase = np.angle(x)
    phase_diff = np.diff(np.unwrap(phase), prepend=phase[0])
    circular = np.asarray(
        [
            float(np.mean(np.sin(phase))),
            float(np.mean(np.cos(phase))),
            float(np.std(np.sin(phase))),
            float(np.std(np.cos(phase))),
        ],
        dtype=np.float64,
    )
    return np.concatenate([robust_stats(amp), robust_stats(phase_diff), circular])


def fft_logmag(window: np.ndarray) -> np.ndarray:
    x = complex_window(window)
    spectrum = np.fft.fftshift(np.fft.fft(x))
    return np.log1p(np.abs(spectrum)).astype(np.float64)


def fft_features(window: np.ndarray, args) -> np.ndarray:
    logmag = fft_logmag(window)
    return np.concatenate(
        [
            block_reduce_1d(logmag, args.fft_bins),
            spectral_shape_features(logmag),
            robust_stats(logmag),
        ]
    )


def stft_features(window: np.ndarray, args) -> np.ndarray:
    x = complex_window(window)
    nperseg = min(args.stft_nperseg, len(x))
    noverlap = min(args.stft_noverlap, max(0, nperseg - 1))
    _, _, zxx = signal.stft(
        x,
        fs=1.0,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    logmag = np.log1p(np.fft.fftshift(np.abs(zxx), axes=0)).astype(np.float64)
    z = (logmag - np.mean(logmag)) / (np.std(logmag) + 1e-8)
    freq_profile = np.mean(z, axis=1)
    time_profile = np.mean(z, axis=0)
    return np.concatenate(
        [
            block_reduce_2d(z, args.stft_freq_bins, args.stft_time_bins),
            block_reduce_1d(freq_profile, args.stft_freq_bins),
            block_reduce_1d(time_profile, args.stft_time_bins),
            robust_stats(z),
        ]
    )


def cepstral_features(window: np.ndarray, args) -> np.ndarray:
    coeffs = dct(fft_logmag(window), type=2, norm="ortho")
    coeffs = coeffs[1 : args.cepstral_coeffs + 1]
    return np.asarray(coeffs, dtype=np.float64)


def cyclo_features(window: np.ndarray, args) -> np.ndarray:
    x = complex_window(window)
    spectrum = np.fft.fft(x)
    denom = float(np.sqrt(np.mean(np.abs(spectrum) ** 2)) + 1e-12)
    spectrum = spectrum / denom
    feats = []
    for shift in args.cyclo_shifts:
        if shift <= 0 or shift >= len(spectrum):
            continue
        prod = spectrum[:-shift] * np.conj(spectrum[shift:])
        mag = np.abs(prod)
        phase = np.angle(prod)
        feats.extend(
            [
                float(np.mean(mag)),
                float(np.std(mag)),
                float(np.quantile(mag, 0.95)),
                float(np.mean(prod.real)),
                float(np.mean(prod.imag)),
                float(np.abs(np.mean(np.exp(1j * phase)))),
            ]
        )
    return np.asarray(feats, dtype=np.float64)


def hos_features(window: np.ndarray, args) -> np.ndarray:
    x = complex_window(window)
    x = x - np.mean(x)
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    m20 = np.mean(x**2)
    m21 = np.mean(np.abs(x) ** 2)
    m40 = np.mean(x**4)
    m41 = np.mean((x**3) * np.conj(x))
    m42 = np.mean(np.abs(x) ** 4)
    c20 = m20
    c21 = m21
    c40 = m40 - 3 * (m20**2)
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - abs(m20) ** 2 - 2 * (m21**2)
    complex_vals = [m20, m40, m41, c20, c40, c41]
    feats = []
    for value in complex_vals:
        feats.extend([float(np.real(value)), float(np.imag(value)), float(abs(value))])
    feats.extend([float(m21), float(m42), float(c21), float(np.real(c42))])
    feats.extend(robust_stats(x.real).tolist())
    feats.extend(robust_stats(x.imag).tolist())
    feats.extend(robust_stats(np.abs(x)).tolist())
    return np.asarray(feats, dtype=np.float64)


def bispectrum_features(window: np.ndarray, args) -> np.ndarray:
    x = complex_window(window)
    spectrum = np.fft.fft(x)
    spectrum = spectrum / (np.sqrt(np.mean(np.abs(spectrum) ** 2)) + 1e-12)
    n = len(spectrum)
    bins = max(4, int(args.bispectrum_bins))
    indices = np.linspace(0, n - 1, bins, dtype=np.int64)
    values = []
    for ki in indices:
        for kj in indices:
            kk = int((ki + kj) % n)
            b = spectrum[ki] * spectrum[kj] * np.conj(spectrum[kk])
            values.append(np.log1p(abs(b)))
    values = np.asarray(values, dtype=np.float64)
    return np.concatenate([values, robust_stats(values)])


FEATURE_BUILDERS = {
    "iq": iq_features,
    "ap": ap_features,
    "fft": fft_features,
    "stft": stft_features,
    "cepstral": cepstral_features,
    "cyclo": cyclo_features,
    "hos": hos_features,
    "bispectrum": bispectrum_features,
}


def select_windows(iq: np.ndarray, args) -> list[np.ndarray]:
    validate_iq(iq, min_len=args.window_size)
    n_all = 1 + (iq.shape[0] - args.window_size) // args.stride
    n_select = min(args.max_windows_per_file, n_all)
    indices = np.linspace(0, n_all - 1, n_select, dtype=np.int64)
    windows = []
    for idx in indices:
        start = int(idx) * args.stride
        windows.append(iq[start : start + args.window_size])
    return windows


def file_features(path: Path, args) -> dict[str, np.ndarray]:
    iq = load_iq_from_mat(str(path), key=args.mat_key)
    iq = normalize_iq(iq, mode=args.norm_mode)
    windows = select_windows(iq, args)
    by_rep = {rep: [] for rep in args.representations}
    for window in windows:
        for rep in args.representations:
            values = FEATURE_BUILDERS[rep](window, args)
            by_rep[rep].append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
    out = {}
    for rep, rows in by_rep.items():
        matrix = np.vstack(rows).astype(np.float64)
        out[rep] = np.concatenate(
            [
                np.mean(matrix, axis=0),
                np.std(matrix, axis=0),
            ]
        )
    return out


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def pca_fit(z: np.ndarray, max_components: int) -> np.ndarray:
    centered = z - np.mean(z, axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(s > 1e-8))
    k = max(1, min(max_components, rank, z.shape[0] - 1, z.shape[1]))
    return vt[:k]


def score_zdist(z: np.ndarray) -> np.ndarray:
    return np.mean(z**2, axis=1)


def score_pca_residual(z: np.ndarray, components: np.ndarray) -> np.ndarray:
    centered = z - np.mean(z, axis=0, keepdims=True)
    proj = centered @ components.T
    recon = proj @ components
    residual = centered - recon
    return np.mean(residual**2, axis=1)


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty_like(sorted_ranks, dtype=np.float64)
    ranks[order] = sorted_ranks
    pos_rank_sum = ranks[y_true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "auc": roc_auc_score(y_true, scores),
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_representation(rep: str, train_x, normal_x, anomaly_by_tx, args) -> tuple[list[dict], list[dict]]:
    mean, std = standardize_fit(train_x)
    train_z = standardize_apply(train_x, mean, std)
    normal_z = standardize_apply(normal_x, mean, std)
    anomaly_z_by_tx = {tx: standardize_apply(x, mean, std) for tx, x in anomaly_by_tx.items()}
    components = pca_fit(train_z, args.pca_components)

    rows = []
    per_tx_rows = []
    scoring = {
        "zdist": (
            score_zdist(train_z),
            score_zdist(normal_z),
            {tx: score_zdist(z) for tx, z in anomaly_z_by_tx.items()},
        ),
        "pca_residual": (
            score_pca_residual(train_z, components),
            score_pca_residual(normal_z, components),
            {tx: score_pca_residual(z, components) for tx, z in anomaly_z_by_tx.items()},
        ),
    }

    for method, (train_scores, normal_scores, anomaly_scores_by_tx) in scoring.items():
        all_anomaly_scores = np.concatenate(list(anomaly_scores_by_tx.values()))
        y = np.concatenate(
            [
                np.zeros(len(normal_scores), dtype=np.int64),
                np.ones(len(all_anomaly_scores), dtype=np.int64),
            ]
        )
        scores = np.concatenate([normal_scores, all_anomaly_scores])
        threshold = float(np.quantile(train_scores, args.threshold_quantile))
        metrics = binary_metrics(y, (scores > threshold).astype(np.int64), scores)
        row = {
            "representation": rep,
            "method": method,
            "feature_dim": int(train_x.shape[1]),
            "pca_components": int(components.shape[0]),
            "train_files": int(len(train_scores)),
            "normal_files": int(len(normal_scores)),
            "anomaly_files": int(len(all_anomaly_scores)),
            "threshold_quantile": args.threshold_quantile,
            "threshold": threshold,
            **metrics,
        }
        rows.append(row)

        for tx, tx_scores in anomaly_scores_by_tx.items():
            tx_y = np.concatenate(
                [
                    np.zeros(len(normal_scores), dtype=np.int64),
                    np.ones(len(tx_scores), dtype=np.int64),
                ]
            )
            tx_all_scores = np.concatenate([normal_scores, tx_scores])
            tx_metrics = binary_metrics(
                tx_y,
                (tx_all_scores > threshold).astype(np.int64),
                tx_all_scores,
            )
            per_tx_rows.append(
                {
                    "representation": rep,
                    "method": method,
                    "tx": tx,
                    "normal_files": int(len(normal_scores)),
                    "anomaly_files": int(len(tx_scores)),
                    "threshold": threshold,
                    **tx_metrics,
                }
            )

    return rows, per_tx_rows


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    return float(xy / (xx * yy + 1e-12))


def pca_whiten(x: np.ndarray, max_components: int) -> np.ndarray:
    x = x - np.mean(x, axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(s > 1e-8))
    k = max(1, min(max_components, rank, x.shape[0] - 1, x.shape[1]))
    scores = x @ vt[:k].T
    return scores / (np.std(scores, axis=0, keepdims=True) + 1e-8)


def cca_summary(x: np.ndarray, y: np.ndarray, max_components: int) -> tuple[float, float, float, int, int]:
    xw = pca_whiten(x, max_components)
    yw = pca_whiten(y, max_components)
    cross = (xw.T @ yw) / max(xw.shape[0] - 1, 1)
    s = np.linalg.svd(cross, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    top1 = float(s[0]) if len(s) else float("nan")
    mean3 = float(np.mean(s[: min(3, len(s))])) if len(s) else float("nan")
    mean5 = float(np.mean(s[: min(5, len(s))])) if len(s) else float("nan")
    return top1, mean3, mean5, int(xw.shape[1]), int(yw.shape[1])


def pca_whiten_fit(x: np.ndarray, max_components: int) -> dict:
    mean = np.mean(x, axis=0, keepdims=True)
    centered = x - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(s > 1e-8))
    k = max(1, min(max_components, rank, x.shape[0] - 1, x.shape[1]))
    components = vt[:k]
    scores = centered @ components.T
    scale = np.std(scores, axis=0, keepdims=True) + 1e-8
    return {"mean": mean, "components": components, "scale": scale}


def pca_whiten_transform(x: np.ndarray, model: dict) -> np.ndarray:
    scores = (x - model["mean"]) @ model["components"].T
    return scores / model["scale"]


def corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = []
    for idx in range(min(x.shape[1], y.shape[1])):
        xi = x[:, idx]
        yi = y[:, idx]
        sx = np.std(xi)
        sy = np.std(yi)
        if sx < 1e-8 or sy < 1e-8:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(xi, yi)[0, 1]))
    return np.abs(np.asarray(values, dtype=np.float64))


def split_half_cca_summary(
    x: np.ndarray,
    y: np.ndarray,
    max_components: int,
    seed: int,
) -> tuple[float, float, float]:
    if x.shape[0] < 8:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    split = max(4, int(round(x.shape[0] * 0.7)))
    split = min(split, x.shape[0] - 4)
    fit_idx = indices[:split]
    eval_idx = indices[split:]

    x_fit = x[fit_idx]
    y_fit = y[fit_idx]
    x_eval = x[eval_idx]
    y_eval = y[eval_idx]

    mx, sx = standardize_fit(x_fit)
    my, sy = standardize_fit(y_fit)
    x_fit = standardize_apply(x_fit, mx, sx)
    y_fit = standardize_apply(y_fit, my, sy)
    x_eval = standardize_apply(x_eval, mx, sx)
    y_eval = standardize_apply(y_eval, my, sy)

    px = pca_whiten_fit(x_fit, max_components)
    py = pca_whiten_fit(y_fit, max_components)
    xw_fit = pca_whiten_transform(x_fit, px)
    yw_fit = pca_whiten_transform(y_fit, py)
    cross = (xw_fit.T @ yw_fit) / max(xw_fit.shape[0] - 1, 1)
    ux, _, vyt = np.linalg.svd(cross, full_matrices=False)

    xw_eval = pca_whiten_transform(x_eval, px)
    yw_eval = pca_whiten_transform(y_eval, py)
    x_can = xw_eval @ ux
    y_can = yw_eval @ vyt.T
    corrs = corr_columns(x_can, y_can)
    if len(corrs) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(corrs[0]),
        float(np.mean(corrs[: min(3, len(corrs))])),
        float(np.mean(corrs[: min(5, len(corrs))])),
    )


def relation_diagnostics(train_features: dict[str, np.ndarray], args) -> list[dict]:
    rows = []
    reps = list(train_features)
    for i, rep_a in enumerate(reps):
        for rep_b in reps[i + 1 :]:
            xa = train_features[rep_a]
            xb = train_features[rep_b]
            ma, sa = standardize_fit(xa)
            mb, sb = standardize_fit(xb)
            za = standardize_apply(xa, ma, sa)
            zb = standardize_apply(xb, mb, sb)
            top1, mean3, mean5, comp_a, comp_b = cca_summary(
                za,
                zb,
                args.pca_components,
            )
            split_top1, split_mean3, split_mean5 = split_half_cca_summary(
                xa,
                xb,
                args.pca_components,
                seed=args.seed + i * 100 + len(rows),
            )
            rows.append(
                {
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "linear_cka": linear_cka(za, zb),
                    "cca_top1": top1,
                    "cca_mean3": mean3,
                    "cca_mean5": mean5,
                    "split_cca_top1": split_top1,
                    "split_cca_mean3": split_mean3,
                    "split_cca_mean5": split_mean5,
                    "cca_components_a": comp_a,
                    "cca_components_b": comp_b,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_default_manifests(args) -> tuple[Path, Path, list[Path]]:
    manifest_dir = Path(args.manifest_dir).resolve()
    train = Path(args.train_manifest).resolve() if args.train_manifest else manifest_dir / "tx1_train_80_seed42_rebased.txt"
    normal = Path(args.normal_manifest).resolve() if args.normal_manifest else manifest_dir / "tx1_test_20_seed42_rebased.txt"
    if args.abnormal_manifests:
        abnormal = [Path(p).resolve() for p in args.abnormal_manifests]
    else:
        abnormal = sorted(manifest_dir.glob("tx*_all_rebased.txt"))
        abnormal = [p for p in abnormal if p.name.lower() != "tx1_all_rebased.txt"]
    return train, normal, abnormal


def main():
    args = parse_args()
    start = time.time()
    train_manifest, normal_manifest, abnormal_manifests = resolve_default_manifests(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    all_jobs = [("train", "Tx1", p) for p in train_files]
    all_jobs += [("normal", "Tx1", p) for p in normal_files]
    for tx, files in anomaly_files_by_tx.items():
        all_jobs += [("anomaly", tx, p) for p in files]

    print(
        f"[screening] files={len(all_jobs)} train={len(train_files)} "
        f"normal={len(normal_files)} anomaly={sum(len(v) for v in anomaly_files_by_tx.values())}"
    )
    print(f"[screening] representations={','.join(args.representations)}")

    matrices = {split: {rep: [] for rep in args.representations} for split in ["train", "normal"]}
    anomaly_matrices = {
        tx: {rep: [] for rep in args.representations} for tx in anomaly_files_by_tx
    }
    file_rows = []

    for job_idx, (split, tx, path) in enumerate(all_jobs, start=1):
        if job_idx == 1 or job_idx % 25 == 0 or job_idx == len(all_jobs):
            elapsed = time.time() - start
            print(f"[screening] {job_idx}/{len(all_jobs)} elapsed={elapsed:.1f}s {split}:{tx}:{path.name}")
        rep_features = file_features(path, args)
        for rep, values in rep_features.items():
            if split == "anomaly":
                anomaly_matrices[tx][rep].append(values)
            else:
                matrices[split][rep].append(values)
        file_rows.append(
            {
                "split": split,
                "tx": tx,
                "path": str(path),
                "basename": path.name,
            }
        )

    train_features = {rep: np.vstack(rows) for rep, rows in matrices["train"].items()}
    normal_features = {rep: np.vstack(rows) for rep, rows in matrices["normal"].items()}
    anomaly_features_by_tx = {
        tx: {rep: np.vstack(rows) for rep, rows in by_rep.items()}
        for tx, by_rep in anomaly_matrices.items()
    }

    metric_rows = []
    per_tx_rows = []
    for rep in args.representations:
        anomaly_by_tx_for_rep = {
            tx: by_rep[rep] for tx, by_rep in anomaly_features_by_tx.items()
        }
        rows, tx_rows = evaluate_representation(
            rep,
            train_features[rep],
            normal_features[rep],
            anomaly_by_tx_for_rep,
            args,
        )
        metric_rows.extend(rows)
        per_tx_rows.extend(tx_rows)

    relation_rows = relation_diagnostics(train_features, args)

    write_csv(output_dir / "representation_metrics.csv", metric_rows)
    write_csv(output_dir / "per_tx_metrics.csv", per_tx_rows)
    write_csv(output_dir / "relation_diagnostics.csv", relation_rows)
    write_csv(output_dir / "sampled_files.csv", file_rows)

    config = vars(args).copy()
    config.update(
        {
            "train_manifest": str(train_manifest),
            "normal_manifest": str(normal_manifest),
            "abnormal_manifests": [str(p) for p in abnormal_manifests],
            "output_dir": str(output_dir),
            "elapsed_seconds": time.time() - start,
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    best = sorted(metric_rows, key=lambda r: (r["auc"], r["f1"]), reverse=True)[:5]
    print("[screening] top metrics")
    for row in best:
        print(
            f"  {row['representation']:12s} {row['method']:12s} "
            f"auc={row['auc']:.6f} f1={row['f1']:.6f} "
            f"fp={row['fp']} fn={row['fn']} dim={row['feature_dim']}"
        )
    print(f"[screening] wrote: {output_dir}")


if __name__ == "__main__":
    main()
