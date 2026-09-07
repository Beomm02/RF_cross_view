from __future__ import annotations

import numpy as np


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    for value in np.unique(s):
        mask = s == value
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    return (pos_rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def average_precision_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-s)
    sorted_y = y[order]
    tp = np.cumsum(sorted_y == 1)
    precision = tp / np.arange(1, y.size + 1)
    return float(np.sum(precision[sorted_y == 1]) / positives)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int64)
    pred = (np.asarray(scores, dtype=np.float64) > float(threshold)).astype(np.int64)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "AUROC": roc_auc_score(y, scores),
        "AUPRC": average_precision_score(y, scores),
        "accuracy": (tp + tn) / max(y.size, 1),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TNR": tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
