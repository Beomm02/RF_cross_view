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
    assert_no_file_leakage,
    assert_tx1_only_fit,
    fit_cca_models,
    load_latent_dir,
    resolve_project_path,
)
from rf_multiview_relation.relation.cka import linear_cka  # noqa: E402
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402
from rf_multiview_relation.utils.plotting import save_heatmap_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 CCA/CKA multi-view relationship analysis.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-samples", type=int, default=50000)
    return parser.parse_args()


def maybe_with_run_name(path: Path, run_name: str) -> Path:
    return path / run_name if run_name else path


def sample_latents(latents: dict[str, np.ndarray], max_samples: int | None, seed: int) -> dict[str, np.ndarray]:
    if max_samples is None or latents["iq"].shape[0] <= int(max_samples):
        return latents
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(latents["iq"].shape[0], size=int(max_samples), replace=False))
    sampled = dict(latents)
    for key in ("file_id", "window_id", "window_start", "device", "split", "iq", "ap", "stft"):
        sampled[key] = latents[key][indices]
    return sampled


def cca_rows(cca_models: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair_name, _, _ in PAIR_KEYS:
        correlations = np.asarray(cca_models[pair_name].correlations_, dtype=np.float64)
        for idx, value in enumerate(correlations, start=1):
            rows.append(
                {
                    "pair": pair_name,
                    "component": idx,
                    "canonical_correlation": float(value),
                    "mean_canonical_correlation": float(np.mean(correlations)),
                }
            )
    return rows


def cka_rows(latent_sets: dict[str, dict[str, np.ndarray]], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    rows = []
    for dataset, latents in sorted(latent_sets.items()):
        sampled = sample_latents(latents, max_samples, seed)
        device_values = sorted(set(sampled["device"].astype(str).tolist()))
        device = device_values[0] if len(device_values) == 1 else ";".join(device_values)
        for pair_name, left, right in PAIR_KEYS:
            rows.append(
                {
                    "dataset": dataset,
                    "device": device,
                    "pair": pair_name,
                    "samples": int(sampled[left].shape[0]),
                    "cka": float(linear_cka(sampled[left], sampled[right])),
                }
            )
    return rows


def save_relation_heatmap(rows: list[dict[str, Any]], path: Path, value_key: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        matrix = np.eye(3, dtype=np.float64)
        pair_to_idx = {"iq_ap": (0, 1), "iq_stft": (0, 2), "ap_stft": (1, 2)}
        for row in rows:
            pair = str(row["pair"])
            if pair not in pair_to_idx:
                continue
            left, right = pair_to_idx[pair]
            value = float(row[value_key])
            matrix[left, right] = value
            matrix[right, left] = value
        save_heatmap_matrix(matrix, path)
        return
    views = ["iq", "ap", "stft"]
    labels = ["IQ", "AP", "STFT"]
    matrix = np.eye(3, dtype=np.float64)
    pair_to_idx = {"iq_ap": (0, 1), "iq_stft": (0, 2), "ap_stft": (1, 2)}
    for row in rows:
        pair = str(row["pair"])
        if pair not in pair_to_idx:
            continue
        left, right = pair_to_idx[pair]
        value = float(row[value_key])
        matrix[left, right] = value
        matrix[right, left] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_title(title)
    for row_idx in range(3):
        for col_idx in range(3):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths_cfg = config["paths"]
    output_dir = resolve_project_path(args.output_dir or paths_cfg["outputs"], PROJECT_ROOT)
    latent_dir = maybe_with_run_name(resolve_project_path(args.latent_dir or (output_dir / "latents"), PROJECT_ROOT), args.run_name)
    tables_dir = maybe_with_run_name(output_dir / "tables", args.run_name)
    figures_dir = maybe_with_run_name(output_dir / "figures", args.run_name)

    latent_sets = load_latent_dir(latent_dir)
    assert_no_file_leakage(latent_sets)
    if "tx1_train" not in latent_sets:
        raise FileNotFoundError("tx1_train.npz is required for CCA fitting")
    train_latents = sample_latents(latent_sets["tx1_train"], args.max_samples, int(config["seed"]))
    assert_tx1_only_fit(train_latents, "tx1_train")
    cca_models = fit_cca_models(train_latents, config)

    cca_table = cca_rows(cca_models)
    cka_table = cka_rows(latent_sets, args.max_samples, int(config["seed"]))
    write_csv(tables_dir / "cca_results.csv", cca_table)
    write_csv(tables_dir / "cka_results.csv", cka_table)

    mean_cca_rows = []
    seen = set()
    for row in cca_table:
        pair = row["pair"]
        if pair in seen:
            continue
        seen.add(pair)
        mean_cca_rows.append({"pair": pair, "mean_canonical_correlation": row["mean_canonical_correlation"]})
    tx1_cka_rows = [row for row in cka_table if row["dataset"] == "tx1_train"]
    save_relation_heatmap(mean_cca_rows, figures_dir / "cca_heatmap.png", "mean_canonical_correlation", "Tx1 CCA")
    save_relation_heatmap(tx1_cka_rows, figures_dir / "relation_heatmap.png", "cka", "Tx1 Linear CKA")

    print(f"[REL] CCA fitted on Tx1 train samples={train_latents['iq'].shape[0]}")
    for row in mean_cca_rows:
        print(f"[REL] pair={row['pair']} mean_canonical_corr={float(row['mean_canonical_correlation']):.4f}")
    print(f"[DONE] wrote {tables_dir / 'cca_results.csv'}")
    print(f"[DONE] wrote {tables_dir / 'cka_results.csv'}")


if __name__ == "__main__":
    main()
