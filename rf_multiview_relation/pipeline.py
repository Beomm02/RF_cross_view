from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rf_multiview_relation.detectors.concat import concat_latents
from rf_multiview_relation.detectors.mahalanobis import MahalanobisDetector
from rf_multiview_relation.models.autoencoder import make_autoencoder
from rf_multiview_relation.relation.cca_alignment import CCAAlignment
from rf_multiview_relation.relation.relation_features import cosine_distance, euclidean_distance, residual_relation


PAIR_KEYS: tuple[tuple[str, str, str], ...] = (
    ("iq_ap", "iq", "ap"),
    ("iq_stft", "iq", "stft"),
    ("ap_stft", "ap", "stft"),
)

MAIN_METHODS: tuple[str, ...] = (
    "iq",
    "ap",
    "stft",
    "concat",
    "raw_relation",
    "cca_relation",
    "absolute_plus_relation",
    "score_fusion",
)


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def encoder_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["encoder"] if "encoder" in config else config["model"]


def stft_shape(config: dict[str, Any]) -> tuple[int, int]:
    data_cfg = config["data"]
    stft_cfg = config["representations"]["stft"]
    freq_bins = int(stft_cfg["n_fft"])
    frames = 1 + (int(data_cfg["window_size"]) - int(stft_cfg["win_length"])) // int(stft_cfg["hop_length"])
    return freq_bins, frames


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def load_encoders(
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    checkpoint_root = Path(checkpoint_dir)
    latent_dim = int(encoder_config(config)["latent_dim"])
    encoders: dict[str, torch.nn.Module] = {}
    for view in ("iq", "ap", "stft"):
        path = checkpoint_root / f"encoder_{view}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing encoder checkpoint: {path}")
        autoencoder = make_autoencoder(view, latent_dim=latent_dim, stft_shape=stft_shape(config)).to(device)
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        autoencoder.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        autoencoder.encoder.eval()
        encoders[view] = autoencoder.encoder
    return encoders


def load_latent_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return {
            "file_id": payload["file_id"].astype(str),
            "window_id": payload["window_id"].astype(np.int64),
            "window_start": payload["window_start"].astype(np.int64),
            "device": payload["device"].astype(str),
            "split": payload["split"].astype(str),
            "iq": payload["z_iq"].astype(np.float64),
            "ap": payload["z_ap"].astype(np.float64),
            "stft": payload["z_stft"].astype(np.float64),
        }


def load_latent_dir(latent_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    root = Path(latent_dir)
    result: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(root.glob("*.npz"), key=lambda p: p.name.lower()):
        result[path.stem] = load_latent_npz(path)
    if not result:
        raise FileNotFoundError(f"No latent npz files found under {root}")
    return result


def assert_tx1_only_fit(latents: dict[str, np.ndarray], split_name: str) -> None:
    devices = set(latents["device"].astype(str).tolist())
    if devices != {"Tx1"}:
        raise AssertionError(f"{split_name} fitting data must contain only Tx1, found {sorted(devices)}")


def assert_no_file_leakage(latent_sets: dict[str, dict[str, np.ndarray]]) -> None:
    names = [name for name in ("tx1_train", "tx1_calibration", "tx1_holdout") if name in latent_sets]
    file_sets = {name: set(latent_sets[name]["file_id"].astype(str).tolist()) for name in names}
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            overlap = file_sets[left] & file_sets[right]
            if overlap:
                raise AssertionError(f"File leakage between {left} and {right}: {sorted(overlap)[0]}")


def concat_feature_matrix(latents: dict[str, np.ndarray]) -> np.ndarray:
    return concat_latents(latents["iq"], latents["ap"], latents["stft"])


def fit_cca_models(train_latents: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, CCAAlignment]:
    assert_tx1_only_fit(train_latents, "tx1_train")
    cca_cfg = config["cca"]
    models: dict[str, CCAAlignment] = {}
    for pair_name, left, right in PAIR_KEYS:
        model = CCAAlignment(
            n_components=int(cca_cfg["components"]),
            standardize=bool(cca_cfg["standardize"]),
            regularization=float(cca_cfg["regularization"]),
        )
        model.fit(train_latents[left], train_latents[right])
        models[pair_name] = model
    return models


def cca_pair_projections(
    latents: dict[str, np.ndarray],
    cca_models: dict[str, CCAAlignment],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pairs = {}
    for pair_name, left, right in PAIR_KEYS:
        pairs[pair_name] = cca_models[pair_name].transform(latents[left], latents[right])
    return pairs


def cca_relation_feature_matrix(
    latents: dict[str, np.ndarray],
    cca_models: dict[str, CCAAlignment],
    mode: str = "signed_residual",
) -> np.ndarray:
    return residual_relation(cca_pair_projections(latents, cca_models), mode=mode)


def raw_relation_feature_matrix(latents: dict[str, np.ndarray]) -> np.ndarray:
    chunks = []
    for _, left, right in PAIR_KEYS:
        chunks.append(cosine_distance(latents[left], latents[right])[:, np.newaxis])
        chunks.append(euclidean_distance(latents[left], latents[right])[:, np.newaxis])
    return np.concatenate(chunks, axis=1).astype(np.float64)


def feature_matrix_for_method(
    method: str,
    latents: dict[str, np.ndarray],
    cca_models: dict[str, CCAAlignment] | None = None,
    relation_mode: str = "signed_residual",
) -> np.ndarray:
    if method in {"iq", "ap", "stft"}:
        return latents[method]
    if method == "concat":
        return concat_feature_matrix(latents)
    if method == "raw_relation":
        return raw_relation_feature_matrix(latents)
    if method == "cca_relation":
        if cca_models is None:
            raise ValueError("cca_models are required for cca_relation")
        return cca_relation_feature_matrix(latents, cca_models, mode=relation_mode)
    if method == "absolute_plus_relation":
        if cca_models is None:
            raise ValueError("cca_models are required for absolute_plus_relation")
        return np.concatenate(
            [
                concat_feature_matrix(latents),
                cca_relation_feature_matrix(latents, cca_models, mode=relation_mode),
            ],
            axis=1,
        )
    raise ValueError(f"Unsupported feature method: {method}")


def fit_mahalanobis(features: np.ndarray, config: dict[str, Any]) -> MahalanobisDetector:
    detector_cfg = config["detector"]
    return MahalanobisDetector(
        covariance=str(detector_cfg["covariance"]),
        eps=float(detector_cfg["mahalanobis_eps"]),
    ).fit(features)


def score_windows_for_method(
    method: str,
    latents: dict[str, np.ndarray],
    detectors: dict[str, MahalanobisDetector],
    cca_models: dict[str, CCAAlignment],
    relation_mode: str,
    fusion: Any | None = None,
) -> np.ndarray:
    if method == "score_fusion":
        if fusion is None:
            raise ValueError("fusion model is required for score_fusion")
        concat_scores = detectors["concat"].score(feature_matrix_for_method("concat", latents))
        relation_scores = detectors["cca_relation"].score(
            feature_matrix_for_method("cca_relation", latents, cca_models, relation_mode=relation_mode)
        )
        return fusion.score(concat_scores, relation_scores)
    return detectors[method].score(feature_matrix_for_method(method, latents, cca_models, relation_mode=relation_mode))


def aggregate_file_scores(
    file_ids: np.ndarray,
    devices: np.ndarray,
    splits: np.ndarray,
    window_scores: np.ndarray,
    method: str,
    threshold: float | None,
    percentile: float | str,
) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    ids = file_ids.astype(str)
    dev = devices.astype(str)
    split_values = splits.astype(str)
    scores = np.asarray(window_scores, dtype=np.float64)
    for idx, file_id in enumerate(ids):
        if file_id not in grouped:
            grouped[file_id] = {
                "method": method,
                "file_id": file_id,
                "device": dev[idx],
                "split": split_values[idx],
                "label": 0 if dev[idx] == "Tx1" else 1,
                "scores": [],
            }
        grouped[file_id]["scores"].append(float(scores[idx]))

    rows = []
    for item in grouped.values():
        values = np.asarray(item.pop("scores"), dtype=np.float64)
        file_score = aggregate_values(values, percentile)
        prediction = "" if threshold is None else int(file_score > float(threshold))
        rows.append(
            {
                **item,
                "num_windows": int(values.size),
                "file_score": float(file_score),
                "threshold": "" if threshold is None else float(threshold),
                "prediction": prediction,
            }
        )
    return rows


def aggregate_values(values: np.ndarray, percentile: float | str) -> float:
    data = np.asarray(values, dtype=np.float64)
    if isinstance(percentile, str):
        lowered = percentile.lower()
        if lowered == "mean":
            return float(np.mean(data))
        if lowered == "max":
            return float(np.max(data))
        if lowered.startswith("p"):
            return float(np.percentile(data, float(lowered[1:])))
        return float(np.percentile(data, float(lowered)))
    return float(np.percentile(data, float(percentile)))


def threshold_from_file_rows(rows: list[dict[str, Any]], percentile: float) -> float:
    scores = np.asarray([float(row["file_score"]) for row in rows], dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot compute threshold from empty calibration rows")
    return float(np.percentile(scores, float(percentile)))


def save_pickle(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
