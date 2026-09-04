from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_multiview_relation.data.dataset import iter_view_batches, read_manifest, split_manifest_path  # noqa: E402
from rf_multiview_relation.models.autoencoder import make_autoencoder  # noqa: E402
from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402
from rf_multiview_relation.utils.plotting import save_loss_curves  # noqa: E402
from rf_multiview_relation.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Tx1-only autoencoder pretraining.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--views", nargs="+", default=["iq", "ap", "stft"], choices=["iq", "ap", "stft"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-calibration-files", type=int, default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def stft_shape(config: dict) -> tuple[int, int]:
    data_cfg = config["data"]
    stft_cfg = config["representations"]["stft"]
    freq_bins = int(stft_cfg["n_fft"])
    frames = 1 + (int(data_cfg["window_size"]) - int(stft_cfg["win_length"])) // int(stft_cfg["hop_length"])
    return freq_bins, frames


def output_subdir(base: Path, run_name: str) -> Path:
    if not run_name:
        return base
    return base / run_name


def evaluate_loss(
    model: torch.nn.Module,
    files: list[Path],
    view_name: str,
    config: dict,
    batch_size: int,
    device: torch.device,
    max_files: int | None,
    seed: int,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    rng = np.random.default_rng(int(seed))
    with torch.no_grad():
        for batch_np in iter_view_batches(
            files,
            view_name,
            config,
            batch_size=batch_size,
            rng=rng,
            shuffle_files=False,
            shuffle_windows=False,
            max_files=max_files,
        ):
            batch = torch.from_numpy(batch_np).to(device)
            reconstruction, _ = model(batch)
            loss = F.mse_loss(reconstruction, batch, reduction="sum")
            total_loss += float(loss.item())
            total_count += int(batch.numel())
    return total_loss / max(total_count, 1), total_count


def train_one_view(
    view_name: str,
    config: dict,
    train_files: list[Path],
    calibration_files: list[Path],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    max_train_files: int | None,
    max_calibration_files: int | None,
) -> list[dict[str, Any]]:
    latent_dim = int(config["model"]["latent_dim"])
    model = make_autoencoder(view_name, latent_dim=latent_dim, stft_shape=stft_shape(config)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    early_stopping = int(config["model"].get("early_stopping", 5))
    best_loss = float("inf")
    stale_epochs = 0
    rows = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        rng = np.random.default_rng(int(config["seed"]) + epoch)
        for batch_np in iter_view_batches(
            train_files,
            view_name,
            config,
            batch_size=batch_size,
            rng=rng,
            shuffle_files=True,
            shuffle_windows=True,
            max_files=max_train_files,
        ):
            batch = torch.from_numpy(batch_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = F.mse_loss(reconstruction, batch, reduction="mean")
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(batch.numel())
            train_count += int(batch.numel())

        train_loss = train_loss_sum / max(train_count, 1)
        calibration_loss, calibration_count = evaluate_loss(
            model,
            calibration_files,
            view_name,
            config,
            batch_size=batch_size,
            device=device,
            max_files=max_calibration_files,
            seed=int(config["seed"]),
        )
        improved = calibration_loss < best_loss - 1e-8
        if improved:
            best_loss = calibration_loss
            stale_epochs = 0
            checkpoint = {
                "view": view_name,
                "config": config,
                "epoch": epoch,
                "best_calibration_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "stft_shape": stft_shape(config),
            }
            torch.save(checkpoint, checkpoint_dir / f"autoencoder_{view_name}.pt")
            torch.save(
                {
                    "view": view_name,
                    "config": config,
                    "epoch": epoch,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "stft_shape": stft_shape(config),
                },
                checkpoint_dir / f"encoder_{view_name}.pt",
            )
        else:
            stale_epochs += 1

        row = {
            "view": view_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "calibration_loss": calibration_loss,
            "best_calibration_loss": best_loss,
            "train_elements": train_count,
            "calibration_elements": calibration_count,
            "improved": improved,
        }
        rows.append(row)
        print(
            f"[AE] view={view_name} epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.8f} cal_loss={calibration_loss:.8f} best={best_loss:.8f}"
        )
        if stale_epochs >= early_stopping:
            print(f"[AE] view={view_name} early_stop epoch={epoch}")
            break
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    paths_cfg = config["paths"]
    data_root = resolve_project_path(args.data_root or paths_cfg["data_root"])
    output_dir = resolve_project_path(args.output_dir or paths_cfg["outputs"])
    checkpoint_dir = output_subdir(output_dir / "checkpoints", args.run_name)
    tables_dir = output_subdir(output_dir / "tables", args.run_name)
    figures_dir = output_subdir(output_dir / "figures", args.run_name)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = split_manifest_path(output_dir, "tx1_train", int(config["seed"]))
    calibration_manifest = split_manifest_path(output_dir, "tx1_calibration", int(config["seed"]))
    train_files = read_manifest(train_manifest, data_root)
    calibration_files = read_manifest(calibration_manifest, data_root)

    epochs = int(args.epochs or config["model"]["epochs"])
    batch_size = int(args.batch_size or config["model"]["batch_size"])
    device = choose_device(args.device)
    print(
        f"[AE] device={device} views={','.join(args.views)} epochs={epochs} batch_size={batch_size} "
        f"train_files={len(train_files)} calibration_files={len(calibration_files)}"
    )
    if args.max_train_files is not None or args.max_calibration_files is not None:
        print(f"[AE] smoke/limited run: max_train_files={args.max_train_files} max_calibration_files={args.max_calibration_files}")

    all_rows = []
    for view in args.views:
        all_rows.extend(
            train_one_view(
                view,
                config,
                train_files,
                calibration_files,
                device,
                checkpoint_dir,
                epochs,
                batch_size,
                args.max_train_files,
                args.max_calibration_files,
            )
        )

    curve_csv = tables_dir / "ae_training_curves.csv"
    curve_png = figures_dir / "ae_training_curves.png"
    write_csv(curve_csv, all_rows)
    save_loss_curves(all_rows, curve_png)
    print(f"[DONE] wrote {curve_csv}")
    print(f"[DONE] wrote {curve_png}")
    print(f"[DONE] checkpoints under {checkpoint_dir}")


if __name__ == "__main__":
    main()
