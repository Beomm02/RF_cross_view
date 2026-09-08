from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "rf_multiview_relation" / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main RF multi-view relation pipeline.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-calibration-files", type=int, default=None)
    parser.add_argument("--max-files-per-split", type=int, default=None)
    parser.add_argument("--minimal", action="store_true", help="Run only Tx1 train/calibration/holdout and Tx2.")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-representation-check", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-relation-screening", action="store_true")
    parser.add_argument("--screen-top-k", type=int, default=5)
    parser.add_argument("--write-window-scores", action="store_true")
    return parser.parse_args()


def append_common(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd.extend(["--config", args.config])
    if args.data_root is not None and Path(cmd[1]).name in {
        "00_audit_dataset.py",
        "01_verify_representations.py",
        "02_train_autoencoders.py",
        "03_extract_latents.py",
    }:
        cmd.extend(["--data-root", args.data_root])
    if args.output_dir is not None:
        cmd.extend(["--output-dir", args.output_dir])
    if args.run_name and Path(cmd[1]).name not in {"00_audit_dataset.py", "01_verify_representations.py"}:
        cmd.extend(["--run-name", args.run_name])
    return cmd


def run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    split_args = ["tx1_train", "tx1_calibration", "tx1_holdout", "tx2"] if args.minimal else []

    if not args.skip_audit:
        run(append_common([sys.executable, str(SCRIPT_ROOT / "00_audit_dataset.py")], args))
    if not args.skip_representation_check:
        run(append_common([sys.executable, str(SCRIPT_ROOT / "01_verify_representations.py")], args))
    if not args.skip_training:
        cmd = append_common([sys.executable, str(SCRIPT_ROOT / "02_train_autoencoders.py")], args)
        cmd.extend(["--device", args.device])
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.max_train_files is not None:
            cmd.extend(["--max-train-files", str(args.max_train_files)])
        if args.max_calibration_files is not None:
            cmd.extend(["--max-calibration-files", str(args.max_calibration_files)])
        run(cmd)

    cmd = append_common([sys.executable, str(SCRIPT_ROOT / "03_extract_latents.py")], args)
    cmd.extend(["--device", args.device])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.max_files_per_split is not None:
        cmd.extend(["--max-files-per-split", str(args.max_files_per_split)])
    if split_args:
        cmd.extend(["--splits", *split_args])
    run(cmd)

    cmd = append_common([sys.executable, str(SCRIPT_ROOT / "04_analyze_relations.py")], args)
    run(cmd)

    if not args.skip_relation_screening:
        cmd = append_common([sys.executable, str(SCRIPT_ROOT / "07_screen_relations.py")], args)
        cmd.extend(["--top-k", str(args.screen_top_k)])
        run(cmd)

    cmd = append_common([sys.executable, str(SCRIPT_ROOT / "05_fit_relation_model.py")], args)
    run(cmd)

    cmd = append_common([sys.executable, str(SCRIPT_ROOT / "06_evaluate.py")], args)
    if split_args:
        cmd.extend(["--eval-splits", "tx1_holdout", "tx2"])
    if args.write_window_scores:
        cmd.append("--write-window-scores")
    run(cmd)


if __name__ == "__main__":
    main()
