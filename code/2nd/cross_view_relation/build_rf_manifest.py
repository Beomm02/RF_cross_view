import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebase RF .mat manifests to the current uploaded data root."
    )
    parser.add_argument("--data-root", default="../../data")
    parser.add_argument("--old-manifest-dir", default="exp_rigorous/manifests")
    parser.add_argument("--output-dir", default="cross_view_relation/manifests")
    parser.add_argument("--normal-tx", default="Tx1")
    parser.add_argument(
        "--anomaly-txs",
        nargs="+",
        default=["Tx2", "Tx3", "Tx4", "Tx5", "Tx6", "Tx7", "Tx8"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_manifest_basenames(path: Path) -> list[str]:
    basenames = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            basenames.append(Path(line).name)
    return basenames


def write_manifest(path: Path, files: list[Path], relative_to: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in files:
            rel = os.path.relpath(item, start=relative_to)
            f.write(rel + "\n")


def tx_file_map(data_root: Path, tx: str) -> dict[str, Path]:
    files = sorted((data_root / tx).glob("*.mat"))
    return {path.name: path.resolve() for path in files}


def rebase_by_basename(old_manifest: Path, data_root: Path, tx: str) -> tuple[list[Path], list[str]]:
    available = tx_file_map(data_root, tx)
    rebased = []
    missing = []
    for basename in read_manifest_basenames(old_manifest):
        path = available.get(basename)
        if path is None:
            missing.append(basename)
        else:
            rebased.append(path)
    return rebased, missing


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    old_manifest_dir = Path(args.old_manifest_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not old_manifest_dir.exists():
        raise FileNotFoundError(f"old manifest dir not found: {old_manifest_dir}")

    summary = {
        "data_root": str(data_root),
        "old_manifest_dir": str(old_manifest_dir),
        "output_dir": str(output_dir),
        "normal_tx": args.normal_tx,
        "anomaly_txs": args.anomaly_txs,
        "seed": args.seed,
        "manifests": {},
    }

    train_old = old_manifest_dir / "tx1_train_80_seed42.txt"
    test_old = old_manifest_dir / "tx1_test_20_seed42.txt"

    train_files, train_missing = rebase_by_basename(train_old, data_root, args.normal_tx)
    test_files, test_missing = rebase_by_basename(test_old, data_root, args.normal_tx)

    outputs = [
        ("tx1_train_80_seed42_rebased.txt", train_files, train_missing),
        ("tx1_test_20_seed42_rebased.txt", test_files, test_missing),
    ]

    for tx in args.anomaly_txs:
        files = sorted((data_root / tx).glob("*.mat"))
        outputs.append((f"{tx.lower()}_all_rebased.txt", [p.resolve() for p in files], []))

    for name, files, missing in outputs:
        out_path = output_dir / name
        write_manifest(out_path, files, relative_to=output_dir)
        summary["manifests"][name] = {
            "count": len(files),
            "path": str(out_path),
            "missing_count": len(missing),
            "missing_basenames": missing[:20],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "manifest_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
