from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402


SCRIPT_ROOT = PROJECT_ROOT / "rf_multiview_relation" / "scripts"
SUMMARY_METHODS = [
    "concat",
    "cca_relation",
    "cca_compact_relation",
    "ap_stft_cca_cosine_direct",
    "ap_stft_cca_l2_direct",
    "ap_stft_cca_abs_mean_direct",
]
SUMMARY_EVALUATIONS = ["closed_tx2_tx8", "external_oracle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seed robustness experiments for the main RF pipeline.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--run-prefix", default="seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-calibration-files", type=int, default=None)
    parser.add_argument("--max-files-per-split", type=int, default=None)
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-representation-check", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-relation-screening", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild seed robustness summary tables from existing runs.")
    parser.add_argument("--screen-top-k", type=int, default=5)
    return parser.parse_args()


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(simple_yaml(payload), encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def simple_yaml(payload: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(simple_yaml(value, indent + 2).rstrip())
        else:
            lines.append(f"{prefix}{key}: {simple_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def simple_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(simple_yaml_scalar(item) for item in value) + "]"
    if isinstance(value, str):
        if value == "" or any(char in value for char in ":#[]{}&,"):
            return '"' + value.replace('"', '\\"') + '"'
        return value
    return str(value)


def default_seeds(config: dict[str, Any]) -> list[int]:
    report_seeds = config.get("experiment", {}).get("report_seeds", [])
    current_seed = int(config["seed"])
    seeds = [int(seed) for seed in report_seeds if int(seed) != current_seed]
    return seeds or [123, 2026]


def seed_config(base_config: dict[str, Any], seed: int) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["seed"] = int(seed)
    config.setdefault("experiment", {})
    config["experiment"]["active_seed"] = int(seed)
    return config


def run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_name_for_seed(seed: int, run_prefix: str) -> str:
    return f"{run_prefix}_{int(seed)}"


def run_seed(seed: int, config_path: Path, args: argparse.Namespace) -> None:
    run_name = run_name_for_seed(seed, str(args.run_prefix))
    cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "08_run_all.py"),
        "--config",
        str(config_path),
        "--run-name",
        run_name,
        "--device",
        args.device,
    ]
    if args.output_dir is not None:
        cmd.extend(["--output-dir", args.output_dir])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.max_train_files is not None:
        cmd.extend(["--max-train-files", str(args.max_train_files)])
    if args.max_calibration_files is not None:
        cmd.extend(["--max-calibration-files", str(args.max_calibration_files)])
    if args.max_files_per_split is not None:
        cmd.extend(["--max-files-per-split", str(args.max_files_per_split)])
    if args.minimal:
        cmd.append("--minimal")
    if args.skip_audit:
        cmd.append("--skip-audit")
    if args.skip_representation_check:
        cmd.append("--skip-representation-check")
    if args.skip_training:
        cmd.append("--skip-training")
    if args.skip_relation_screening:
        cmd.append("--skip-relation-screening")
    cmd.extend(["--screen-top-k", str(args.screen_top_k)])
    run(cmd)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_float(rows: list[dict[str, str]], predicate: dict[str, str], field: str) -> float:
    for row in rows:
        if all(row.get(key) == value for key, value in predicate.items()):
            return float(row[field])
    return float("nan")


def top_screen_candidate(tables_dir: Path) -> str:
    rows = read_csv(tables_dir / "tx1_relation_screening_selected.csv")
    return str(rows[0].get("candidate", "")) if rows else ""


def collect_seed_rows(output_dir: Path, seeds: list[int], run_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_name = run_name_for_seed(seed, run_prefix)
        tables_dir = output_dir / "tables" / run_name
        main_rows = read_csv(tables_dir / "main_results.csv")
        if not main_rows:
            continue
        cca_rows = read_csv(tables_dir / "cca_results.csv")
        cka_rows = read_csv(tables_dir / "cka_results.csv")
        ap_stft_cca = first_float(cca_rows, {"pair": "ap_stft"}, "mean_canonical_correlation")
        tx1_ap_stft_cka = first_float(cka_rows, {"dataset": "tx1_train", "pair": "ap_stft"}, "cka")
        selected = top_screen_candidate(tables_dir)
        for row in main_rows:
            if row.get("method") not in SUMMARY_METHODS or row.get("evaluation") not in SUMMARY_EVALUATIONS:
                continue
            rows.append(
                {
                    "seed": int(seed),
                    "run_name": run_name,
                    "method": row["method"],
                    "method_label": row.get("method_label", row["method"]),
                    "evaluation": row["evaluation"],
                    "top_tx1_screen_candidate": selected,
                    "tx1_ap_stft_cka": tx1_ap_stft_cka,
                    "tx1_ap_stft_cca_mean_corr": ap_stft_cca,
                    "normal_files": row.get("normal_files", ""),
                    "anomaly_files": row.get("anomaly_files", ""),
                    "AUROC": float(row["AUROC"]),
                    "AUPRC": float(row["AUPRC"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "F1": float(row["F1"]),
                    "FPR": float(row["FPR"]),
                    "TNR": float(row["TNR"]),
                }
            )
    return rows


def collect_aggregate_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = (str(row["method"]), str(row["evaluation"]))
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    metric_keys = ["AUROC", "AUPRC", "precision", "recall", "F1", "FPR", "TNR"]
    for (method, evaluation), rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "method": method,
            "method_label": rows[0]["method_label"],
            "evaluation": evaluation,
            "seeds": ";".join(str(row["seed"]) for row in rows),
            "seed_count": len(rows),
        }
        for key in metric_keys:
            values = [float(row[key]) for row in rows]
            out[f"{key}_mean"] = mean(values)
            out[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(out)
    return aggregate_rows


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    output_dir = resolve_project_path(args.output_dir or base_config["paths"]["outputs"])
    seeds = [int(seed) for seed in (args.seeds or default_seeds(base_config))]
    config_dir = output_dir / "seed_configs"

    if not args.summary_only:
        for seed in seeds:
            config = seed_config(base_config, seed)
            config_path = config_dir / f"seed_{seed}.yaml"
            dump_yaml(config_path, config)
            run_seed(seed, config_path, args)

    seed_rows = collect_seed_rows(output_dir, seeds, str(args.run_prefix))
    aggregate_rows = collect_aggregate_rows(seed_rows)
    tables_dir = output_dir / "tables"
    write_csv(tables_dir / "seed_robustness_results.csv", seed_rows)
    write_csv(tables_dir / "seed_robustness_aggregate.csv", aggregate_rows)
    print(f"[DONE] wrote {tables_dir / 'seed_robustness_results.csv'}")
    print(f"[DONE] wrote {tables_dir / 'seed_robustness_aggregate.csv'}")
    for row in aggregate_rows:
        if row["method"] == "ap_stft_cca_cosine_direct":
            print(
                "[SEED] "
                f"{row['evaluation']} AP-STFT CCA cosine "
                f"AUROC={float(row['AUROC_mean']):.4f}+/-{float(row['AUROC_std']):.4f} "
                f"F1={float(row['F1_mean']):.4f}+/-{float(row['F1_std']):.4f} "
                f"seeds={row['seeds']}"
            )


if __name__ == "__main__":
    main()
