from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rf_multiview_relation.utils.config import load_config  # noqa: E402
from rf_multiview_relation.utils.io import write_csv  # noqa: E402


SCRIPT_ROOT = PROJECT_ROOT / "rf_multiview_relation" / "scripts"
PHASE_VARIANTS: dict[str, dict[str, Any]] = {
    "phase_unwrap_raw": {"phase_unwrap": True, "phase_transform": "raw"},
    "phase_unwrap_raw_pilot": {"phase_unwrap": True, "phase_transform": "raw"},
    "phase_unwrap_center_unit": {"phase_unwrap": True, "phase_transform": "center_unit"},
    "phase_unwrap_detrend_unit": {"phase_unwrap": True, "phase_transform": "detrend_unit"},
    "phase_unwrap_trend_unit": {"phase_unwrap": True, "phase_transform": "trend_unit"},
    "phase_unwrap_slope_unit": {"phase_unwrap": True, "phase_transform": "slope_unit"},
    "phase_wrapped_unit": {"phase_unwrap": False, "phase_transform": "unit"},
    "phase_unwrap_zscore": {"phase_unwrap": True, "phase_transform": "zscore"},
    "phase_unwrap_diff_unit": {"phase_unwrap": True, "phase_transform": "diff_unit"},
}
DEFAULT_VARIANTS = [
    "phase_unwrap_raw",
    "phase_unwrap_center_unit",
    "phase_unwrap_detrend_unit",
    "phase_unwrap_trend_unit",
    "phase_unwrap_slope_unit",
    "phase_wrapped_unit",
    "phase_unwrap_zscore",
    "phase_unwrap_diff_unit",
]
EVAL_METHODS = [
    "ap",
    "concat",
    "cca_relation",
    "cca_compact_relation",
    "ap_stft_cca_cosine_direct",
    "ap_stft_cca_l2_direct",
    "ap_stft_cca_abs_mean_direct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AP phase preprocessing ablation with Tx1-only fitting.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "rf_multiview_relation" / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-calibration-files", type=int, default=None)
    parser.add_argument("--max-files-per-split", type=int, default=None)
    parser.add_argument("--screen-top-k", type=int, default=5)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-latents", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
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


def run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def variant_config(base_config: dict[str, Any], variant_name: str) -> dict[str, Any]:
    if variant_name not in PHASE_VARIANTS:
        raise ValueError(f"Unknown phase variant={variant_name}. Choices: {sorted(PHASE_VARIANTS)}")
    config = deepcopy(base_config)
    config.setdefault("representations", {})
    config["representations"].update(PHASE_VARIANTS[variant_name])
    config.setdefault("experiment", {})
    config["experiment"]["phase_ablation_variant"] = variant_name
    return config


def prepare_checkpoint_dir(output_dir: Path, run_name: str) -> None:
    base_dir = output_dir / "checkpoints"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for view in ("iq", "stft"):
        for prefix in ("encoder", "autoencoder"):
            source = base_dir / f"{prefix}_{view}.pt"
            target = run_dir / f"{prefix}_{view}.pt"
            if not source.exists():
                raise FileNotFoundError(f"Missing baseline checkpoint needed for reuse: {source}")
            shutil.copy2(source, target)
            print(f"[COPY] {source} -> {target}")


def extend_optional(cmd: list[str], args: argparse.Namespace, include_batch: bool = True) -> list[str]:
    if args.output_dir is not None:
        cmd.extend(["--output-dir", args.output_dir])
    if include_batch and args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    return cmd


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def min_ap_loss(tables_dir: Path) -> float:
    rows = [row for row in read_csv(tables_dir / "ae_training_curves.csv") if row.get("view") == "ap"]
    if not rows:
        return float("nan")
    return min(float(row["calibration_loss"]) for row in rows)


def ap_stft_cka(tables_dir: Path) -> float:
    rows = read_csv(tables_dir / "cka_results.csv")
    for row in rows:
        if row.get("dataset") == "tx1_train" and row.get("pair") == "ap_stft":
            return float(row["cka"])
    return float("nan")


def ap_stft_cca(tables_dir: Path) -> float:
    rows = read_csv(tables_dir / "cca_results.csv")
    for row in rows:
        if row.get("pair") == "ap_stft":
            return float(row["mean_canonical_correlation"])
    return float("nan")


def metric_lookup(tables_dir: Path, filename: str, method_key: str, evaluation: str) -> tuple[float, float]:
    rows = read_csv(tables_dir / filename)
    for row in rows:
        key = row.get("method") or row.get("candidate")
        if key == method_key and row.get("evaluation") == evaluation:
            auroc = row.get("AUROC", row.get("auroc", "nan"))
            f1 = row.get("F1", row.get("f1", "nan"))
            return float(auroc), float(f1)
    return float("nan"), float("nan")


def top_screen_candidate(tables_dir: Path) -> str:
    rows = read_csv(tables_dir / "tx1_relation_screening_selected.csv")
    if not rows:
        return ""
    return str(rows[0].get("candidate", ""))


def collect_summary(output_dir: Path, variant_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_tables = output_dir / "tables"
    if base_tables.exists():
        closed_auroc, closed_f1 = metric_lookup(base_tables, "tx1_screened_relation_results.csv", "ap_stft_cca_cosine", "closed_tx2_tx8")
        oracle_auroc, oracle_f1 = metric_lookup(base_tables, "tx1_screened_relation_results.csv", "ap_stft_cca_cosine", "external_oracle")
        rows.append(
            {
                "variant": "phase_unwrap_raw_baseline",
                "phase_unwrap": True,
                "phase_transform": "raw",
                "ap_best_calibration_loss": min_ap_loss(base_tables),
                "tx1_ap_stft_cka": ap_stft_cka(base_tables),
                "tx1_ap_stft_cca_mean_corr": ap_stft_cca(base_tables),
                "top_tx1_screen_candidate": top_screen_candidate(base_tables),
                "ap_stft_cca_cosine_closed_auroc": closed_auroc,
                "ap_stft_cca_cosine_closed_f1": closed_f1,
                "ap_stft_cca_cosine_oracle_auroc": oracle_auroc,
                "ap_stft_cca_cosine_oracle_f1": oracle_f1,
            }
        )
    for variant_name in variant_names:
        tables_dir = output_dir / "tables" / variant_name
        variant = PHASE_VARIANTS[variant_name]
        closed_auroc, closed_f1 = metric_lookup(tables_dir, "tx1_screened_relation_results.csv", "ap_stft_cca_cosine", "closed_tx2_tx8")
        oracle_auroc, oracle_f1 = metric_lookup(tables_dir, "tx1_screened_relation_results.csv", "ap_stft_cca_cosine", "external_oracle")
        rows.append(
            {
                "variant": variant_name,
                "phase_unwrap": variant["phase_unwrap"],
                "phase_transform": variant["phase_transform"],
                "ap_best_calibration_loss": min_ap_loss(tables_dir),
                "tx1_ap_stft_cka": ap_stft_cka(tables_dir),
                "tx1_ap_stft_cca_mean_corr": ap_stft_cca(tables_dir),
                "top_tx1_screen_candidate": top_screen_candidate(tables_dir),
                "ap_stft_cca_cosine_closed_auroc": closed_auroc,
                "ap_stft_cca_cosine_closed_f1": closed_f1,
                "ap_stft_cca_cosine_oracle_auroc": oracle_auroc,
                "ap_stft_cca_cosine_oracle_f1": oracle_f1,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    output_dir = resolve_project_path(args.output_dir or base_config["paths"]["outputs"])
    config_dir = output_dir / "phase_ablation_configs"
    variants = [str(item) for item in args.variants]

    for variant_name in variants:
        run_name = variant_name
        config = variant_config(base_config, variant_name)
        config_path = config_dir / f"{variant_name}.yaml"
        dump_yaml(config_path, config)
        prepare_checkpoint_dir(output_dir, run_name)

        if not args.skip_training:
            cmd = [
                sys.executable,
                str(SCRIPT_ROOT / "02_train_autoencoders.py"),
                "--config",
                str(config_path),
                "--views",
                "ap",
                "--run-name",
                run_name,
                "--device",
                args.device,
            ]
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.max_train_files is not None:
                cmd.extend(["--max-train-files", str(args.max_train_files)])
            if args.max_calibration_files is not None:
                cmd.extend(["--max-calibration-files", str(args.max_calibration_files)])
            run(extend_optional(cmd, args, include_batch=False))

        if not args.skip_latents:
            cmd = [
                sys.executable,
                str(SCRIPT_ROOT / "03_extract_latents.py"),
                "--config",
                str(config_path),
                "--run-name",
                run_name,
                "--device",
                args.device,
            ]
            if args.max_files_per_split is not None:
                cmd.extend(["--max-files-per-split", str(args.max_files_per_split)])
            run(extend_optional(cmd, args))

        if not args.skip_evaluation:
            run(
                extend_optional(
                    [sys.executable, str(SCRIPT_ROOT / "04_analyze_relations.py"), "--config", str(config_path), "--run-name", run_name],
                    args,
                    include_batch=False,
                )
            )
            cmd = [
                sys.executable,
                str(SCRIPT_ROOT / "07_screen_relations.py"),
                "--config",
                str(config_path),
                "--run-name",
                run_name,
                "--top-k",
                str(args.screen_top_k),
            ]
            run(extend_optional(cmd, args, include_batch=False))
            cmd = [
                sys.executable,
                str(SCRIPT_ROOT / "05_fit_relation_model.py"),
                "--config",
                str(config_path),
                "--run-name",
                run_name,
                "--methods",
                *EVAL_METHODS,
            ]
            run(extend_optional(cmd, args, include_batch=False))
            cmd = [
                sys.executable,
                str(SCRIPT_ROOT / "06_evaluate.py"),
                "--config",
                str(config_path),
                "--run-name",
                run_name,
                "--methods",
                *EVAL_METHODS,
                "--stat-method",
                "ap_stft_cca_cosine_direct",
            ]
            run(extend_optional(cmd, args, include_batch=False))

    summary_rows = collect_summary(output_dir, variants)
    summary_path = output_dir / "tables" / "phase_ablation_summary.csv"
    write_csv(summary_path, summary_rows)
    print(f"[DONE] wrote {summary_path}")
    for row in summary_rows:
        print(
            f"[SUMMARY] {row['variant']} ap_loss={float(row['ap_best_calibration_loss']):.6f} "
            f"cka={float(row['tx1_ap_stft_cka']):.4f} cca={float(row['tx1_ap_stft_cca_mean_corr']):.4f} "
            f"closed_auroc={float(row['ap_stft_cca_cosine_closed_auroc']):.4f} "
            f"oracle_auroc={float(row['ap_stft_cca_cosine_oracle_auroc']):.4f}"
        )


if __name__ == "__main__":
    main()
