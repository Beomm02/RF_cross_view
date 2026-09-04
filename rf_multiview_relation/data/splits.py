from __future__ import annotations

from pathlib import Path

import numpy as np


def canonical_path(path: str | Path) -> str:
    return str(Path(path).resolve()).lower()


def assert_disjoint(groups: dict[str, list[Path]]) -> None:
    names = list(groups)
    canonical_groups = {
        name: {canonical_path(path) for path in paths}
        for name, paths in groups.items()
    }
    for left_idx, left_name in enumerate(names):
        for right_name in names[left_idx + 1 :]:
            overlap = canonical_groups[left_name] & canonical_groups[right_name]
            if overlap:
                example = sorted(overlap)[0]
                raise AssertionError(f"File leakage between {left_name} and {right_name}: {example}")


def split_tx1_files(
    files: list[str | Path],
    seed: int,
    train_total: int,
    fit_count: int,
    calibration_count: int,
    holdout_count: int,
) -> dict[str, list[Path]]:
    paths = sorted([Path(path) for path in files], key=lambda p: p.name.lower())
    required = int(train_total) + int(holdout_count)
    if len(paths) < required:
        raise ValueError(f"Tx1 requires at least {required} files, found {len(paths)}")
    if int(train_total) != int(fit_count) + int(calibration_count):
        raise ValueError("tx1_train_files must equal tx1_fit_files + tx1_calibration_files")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(paths))
    selected = [paths[int(idx)] for idx in order[:required]]
    train_pool = selected[: int(train_total)]
    holdout = selected[int(train_total) : required]
    fit = train_pool[: int(fit_count)]
    calibration = train_pool[int(fit_count) : int(fit_count) + int(calibration_count)]

    groups = {
        "tx1_train": sorted(fit, key=lambda p: p.name.lower()),
        "tx1_calibration": sorted(calibration, key=lambda p: p.name.lower()),
        "tx1_holdout": sorted(holdout, key=lambda p: p.name.lower()),
        "tx1_train_all": sorted(train_pool, key=lambda p: p.name.lower()),
    }
    assert_disjoint(
        {
            "tx1_train": groups["tx1_train"],
            "tx1_calibration": groups["tx1_calibration"],
            "tx1_holdout": groups["tx1_holdout"],
        }
    )
    return groups


def write_split_manifest(paths: list[Path], output_path: str | Path, base_dir: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = Path(base_dir).resolve()
    lines = []
    for path in sorted(paths, key=lambda p: p.name.lower()):
        try:
            lines.append(path.resolve().relative_to(base).as_posix())
        except ValueError:
            lines.append(str(path.resolve()))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
