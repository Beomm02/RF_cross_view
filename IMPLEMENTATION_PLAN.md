# Implementation Plan

Updated on 2026-09-07 after cleanup to match the final specification.

## Research Objective

Implement and evaluate Tx1-only one-class RF transmitter anomaly detection using:

```text
learned absolute latent features
+
explicit IQ/AP/STFT multi-view relation features
```

Primary comparison:

```text
Concat
vs
CCA Relation
vs
Concat + Relation
```

## One-Class Boundary

Fitting allowed:

```text
Tx1 fit/train only
```

Calibration allowed:

```text
Tx1 calibration only
```

Evaluation only:

```text
Tx1 holdout
Tx2-Tx8
Oracle SigMF
```

## Current Package

```text
rf_multiview_relation/
├── configs/
│   ├── default.yaml
│   └── ablation.yaml
├── data/
├── models/
├── relation/
├── detectors/
├── scripts/
└── utils/
```

## Module Inputs And Outputs

| Module | Input | Output |
| --- | --- | --- |
| `data/mat.py` | `.mat` path, key `rxData` | raw IQ `[N,2]` |
| `data/sigmf.py` | Oracle `.sigmf-data` path | dtype/count/metadata |
| `data/windowing.py` | sample count, window config | window starts |
| `data/representations.py` | window `[2048,2]` | IQ `[2,2048]`, AP `[2,2048]`, STFT `[1,128,31]` |
| `models/*` | view tensors | latent `[B,64]` and reconstruction |
| `relation/cca_alignment.py` | paired Tx1 latents | CCA projections `[B,16]` |
| `relation/relation_features.py` | CCA pair projections | relation `[B,48]` |
| `detectors/mahalanobis.py` | feature `[B,D]` | anomaly scores `[B]` |
| `detectors/combined.py` | absolute and relation scores | fused score `[B]` |
| `pipeline.py` | latent dictionaries and fitted models | shared relation, scoring, leakage utilities |

## Tensor Shapes

```text
Raw IQ:                  [N, 2]
Window:                  [2048, 2]
IQ view:                 [B, 2, 2048]
AP view:                 [B, 2, 2048]
STFT view:               [B, 1, 128, 31]
z_iq / z_ap / z_stft:    [B, 64]
Concat latent:           [B, 192]
CCA projection:          [B, 16]
CCA relation residual:   [B, 48]
Absolute + relation:     [B, 240]
```

## Implemented Scripts

| Phase | Script | Role |
| --- | --- | --- |
| 0 | `scripts/00_audit_dataset.py` | audit Tx/Oracle files and create Tx1 file-level splits |
| 1 | `scripts/01_verify_representations.py` | generate IQ/AP/STFT examples and shape table |
| 2 | `scripts/02_train_autoencoders.py` | train IQ/AP/STFT autoencoders on Tx1 train only |
| 3 | `scripts/03_extract_latents.py` | extract paired window latents for Tx1, Tx2-Tx8, Oracle |
| 4 | `scripts/04_analyze_relations.py` | fit Tx1-only CCA and compute CKA analysis |
| 5 | `scripts/05_fit_relation_model.py` | fit baselines, relation models, fusion, and Tx1 calibration thresholds |
| 6 | `scripts/06_evaluate.py` | compute file-level scores, metrics, tests, and figures |
| 8 | `scripts/08_run_all.py` | orchestrate the full reproducible pipeline |

## Outputs

Latents:

```text
outputs/latents/tx1_train.npz
outputs/latents/tx1_calibration.npz
outputs/latents/tx1_holdout.npz
outputs/latents/tx2.npz
...
outputs/latents/tx8.npz
outputs/latents/oracle.npz
```

Each latent file contains:

```text
file_id
window_id
window_start
device
split
z_iq
z_ap
z_stft
```

Main tables:

```text
outputs/tables/cca_results.csv
outputs/tables/cka_results.csv
outputs/tables/detector_thresholds.csv
outputs/tables/main_results.csv
outputs/tables/device_results.csv
outputs/tables/statistical_tests.csv
outputs/tables/score_complementarity.csv
```

Main scores and figures:

```text
outputs/scores/file_scores.csv
outputs/figures/score_distribution.png
outputs/figures/roc_curve.png
outputs/figures/relation_heatmap.png
outputs/figures/oracle_distribution.png
outputs/figures/absolute_vs_relation_score.png
```

Window-score CSV export is available but optional because it can be very large:

```bash
python rf_multiview_relation/scripts/06_evaluate.py --config rf_multiview_relation/configs/default.yaml --write-window-scores
```

## Data Leakage Prevention

- `00_audit_dataset.py` creates Tx1 train/calibration/holdout manifests at file level.
- `pipeline.assert_no_file_leakage()` checks Tx1 train, calibration, and holdout file IDs after latent extraction.
- `pipeline.assert_tx1_only_fit()` is called before fitting CCA, covariance models, and calibration thresholds.
- Tx2-Tx8 and Oracle are loaded only by extraction/evaluation steps and are not used for fitting.
- All generated outputs, checkpoints, latents, pickles, raw data, and local logs are ignored by Git.

## Smoke Validation

Validated after cleanup with:

```bash
python rf_multiview_relation/scripts/08_run_all.py --config rf_multiview_relation/configs/default.yaml --run-name phase2_smoke_after_cleanup --minimal --skip-audit --skip-representation-check --skip-training --max-files-per-split 2 --device auto
```

This confirmed:

```text
Phase 3 latent extraction
Phase 4 CCA/CKA analysis
Phase 5 Tx1-only fitting and threshold calibration
Phase 6 minimal Tx1 holdout vs Tx2 evaluation
```

## Main Commands

Repository root:

```bash
python rf_multiview_relation/scripts/08_run_all.py --config rf_multiview_relation/configs/default.yaml
```

Inside `rf_multiview_relation/`:

```bash
python scripts/08_run_all.py --config configs/default.yaml
```

## Immediate Execution Plan

1. Commit and push the cleanup plus Phase 3-8 implementation; ignored outputs and local logs stay out of GitHub.
2. Start full Tx1-only AE training with seed 42.
3. Extract full latents for Tx1 train/calibration/holdout, Tx2-Tx8, and Oracle.
4. Run CCA/CKA relation analysis.
5. Fit Tx1-only relation/baseline detectors and Tx1 calibration thresholds.
6. Evaluate minimal Tx2 first, then full Tx2-Tx8 and Oracle outputs.
