# Implementation Plan

Updated on 2026-09-07 for the final Tx1-only one-class multi-view relation specification.

## Research Definition

Main question:

```text
In a Tx1-only one-class setting, does explicit IQ/AP/STFT multi-view relation
provide complementary anomaly information beyond learned absolute latent features?
```

The study is not framed as:

```text
Relation must outperform every deep feature baseline.
```

The central comparison is:

```text
Concat
vs
CCA Relation
vs
Concat + Relation
```

## Strict One-Class Protocol

Fitting data:

```text
Tx1 only
```

Evaluation data:

```text
Tx1 holdout: normal test
Tx2-Tx8: unseen anomaly test
Oracle SigMF: external anomaly test
```

The following must use only Tx1 fit/training data:

- encoder/autoencoder training
- StandardScaler fitting
- CCA fitting
- covariance fitting
- threshold calibration
- hyperparameter selection

Tx2-Tx8 and Oracle are final evaluation data only.

## Current Dataset Split

| Split | Role | Files |
| --- | --- | ---: |
| Tx1 fit/train | AE, CCA, covariance fitting | 320 |
| Tx1 calibration | threshold and score-fusion robust reference | 80 |
| Tx1 holdout | normal test | 100 |
| Tx2-Tx8 | unseen anomaly test | 500 each |
| Oracle SigMF | external anomaly test | 128 |

All splits are file-level. Windowing happens only after files are assigned to a split.

## Reused Files

| File | Reuse |
| --- | --- |
| `code/2nd/preprocessing.py` | MAT IQ loading and validation |
| `code/2nd/dataset.py` | window indexing policy reference |
| `code/2nd/model.py` | Conv1D/Conv2D encoder architecture reference |
| `code/2nd/evaluate_final_oracle_confusion.py` | SigMF dtype/count reference only |
| `code/2nd/evaluate.py` | metric and aggregation reference |
| `code/2nd/cross_view_relation/representation_screening.py` | CKA/CCA diagnostic reference |

## New File Structure

```text
rf_multiview_relation/
├── configs/
│   ├── default.yaml
│   └── ablation.yaml
├── data/
│   ├── dataset.py
│   ├── sigmf.py
│   ├── splits.py
│   ├── windowing.py
│   └── representations.py
├── models/
│   ├── encoder_iq.py
│   ├── encoder_ap.py
│   ├── encoder_stft.py
│   ├── decoders.py
│   └── autoencoder.py
├── relation/
│   ├── cca_alignment.py
│   ├── cka.py
│   ├── relation_features.py
│   └── pair_analysis.py
├── detectors/
│   ├── mahalanobis.py
│   ├── single_view.py
│   ├── concat.py
│   ├── relation.py
│   └── combined.py
├── scripts/
│   ├── 00_audit_dataset.py
│   ├── 01_verify_representations.py
│   ├── 02_train_autoencoders.py
│   ├── 03_extract_latents.py
│   ├── 04_analyze_relations.py
│   ├── 05_fit_detectors.py
│   ├── 06_evaluate.py
│   ├── 07_ablation.py
│   └── 08_run_all.py
├── utils/
│   ├── config.py
│   ├── io.py
│   ├── metrics.py
│   ├── plotting.py
│   └── seed.py
├── outputs/
├── tests/
├── requirements.txt
└── README.md
```

`outputs/` is intentionally ignored by Git.

## Module Input/Output

| Module | Input | Output |
| --- | --- | --- |
| `data/representations.py` | IQ window `[2048,2]` | IQ `[2,2048]`, AP `[2,2048]`, STFT `[1,128,31]` |
| `models/encoder_iq.py` | batch `[B,2,2048]` | `z_iq [B,64]` |
| `models/encoder_ap.py` | batch `[B,2,2048]` | `z_ap [B,64]` |
| `models/encoder_stft.py` | batch `[B,1,128,31]` | `z_stft [B,64]` |
| `relation/cca_alignment.py` | paired latents `[N,64]`, `[N,64]` | canonical projections `[N,16]`, correlations |
| `relation/relation_features.py` | three CCA-projected pairs | relation vector `[N,48]` for signed/absolute residual |
| `detectors/mahalanobis.py` | feature matrix `[N,D]` | window anomaly score `[N]` |
| `detectors/combined.py` | absolute score, relation score | fused score with fixed `alpha=0.5` |

## Tensor Shapes

```text
Raw loaded IQ:              [N, 2]
Windowed raw IQ:            [2048, 2]
IQ view:                    [B, 2, 2048]
AP view:                    [B, 2, 2048]
STFT view:                  [B, 1, 128, 31]
z_iq / z_ap / z_stft:       [B, 64]
Concat latent:              [B, 192]
CCA pair projection:        [B, 16]
CCA relation residual:      [B, 48]
Absolute + relation direct: [B, 240]
```

## Training Flow

1. Run dataset audit and file-level split.
2. Verify IQ/AP/STFT examples.
3. Train IQ, AP, and STFT autoencoders independently using Tx1 fit files only.
4. Save encoder and autoencoder checkpoints under `outputs/checkpoints/`.
5. Do not use Tx1 calibration, Tx1 holdout, Tx2-Tx8, or Oracle for AE optimization.

Checkpoint format:

```text
view
config
epoch
best_calibration_loss
model_state_dict
encoder_state_dict
stft_shape
```

Calibration loss is monitored using Tx1 calibration only. It is not an anomaly threshold result.

## Latent Extraction Flow

For every split/device:

```text
file_id
window_id
device
split
z_iq
z_ap
z_stft
```

Output:

```text
outputs/latents/tx1_train.npz
outputs/latents/tx1_calibration.npz
outputs/latents/tx1_holdout.npz
outputs/latents/tx2.npz
...
outputs/latents/tx8.npz
outputs/latents/oracle.npz
```

Latents are local generated artifacts and are not uploaded to GitHub.

## Relationship Analysis Flow

Using Tx1 fit latents only:

1. Fit CCA for IQ-AP, IQ-STFT, AP-STFT.
2. Save canonical correlations:

```text
outputs/tables/cca_results.csv
```

3. Compute dataset-level CKA per device:

```text
outputs/tables/cka_results.csv
```

First interpretation target:

```text
Is R_Tx1 stable?
```

## Detector Flow

### A. Single View

```text
z_iq   -> Mahalanobis
z_ap   -> Mahalanobis
z_stft -> Mahalanobis
```

### B. Absolute Multi-View Concat

```text
[z_iq; z_ap; z_stft] -> Mahalanobis
```

### C. Relation Only

```text
CCA signed residual R -> Mahalanobis
```

### D. Absolute + Relation

Direct version:

```text
[z_iq; z_ap; z_stft; R] -> Mahalanobis
```

Score-fusion version:

```text
S_combined = 0.5 * robust(S_abs) + 0.5 * robust(S_rel)
```

The fusion reference distribution is Tx1 calibration only. `alpha` is fixed at `0.5` unless a later Tx1-only sensitivity analysis is explicitly run.

## Evaluation Flow

Window score output:

```text
file_id
window_id
device
label
method
score
```

File score output:

```text
file_id
device
label
method
num_windows
file_score
threshold
prediction
```

Main file aggregation:

```text
p60
```

Main threshold:

```text
P95(Tx1 calibration file scores)
```

Required metrics:

```text
AUROC
AUPRC
accuracy
precision
recall
F1
FPR
TNR
```

## Required Result Tables

```text
outputs/tables/main_results.csv
outputs/tables/device_results.csv
outputs/tables/cca_results.csv
outputs/tables/cka_results.csv
outputs/tables/ablation_results.csv
```

## Required Figures

```text
outputs/figures/pipeline.png
outputs/figures/example_iq.png
outputs/figures/example_ap.png
outputs/figures/example_stft.png
outputs/figures/relation_score_distribution.png
outputs/figures/roc_comparison.png
outputs/figures/cca_heatmap.png
outputs/figures/cka_heatmap.png
outputs/figures/absolute_vs_relation_score.png
outputs/figures/oracle_score_distribution.png
```

## Implementation Order

1. Repository audit
2. Dataset audit
3. File-level split verification
4. IQ/AP/STFT representation verification
5. Autoencoder training
6. Latent extraction
7. CCA/CKA relationship analysis
8. Single-view baseline
9. Multi-view concat baseline
10. Raw relation baseline
11. CCA relation detector
12. Absolute + relation detector
13. Minimal Tx1-vs-Tx2 feasibility experiment
14. Full Tx2-Tx8 evaluation
15. Oracle external evaluation
16. Ablation
17. Tables / figures
18. Reproducible run-all script

## Commands

From repository root:

```bash
python rf_multiview_relation/scripts/00_audit_dataset.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/01_verify_representations.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/02_train_autoencoders.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/03_extract_latents.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/04_analyze_relations.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/05_fit_detectors.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/06_evaluate.py --config rf_multiview_relation/configs/default.yaml --scenario minimal_tx2
python rf_multiview_relation/scripts/07_ablation.py --config rf_multiview_relation/configs/default.yaml --ablation-config rf_multiview_relation/configs/ablation.yaml
python rf_multiview_relation/scripts/08_run_all.py --config rf_multiview_relation/configs/default.yaml
```

From inside `rf_multiview_relation/`, the final target command is:

```bash
python scripts/08_run_all.py --config configs/default.yaml
```

## Data Leakage Prevention

File-level assertions:

```python
assert train_files.isdisjoint(calibration_files)
assert train_files.isdisjoint(test_files)
assert calibration_files.isdisjoint(test_files)
```

Device-level fitting assertions:

```python
assert set(model_fit_devices) == {"Tx1"}
assert set(cca_fit_devices) == {"Tx1"}
assert set(threshold_fit_devices) == {"Tx1"}
```

Policy:

- no threshold changes after seeing Tx2-Tx8
- no CCA dimension changes based on anomaly performance
- no score normalization using Tx2-Tx8 or Oracle
- no Oracle use before final external evaluation

## Minimal Feasibility Experiment

Before full Tx2-Tx8:

```text
Normal: Tx1 holdout
Anomaly: Tx2
Methods: IQ-only, Concat, CCA Relation, Absolute + Relation
```

Outputs:

```text
AUROC
F1
Tx1/Tx2 score distribution
corr(S_abs, S_rel)
absolute_vs_relation_score.png
```

Decision checks:

1. Tx1 relation distribution is stable.
2. Tx2 relation score increases or shifts.
3. Relation score is not identical to absolute score.
4. `Concat + Relation > Concat` is enough to support the complementary-information story.

## Current Status

Completed:

- `CURRENT_CODE_AUDIT.md`
- `IMPLEMENTATION_PLAN.md`
- Phase 0 dataset audit
- Phase 1 representation verification
- Phase 2 AE training implementation and 1-file smoke run
- config re-established to the 2026-09-07 specification
- core `relation/` and `detectors/` module locations established

Next:

```text
Full Tx1-only AE training, then latent extraction.
```
