# Implementation Plan

## Research Protocol

The main study is Tx1-only one-class RF anomaly detection.

Fitting data:

```text
Tx1 train only
```

Evaluation data:

```text
Tx1 holdout as normal test
Tx2-Tx8 as unseen/anomaly test
Oracle SigMF as external anomaly test
```

Tx2-Tx8 and Oracle must not be used for encoder training, CCA fitting, covariance fitting, threshold tuning, hyperparameter tuning, or score normalization.

## Reused Files

| File | Reuse |
| --- | --- |
| `code/2nd/preprocessing.py` | MAT IQ loading, IQ validation, power normalization reference |
| `code/2nd/dataset.py` | file-level window index policy and RFWindowDataset reference |
| `code/2nd/model.py` | Conv1D/Conv2D encoder patterns |
| `code/2nd/evaluate.py` | binary metrics, AUROC, file aggregation reference |
| `code/2nd/cross_view_relation/representation_screening.py` | CKA/CCA diagnostic formulas |
| `code/2nd/evaluate_final_oracle_confusion.py` | Oracle SigMF dtype/count/memmap logic reference |

## New Files

| File | Role |
| --- | --- |
| `rf_multiview_relation/configs/default.yaml` | main experiment configuration |
| `rf_multiview_relation/configs/representation.yaml` | later representation-only overrides |
| `rf_multiview_relation/configs/experiment.yaml` | later experiment run overrides |
| `rf_multiview_relation/utils/config.py` | config loading; use PyYAML when available and stdlib fallback for Phase 0 |
| `rf_multiview_relation/utils/io.py` | JSON/CSV output helpers |
| `rf_multiview_relation/utils/seed.py` | reproducibility seed helpers |
| `rf_multiview_relation/data/sigmf.py` | clean Oracle SigMF discovery and dtype/count/memmap helpers |
| `rf_multiview_relation/data/splits.py` | file-level Tx1 train/calibration/holdout split and leakage assertions |
| `rf_multiview_relation/data/windowing.py` | deterministic window starts and p60 aggregation helpers |
| `rf_multiview_relation/data/representations.py` | IQ/AP/STFT representation builders matching the spec |
| `rf_multiview_relation/data/dataset.py` | PyTorch datasets for MAT and Oracle windows |
| `rf_multiview_relation/models/encoder_iq.py` | IQ Conv1D encoder |
| `rf_multiview_relation/models/encoder_ap.py` | AP Conv1D encoder |
| `rf_multiview_relation/models/encoder_stft.py` | STFT Conv2D encoder |
| `rf_multiview_relation/models/autoencoder.py` | view-specific AE wrappers |
| `rf_multiview_relation/models/decoders.py` | lightweight decoders |
| `rf_multiview_relation/relation/cca.py` | CCA fit/transform using Tx1 train only |
| `rf_multiview_relation/relation/cka.py` | dataset-level linear CKA |
| `rf_multiview_relation/relation/relation_features.py` | compact, raw, and CCA residual relation vectors |
| `rf_multiview_relation/relation/covariance.py` | Ledoit-Wolf or shrinkage covariance fit |
| `rf_multiview_relation/relation/scoring.py` | Mahalanobis scoring and threshold application |
| `rf_multiview_relation/baselines/single_view.py` | IQ/AP/STFT-only baseline |
| `rf_multiview_relation/baselines/concat.py` | feature concatenation baseline |
| `rf_multiview_relation/baselines/raw_relation.py` | raw pairwise relation without CCA |
| `rf_multiview_relation/scripts/00_audit_dataset.py` | Phase 0 dataset audit |
| `rf_multiview_relation/scripts/01_extract_views.py` | representation sanity figures |
| `rf_multiview_relation/scripts/02_train_autoencoders.py` | Tx1-only AE pretraining |
| `rf_multiview_relation/scripts/03_extract_latents.py` | latent export |
| `rf_multiview_relation/scripts/04_analyze_relations.py` | CCA/CKA relation analysis |
| `rf_multiview_relation/scripts/05_fit_relation_model.py` | CCA residual + covariance fitting |
| `rf_multiview_relation/scripts/06_evaluate.py` | main/baseline evaluation |
| `rf_multiview_relation/scripts/07_run_all.py` | one-command reproduction |

## Modified Files

Initial phase should avoid changing old experiment code.

Allowed small modifications later:

- `.gitignore`: only if new local outputs/checkpoints/caches are not already ignored.
- `README.md`: keep high-level research status and compact result summary only.
- `code/2nd/requirements.txt` or new `rf_multiview_relation/requirements.txt`: add `pyyaml`, `scikit-learn`, and `matplotlib` when needed.

## Implementation Order

1. Phase 0 Dataset Audit
2. Phase 1 Representation Check
3. Phase 2 Autoencoder Training
4. Phase 3 Latent Extraction
5. Phase 4 CCA/CKA Relation Analysis
6. Phase 5 Relation Detector
7. Phase 6 Evaluation
8. Phase 7 Baselines and Ablation
9. Final `07_run_all.py` orchestration

## Expected Tensor Shapes

Raw MAT IQ:

```text
load_iq_from_mat -> [N, 2]
```

Windowed IQ:

```text
windows -> [num_windows, 2048, 2]
```

IQ view:

```text
[B, 2, 2048]
```

AP view:

```text
[B, 2, 2048]
channel 0 = amplitude
channel 1 = phase
```

STFT view:

```text
[B, 1, 128, 31]
```

With `n_fft=128`, `win_length=128`, `hop_length=64`, and `window_size=2048`, the expected frame count is:

```text
1 + floor((2048 - 128) / 64) = 31
```

Latents:

```text
z_iq   -> [B, 64]
z_ap   -> [B, 64]
z_stft -> [B, 64]
```

CCA projected pair:

```text
u, v -> [B, 16]
```

Proposed relation vector:

```text
[r_iq_ap; r_iq_stft; r_ap_stft] -> [B, 48]
```

Compact relation baseline:

```text
3 pairs * 2 metrics -> [B, 6]
```

Concat baseline:

```text
[z_iq; z_ap; z_stft] -> [B, 192]
```

## Experiment Commands

Phase 0:

```bash
python rf_multiview_relation/scripts/00_audit_dataset.py --config rf_multiview_relation/configs/default.yaml
```

Phase 1:

```bash
python rf_multiview_relation/scripts/01_extract_views.py --config rf_multiview_relation/configs/default.yaml
```

AE training:

```bash
python rf_multiview_relation/scripts/02_train_autoencoders.py --config rf_multiview_relation/configs/default.yaml
```

Latent extraction:

```bash
python rf_multiview_relation/scripts/03_extract_latents.py --config rf_multiview_relation/configs/default.yaml
```

Relation analysis:

```bash
python rf_multiview_relation/scripts/04_analyze_relations.py --config rf_multiview_relation/configs/default.yaml
```

Relation detector:

```bash
python rf_multiview_relation/scripts/05_fit_relation_model.py --config rf_multiview_relation/configs/default.yaml
```

Evaluation:

```bash
python rf_multiview_relation/scripts/06_evaluate.py --config rf_multiview_relation/configs/default.yaml
```

Final reproduction command:

```bash
python rf_multiview_relation/scripts/07_run_all.py --config rf_multiview_relation/configs/default.yaml
```

## Data Leakage Prevention

File-level rules:

- split files before creating windows
- assert `train_files ∩ calibration_files = empty`
- assert `train_files ∩ holdout_files = empty`
- assert `calibration_files ∩ holdout_files = empty`
- assert Tx2-Tx8 and Oracle files are never in train/calibration

Fitting rules:

- encoder training devices must be `Tx1` only
- CCA fitting devices must be `Tx1` only
- scaler fitting devices must be `Tx1` only
- covariance fitting devices must be `Tx1` only
- threshold fitting devices must be Tx1 calibration only

Code assertions:

```python
assert all(device == "Tx1" for device in encoder_train_devices)
assert all(device == "Tx1" for device in cca_fit_devices)
assert all(device == "Tx1" for device in covariance_fit_devices)
assert all(device == "Tx1" for device in threshold_fit_devices)
```

Protocol rules:

- Do not tune `file_score_percentile`, CCA components, AE epochs, threshold percentile, or feature options using Tx2-Tx8.
- Treat Tx2-Tx8 as final closed dataset anomaly evaluation only.
- Treat Oracle SigMF as external anomaly evaluation only.
- Keep raw data, outputs, checkpoints, latents, `.npz`, `.pt`, `.pkl`, and logs out of GitHub.

## Phase 0 Output

Local outputs:

```text
outputs/dataset_audit.json
outputs/tables/dataset_audit_devices.csv
outputs/tables/dataset_audit_oracle.csv
outputs/splits/tx1_train_320_seed42.txt
outputs/splits/tx1_calibration_80_seed42.txt
outputs/splits/tx1_holdout_100_seed42.txt
```

The output files are intentionally local because `outputs/` is ignored by Git.
