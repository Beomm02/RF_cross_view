# Current Code Audit

## Scope

This audit summarizes the current repository before starting the new Tx1-only one-class multi-view relation project.

Main working root:

```text
C:\Users\Beomm\Desktop\project\모델 관련 자료\project
```

Current Git remote:

```text
https://github.com/Beomm02/RF_cross_view.git
```

## 1. Repository Structure

Top-level structure:

```text
.
├── README.md
├── .gitignore
├── code/
│   ├── 2nd/
│   ├── multi-cons/
│   ├── one_class_self_consistency/
│   └── tools/
├── data/
│   ├── Tx1/
│   ├── Tx2/
│   ├── ...
│   ├── Tx8/
│   └── oracle/
├── docs/
├── outputs/
└── analysis_output/
```

Git currently tracks the compact cross-view relation scripts/results and `code/2nd/preprocessing.py`. Many older experiment files under `code/2nd`, `code/multi-cons`, `code/one_class_self_consistency`, `docs`, and `outputs` exist locally as untracked legacy material. They should not be deleted, but the new project should be isolated in `rf_multiview_relation/`.

## 2. RF Dataset Loader Location

Reusable MAT/window dataset code:

- `code/2nd/dataset.py`
  - `resolve_mat_files(data_source)`
  - `RFWindowDataset`
  - file-level window index bookkeeping
  - returns `(iq_view, ap_view, freq_view, path, window_idx)`

Low-level MAT IQ loader:

- `code/2nd/preprocessing.py`
  - `load_iq_from_mat(mat_path, key="rxData")`
  - `validate_iq(iq, min_len)`
  - `normalize_iq(iq, mode)`

Older duplicate preprocessing exists at `code/preprocessing.py`, but `code/2nd/preprocessing.py` is the better reuse target because it already supports `power`, `power_dc`, `dc_only`, and STFT options.

## 3. SigMF Loader Location

Oracle SigMF support currently exists in:

- `code/2nd/evaluate_final_oracle_confusion.py`
  - `discover_oracle_files(oracle_root)`
  - `sigmf_dtype_and_count(path)`
  - `sigmf_memmap(path)`
  - `complex_windows(...)`
  - `OracleSigMFWindowDataset`

This file is monolithic and imports final-model evaluation code at module import time. For the new project, the SigMF dtype/count/memmap logic should be moved into a clean standalone module rather than importing the whole final Oracle evaluation script.

## 4. Windowing Implementation Location

Reusable windowing implementations:

- `code/2nd/preprocessing.py`
  - `make_windows(iq, window_size=2048, stride=1024)`
- `code/2nd/dataset.py`
  - computes `n_windows_all = 1 + (len - window_size) // stride`
  - supports uniform subsampling with `max_windows_per_file`
- `code/evaluate_device_identification_multiclass.py`
  - `window_starts(n_samples, window_size, windows_per_file)`
- `code/2nd/evaluate_final_oracle_confusion.py`
  - `complex_windows(...)` for SigMF arrays

The new implementation should keep file-level split first, then generate windows inside each split. This is the most important leakage boundary.

## 5. Existing IQ/AP/FFT Preprocessing Location

Primary reusable source:

- `code/2nd/preprocessing.py`
  - IQ: `build_iq_view`
  - AP: `build_ap_view`
  - FFT/STFT: `build_freq_view`

Important mismatch with the new specification:

- Existing AP view is usually 3-channel or 4-channel (`amp, phase, phase_diff` or `amp, sin(phase), cos(phase), phase_diff`).
- New study requires the initial AP representation to be 2-channel: `amplitude` and `phase`, with `phase_unwrap` configurable.
- Existing frequency view can be FFT or STFT, but STFT is returned as a 2D array that older code sometimes treats as Conv1D channels. New study should use explicit STFT 2D input shape `[1, frequency_bins, time_frames]`.

Therefore, the new project should reuse `load_iq_from_mat`, `normalize_iq`, and window selection logic, but implement clean IQ/AP/STFT representation builders matching the new tensor shapes.

## 6. Existing CNN/AE/DAGMM Model Location

Reusable neural modules:

- `code/2nd/model.py`
  - `RFEncoder`
  - `STFTConv2DEncoder`
  - `STFTHybridEncoder`
  - `MultiViewModel`
  - `consistency_loss`
- `code/2nd/dagmm_module.py`
  - `DAGMM`
  - GMM parameter estimation
  - sample energy
- `code/one_class_self_consistency/run_experiment.py`
  - lightweight MLP `ViewEncoder`
  - `SelfConsistencyAE`
  - one-class train/test scoring example
- `code/multi-cons/`
  - older multi-view DAGMM training/evaluation variants

For the new KCI-oriented study, DAGMM should be treated as optional later comparison. The main path should use simpler view-specific autoencoders, CCA alignment, relation residuals, and Mahalanobis scoring.

## 7. Existing Evaluation Metric Code

Reusable metric implementations:

- `code/2nd/evaluate.py`
  - `binary_metrics`
  - `roc_auc_score`
  - file-level print/evaluation utilities
- `code/2nd/cross_view_relation/representation_screening.py`
  - `binary_metrics`
  - `roc_auc_score`
  - `linear_cka`
  - `cca_summary`
- `code/one_class_self_consistency/run_experiment.py`
  - `metrics_at_threshold`
  - `best_f1`
- `code/evaluate_device_identification_multiclass.py`
  - multiclass metrics and file aggregation, but this is not the main one-class target.

Current environment note:

- `scikit-learn` is not installed in the active `code/2nd/.venv`.
- The new project can run Phase 0 without it.
- Later CCA, LedoitWolf, AUPRC, and plotting should either add `scikit-learn`, `PyYAML`, and `matplotlib` to requirements or implement minimal equivalents locally.

## 8. Existing File-Level Score Aggregation Code

Reusable aggregation implementations:

- `code/2nd/evaluate.py`
  - `aggregate_by_file(energies, paths, mode)`
  - supports `mean`, `max`, `p95`
- `code/one_class_self_consistency/run_experiment.py`
  - `aggregate_file_scores`
  - currently uses mean and p90 examples
- `code/evaluate_device_identification_multiclass.py`
  - `aggregate_scores`
  - supports `mean`, `median`, `trimmed_mean`, `p75`, `p90`, `logit_mean`

New requirement:

- Main file score aggregation is `p60`.
- Ablation should include `mean`, `p50`, `p60`, `p75`, `p90`, `p95`, and `max`.
- The p60 choice must be fixed before looking at Tx2-Tx8 anomaly results.

## 9. Reusable Code

Recommended reuse:

- MAT loader: `code/2nd/preprocessing.py::load_iq_from_mat`
- IQ validation: `code/2nd/preprocessing.py::validate_iq`
- normalization: `code/2nd/preprocessing.py::normalize_iq`
- window count/index policy: adapt from `code/2nd/dataset.py`
- simple binary AUROC/F1 code: adapt from `code/2nd/evaluate.py`
- CKA/CCA diagnostic math: adapt from `code/2nd/cross_view_relation/representation_screening.py`
- Oracle SigMF parsing: extract the minimal logic from `code/2nd/evaluate_final_oracle_confusion.py`
- CNN encoder patterns: adapt from `code/2nd/model.py`

Do not reuse directly for the main method:

- old Tx1-Tx4 known-normal pooled experiment as the main protocol
- supervised multiclass classification scripts
- DAGMM as the main proposed detector
- result-driven threshold or hyperparameter selection using Tx2-Tx8

## 10. Code To Implement

New project code should be added under:

```text
rf_multiview_relation/
```

Required new modules:

- config loading and default config
- dataset audit script
- clean split manager with file-level leakage assertions
- representation builders for IQ/AP/STFT matching the new spec
- view-specific encoder/decoder autoencoders
- latent extraction
- CCA/CKA analysis
- relation residual feature construction
- covariance/Mahalanobis scoring
- p60 file aggregation
- baseline evaluation for IQ-only, AP-only, STFT-only, concat, raw relation, and proposed CCA relation
- plotting utilities
- `scripts/07_run_all.py`

Immediate next step:

```text
Phase 0 Dataset Audit
```

This should create `outputs/dataset_audit.json` locally and print a terminal summary. Because `outputs/` is ignored by `.gitignore`, the JSON audit output will remain local unless explicitly added later.
