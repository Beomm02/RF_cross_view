# Current Code Audit

Updated on 2026-09-07 after re-establishing the repository around the final Tx1-only specification.

## Scope

This repository is now scoped to:

```text
RF Multi-View Relation 기반 One-Class 이상 탐지
```

Current main package:

```text
rf_multiview_relation/
```

Raw RF data remains local under `data/` and is not tracked by Git.

## Cleanup Decision

Files not aligned with the current specification are removed from the active repository:

- legacy `code/` experiments
- legacy `docs/` materials
- legacy cross-view handcrafted feature experiments
- old Tx1-Tx4 known-normal result tables
- old project organization notes
- old generated report/presentation outputs

The only retained source structure is the new `rf_multiview_relation/` package plus top-level research control documents.

## Current Repository Structure

| Path | Purpose | Based on current spec? | Risk |
| --- | --- | --- | --- |
| `.gitignore` | Prevent raw data, outputs, checkpoints, caches, logs from Git upload | Yes | Low |
| `README.md` | Current research overview and run commands | Yes | Low |
| `CURRENT_CODE_AUDIT.md` | Active-code audit after cleanup | Yes | Low |
| `IMPLEMENTATION_PLAN.md` | Implementation plan for the final pipeline | Yes | Low |
| `rf_multiview_relation/configs/default.yaml` | Main Tx1-only configuration | Yes | Low |
| `rf_multiview_relation/configs/ablation.yaml` | Planned ablation settings | Yes | Low |
| `rf_multiview_relation/data/mat.py` | MAT `rxData` IQ loader and validation | Yes | Low |
| `rf_multiview_relation/data/sigmf.py` | Oracle SigMF discovery and metadata parsing | Yes | Medium |
| `rf_multiview_relation/data/splits.py` | Tx1 file-level split and leakage checks | Yes | Low |
| `rf_multiview_relation/data/windowing.py` | Deterministic window selection | Yes | Low |
| `rf_multiview_relation/data/representations.py` | IQ/AP/STFT representation builders | Yes | Medium |
| `rf_multiview_relation/data/dataset.py` | Streaming view batch iterator | Yes | Medium |
| `rf_multiview_relation/models/*` | IQ/AP/STFT encoders, decoders, autoencoder wrapper | Yes | Medium |
| `rf_multiview_relation/relation/*` | CCA alignment, CKA, relation feature helpers | Yes | Medium |
| `rf_multiview_relation/detectors/*` | Mahalanobis and score fusion helpers | Yes | Medium |
| `rf_multiview_relation/utils/*` | Config, IO, metrics, plotting, seed helpers | Yes | Low |
| `rf_multiview_relation/scripts/00_audit_dataset.py` | Phase 0 dataset audit | Yes | Low |
| `rf_multiview_relation/scripts/01_verify_representations.py` | Phase 1 representation verification | Yes | Low |
| `rf_multiview_relation/scripts/02_train_autoencoders.py` | Phase 2 Tx1-only AE training | Yes | Medium |
| `rf_multiview_relation/scripts/03_extract_latents.py` | Phase 3 paired latent extraction | Yes | Medium |
| `rf_multiview_relation/scripts/04_analyze_relations.py` | Phase 4 CCA/CKA relation analysis | Yes | Medium |
| `rf_multiview_relation/scripts/05_fit_relation_model.py` | Phase 5 Tx1-only detector fitting and calibration | Yes | Medium |
| `rf_multiview_relation/scripts/06_evaluate.py` | Phase 6 metrics, statistical tests, figures | Yes | Medium |
| `rf_multiview_relation/scripts/08_run_all.py` | Full pipeline orchestration | Yes | Low |

## Implemented Functions And Classes

| File | Function/class | Purpose | Can reuse? | Required modification | Risk |
| --- | --- | --- | --- | --- | --- |
| `data/mat.py` | `load_iq_from_mat` | Load complex MAT `rxData` as `[N,2]` | Yes | None | Low |
| `data/mat.py` | `validate_iq` | Validate IQ shape, length, NaN/Inf | Yes | None | Low |
| `data/mat.py` | `normalize_iq` | Audit-time normalization check | Yes | Main training uses window energy normalization | Low |
| `data/sigmf.py` | `discover_sigmf_data_files` | Find standard Oracle `.sigmf-data` | Yes | Add full Oracle window dataset in Phase 3 | Medium |
| `data/sigmf.py` | `infer_sigmf_dtype_and_count` | Infer complex64/complex128 and count | Yes | None | Medium |
| `data/splits.py` | `split_tx1_files` | Create 320/80/100 Tx1 split | Yes | None | Low |
| `data/splits.py` | `assert_disjoint` | File-level leakage assertion | Yes | Use in all later scripts | Low |
| `data/windowing.py` | `window_start_positions` | Deterministic selected window starts | Yes | None | Low |
| `data/representations.py` | `build_iq_view` | IQ `[2,2048]` | Yes | None | Low |
| `data/representations.py` | `build_ap_view` | AP `[2,2048]` with configurable unwrap | Yes | Later ablation for raw phase/scaling | Medium |
| `data/representations.py` | `build_stft_view` | STFT log magnitude `[1,128,31]` | Yes | None | Medium |
| `models/encoder_iq.py` | `IQEncoder` | Conv1D IQ encoder to 64-D latent | Yes | None | Medium |
| `models/encoder_ap.py` | `APEncoder` | Independent AP Conv1D encoder | Yes | None | Medium |
| `models/encoder_stft.py` | `STFTEncoder` | Conv2D STFT encoder to 64-D latent | Yes | None | Medium |
| `models/autoencoder.py` | `ViewAutoencoder`, `make_autoencoder` | AE wrapper/factory | Yes | None | Medium |
| `relation/cca_alignment.py` | `CCAAlignment` | Tx1-only CCA fitting and transform | Yes | Validate on real latents | Medium |
| `relation/cka.py` | `linear_cka` | Dataset-level relation analysis | Yes | None | Low |
| `relation/relation_features.py` | `residual_relation`, `distance_relation` | CCA/raw relation features | Yes | Integrate into detector scripts | Medium |
| `detectors/mahalanobis.py` | `MahalanobisDetector` | LedoitWolf if available, fallback shrinkage otherwise | Yes | Prefer sklearn when installed | Medium |
| `detectors/combined.py` | `RobustScoreFusion` | Fixed-alpha absolute/relation score fusion | Yes | Fit reference on Tx1 calibration only | Medium |
| `pipeline.py` | shared helpers | CCA fitting, relation features, leakage checks, aggregation, pickle IO | Yes | None | Medium |
| `utils/metrics.py` | `binary_metrics` | AUROC/AUPRC/F1/FPR/TNR | Yes | None | Low |

## Completed Checks

Phase 0:

```text
Tx1-Tx8: 500 files each
Tx1 split: 320 train, 80 calibration, 100 holdout
Oracle standard SigMF: 128 files
file-level leakage check: passed
```

Phase 1:

```text
IQ:   [2, 2048]
AP:   [2, 2048]
STFT: [1, 128, 31]
NaN/Inf: none in sampled examples
```

Phase 2 smoke:

```text
IQ/AP/STFT AE forward/backward/checkpoint save: passed
```

Phase 3-6 minimal smoke:

```text
Tx1 train/calibration/holdout + Tx2, 2 files per split
latent extraction: passed
CCA/CKA analysis: passed
Tx1-only detector fitting: passed
Tx1 calibration thresholding: passed
file-level evaluation output: passed
```

## Current Risk Register

| Risk | Mitigation |
| --- | --- |
| Full AE training may be IO-heavy | Start with Tx1-only full run, then decide whether local ignored caches are needed |
| AP unwrapped phase scale is large | Keep main config fixed; compare raw phase/scaling only in ablation |
| `scikit-learn` may be unavailable | `MahalanobisDetector` uses sklearn LedoitWolf when available and analytic Ledoit-Wolf otherwise |
| Oracle format differs from MAT | Keep Oracle handling isolated in `data/sigmf.py` |

## Next Active Step

```text
Run full Tx1-only autoencoder training,
then extract full latents and execute the main evaluation.
```
