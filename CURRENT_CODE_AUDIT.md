# Current Code Audit

Updated on 2026-09-07 for the Tx1-only one-class multi-view relation specification.

## Repository Snapshot

Working root:

```text
C:\Users\Beomm\Desktop\project\모델 관련 자료\project
```

Main research package:

```text
rf_multiview_relation/
```

Legacy experiment code remains under `code/` and is used only as a reference or for small reusable utilities. The new protocol must not reuse the previous Tx1-Tx4 pooled-normal assumption as the main study.

## Existing Code Audit Matrix

| Existing file | Existing function/class | Purpose | Can reuse? | Required modification | Risk |
| --- | --- | --- | --- | --- | --- |
| `code/2nd/preprocessing.py` | `load_iq_from_mat` | Load MATLAB `rxData` complex samples as `[N,2]` I/Q float array | Yes | Keep as low-level MAT loader | Low |
| `code/2nd/preprocessing.py` | `validate_iq` | Check `[N,2]`, NaN/Inf, minimum length | Yes | Reuse in audit/dataset scripts | Low |
| `code/2nd/preprocessing.py` | `normalize_iq` | Power/zscore/minmax/DC normalization | Partial | Main spec uses window-level energy normalization, so use this only for audit or file-level sanity checks | Medium |
| `code/2nd/preprocessing.py` | `build_iq_view` | Old IQ view construction | Partial | New implementation lives in `rf_multiview_relation/data/representations.py` | Low |
| `code/2nd/preprocessing.py` | `build_ap_view` | Old AP view with phase diff | No for main | Main spec requires 2-channel amplitude/phase only | Medium |
| `code/2nd/preprocessing.py` | `build_freq_view` | FFT/STFT magnitude helper | Reference only | New STFT builder must return `[1,F,T]` and use config names from new spec | Medium |
| `code/2nd/dataset.py` | `RFWindowDataset` | Legacy PyTorch MAT window dataset | Reference only | New dataset must preserve Tx1-only split boundaries and exact IQ/AP/STFT shapes | Medium |
| `code/2nd/dataset.py` | file-window index construction | Uniform `max_windows_per_file` selection | Yes | Reimplemented in `rf_multiview_relation/data/windowing.py` | Low |
| `code/2nd/evaluate_final_oracle_confusion.py` | `discover_oracle_files` | Locate Oracle `.sigmf-data` files | Partial | Extracted into clean `rf_multiview_relation/data/sigmf.py` | Low |
| `code/2nd/evaluate_final_oracle_confusion.py` | `sigmf_dtype_and_count`, `sigmf_memmap` | Infer SigMF dtype/count and memmap | Yes | Reimplemented without importing monolithic final evaluator | Medium |
| `code/2nd/model.py` | `RFEncoder` | Existing Conv1D RF encoder pattern | Reference only | New IQ/AP encoders follow spec exactly and use 64-D output | Low |
| `code/2nd/model.py` | `STFTConv2DEncoder` | Existing Conv2D STFT encoder pattern | Reference only | New `STFTEncoder` follows current spec and decoder pair | Low |
| `code/2nd/model.py` | `MultiViewModel`, `consistency_loss` | Previous multi-view consistency model | No for main | Current method uses AE pretraining, CCA relation, Mahalanobis | Medium |
| `code/2nd/dagmm_module.py` | `DAGMM` | Deep density model | Optional later | Treat only as low-priority comparison, not proposed method | Medium |
| `code/2nd/evaluate.py` | `binary_metrics`, `roc_auc_score` | Evaluation helpers | Partial | New `utils/metrics.py` adds AUROC/AUPRC/FPR/TNR with consistent names | Low |
| `code/2nd/evaluate.py` | `aggregate_by_file` | Legacy file score aggregation | Partial | New main needs p60 plus ablation modes; implement in new scoring/evaluation path | Medium |
| `code/2nd/cross_view_relation/representation_screening.py` | `linear_cka`, `cca_summary` | Previous relation diagnostics | Reference only | New `relation/cka.py` and `relation/cca_alignment.py` follow Tx1-only latent protocol | Medium |
| `code/one_class_self_consistency/run_experiment.py` | `aggregate_file_scores`, `metrics_at_threshold` | One-class scoring examples | Reference only | Useful logic, but not the current neural CCA relation pipeline | Medium |

## New Code Already Established

| New file | Purpose | Status |
| --- | --- | --- |
| `rf_multiview_relation/configs/default.yaml` | Canonical Tx1-only one-class config | Re-established to new spec |
| `rf_multiview_relation/configs/ablation.yaml` | Planned ablation settings | Added |
| `rf_multiview_relation/data/windowing.py` | Deterministic window count/start helpers | Implemented |
| `rf_multiview_relation/data/splits.py` | Tx1 train/calibration/holdout split and leakage assertions | Implemented |
| `rf_multiview_relation/data/sigmf.py` | Clean Oracle SigMF discovery and metadata parsing | Implemented |
| `rf_multiview_relation/data/representations.py` | IQ/AP/STFT builders matching `[2,2048]`, `[2,2048]`, `[1,128,31]` | Implemented |
| `rf_multiview_relation/models/*` | IQ/AP/STFT encoders, decoders, AE wrapper | Implemented and smoke-tested |
| `rf_multiview_relation/relation/*` | CCA alignment, CKA, relation feature helpers | Established |
| `rf_multiview_relation/detectors/*` | Mahalanobis, concat, single-view, relation, score fusion helpers | Established |
| `rf_multiview_relation/utils/metrics.py` | AUROC/AUPRC/binary metric helpers | Established |
| `rf_multiview_relation/scripts/00_audit_dataset.py` | Dataset audit | Implemented and run |
| `rf_multiview_relation/scripts/01_verify_representations.py` | Canonical representation verification entrypoint | Added as spec-aligned wrapper |
| `rf_multiview_relation/scripts/02_train_autoencoders.py` | Tx1-only AE pretraining | Implemented and smoke-tested |

## Audit Findings

Dataset audit results from Phase 0:

| Dataset | Files | Sample length | Possible windows/file | Selected windows/file |
| --- | ---: | ---: | ---: | ---: |
| Tx1 | 500 | 2,000,000 | 1,952 | 256 |
| Tx2 | 500 | 2,000,000 | 1,952 | 256 |
| Tx3 | 500 | 2,000,000 | 1,952 | 256 |
| Tx4 | 500 | 2,000,000 | 1,952 | 256 |
| Tx5 | 500 | 2,000,000 | 1,952 | 256 |
| Tx6 | 500 | 2,000,000 | 1,952 | 256 |
| Tx7 | 500 | 2,000,000 | 1,952 | 256 |
| Tx8 | 500 | 2,000,000 | 1,952 | 256 |
| Oracle SigMF | 128 standard `.sigmf-data` | 20,006,400 | 19,536 | 256 |

Tx1 split:

```text
Tx1 fit/train:   320 files
Tx1 calibration:  80 files
Tx1 holdout:     100 files
```

Representation verification:

```text
IQ:   [2, 2048]
AP:   [2, 2048]
STFT: [1, 128, 31]
```

AP uses `phase_unwrap: true` in the main config. The smoke check showed that unwrapped phase can have a much larger MSE scale than IQ/STFT, so `phase_unwrap: false` or AP channel scaling should be kept as an ablation candidate only after the main fixed-protocol run.

## Current Gaps

The following parts still need implementation before full paper-level evaluation:

| Gap | Needed module/script |
| --- | --- |
| Full AE training run | `scripts/02_train_autoencoders.py` with all 320 Tx1 fit files |
| Latent extraction | `scripts/03_extract_latents.py` |
| CCA/CKA tables | `scripts/04_analyze_relations.py`, `relation/*` |
| Single-view/concat/relation detector fitting | `scripts/05_fit_detectors.py`, `detectors/*` |
| Minimal Tx1-vs-Tx2 feasibility | `scripts/06_evaluate.py` with mode or separate config |
| Absolute + relation combined detector | `detectors/combined.py` plus evaluation integration |
| Full Tx2-Tx8 and Oracle evaluation | `scripts/06_evaluate.py` |
| Ablations | `scripts/07_ablation.py` |
| Final orchestration | `scripts/08_run_all.py` |

## Reuse Decision

Main reuse is intentionally narrow:

- Reuse old MAT/SigMF loading knowledge.
- Reuse old model patterns only as references.
- Do not reuse prior Tx1-Tx4 known-normal experiment as current evidence.
- Do not tune any choice using Tx2-Tx8 or Oracle.

The current main research question is:

```text
Does explicit multi-view relation provide complementary anomaly information
beyond learned absolute latent features in Tx1-only one-class RF anomaly detection?
```
