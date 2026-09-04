# RF Multi-View Follow-up Research

## Research Topic

This repository tracks a follow-up study on one-class unseen-transmitter detection with multi-view RF representations.

The core question is whether relationships between IQ, amplitude/phase, FFT, and STFT views can improve anomaly detection when only Tx1 normal data is available during training and calibration.

## Evaluation Boundary

This study uses a strict one-class protocol.

- Train and calibration data: Tx1 train-normal only.
- Normal evaluation data: held-out Tx1 files only.
- Anomaly evaluation data: Tx2 through Tx8 files only.
- No Tx1 holdout, Tx2-Tx8, oracle, test, or anomaly files may be used for representation choice, threshold fitting, feature selection, density fitting, or score normalization.
- Final metrics are reported at file level.
- Results must not be described as receiver, channel, distance, or environment robustness unless a separate controlled dataset supports that claim.

## Current Project Root

Use this folder as the working root:

```text
C:\Users\Beomm\Desktop\project\모델 관련 자료\project
```

The active experiment code is mainly under:

```text
code\2nd
```

The usable Python environment is:

```text
code\2nd\.venv\Scripts\python.exe
```

## GitHub Tracking Policy

GitHub is used for source code, experiment design, reproducible commands, compact result tables, and this README.

Do not commit raw RF data, model checkpoints, virtual environments, feature caches, latent arrays, temporary scratch outputs, or runtime logs.

New experiment logs are local-only. Result summaries may be committed when they are small and directly support the README experiment table.

Allowed examples:

- `README.md`
- source scripts
- manifest-generation scripts
- small `.csv`, `.json`, or `.md` result summaries
- baseline reference summaries

Ignored examples:

- `data/`
- `.venv/`
- `checkpoints/`
- `artifacts/logs/`
- `*.log`
- `*.out`
- `*.err`
- `*.pth`
- `*.pt`
- `*.npy`
- `*.npz`

## Uploaded Data State

The uploaded RF data is available under `data/`.

| Device | File count |
| --- | ---: |
| Tx1 | 499 |
| Tx2 | 500 |
| Tx3 | 500 |
| Tx4 | 500 |
| Tx5 | 500 |
| Tx6 | 500 |
| Tx7 | 500 |
| Tx8 | 500 |

The existing rigorous split manifests are under:

```text
code\2nd\exp_rigorous\manifests
```

Current split files:

| Manifest | Count | Role |
| --- | ---: | --- |
| `tx1_train_80_seed42.txt` | 399 | Tx1 train-normal |
| `tx1_test_20_seed42.txt` | 100 | Tx1 held-out normal |
| `tx2_all.txt` | 500 | Prior anomaly manifest |

The uploaded manifest paths currently point to an older absolute root, so the first housekeeping task is to rebuild or rebase manifests against the current `data/` folder while preserving the same Tx1 train/test basenames.

## Existing Baselines

The current reproducible baseline bundle is:

```text
code\2nd\current_baseline_model
```

Reference baseline:

| Item | Value |
| --- | ---: |
| Score | `fusion_midrank_midlogw2_rank_max` |
| AUC mean | 0.829089 |
| AUC min | 0.796060 |
| p60 F1 mean | 0.942966 |
| p60 F1 min | 0.926441 |

The final hybrid reference code is:

```text
code\2nd\final_model_code_20260621
```

Final hybrid reference:

| Item | Value |
| --- | ---: |
| Score | `feature_tail_score + 0.25 * dagmm_energy_p95_tail` |
| AUC mean | 0.912354 |
| AUC min | 0.827540 |
| operational F1 mean | 0.861915 |
| operational F1 min | 0.616011 |

These baselines are comparison anchors. They are not yet evidence that the new cross-view relation idea works.

## Representation Shapes

Default windowing uses `window_size=2048` and `stride=1024`.

| Representation | Shape |
| --- | --- |
| raw IQ file | `(2000000, 2)` |
| full file windows | `(1952, 2048, 2)` |
| IQ view | `(2, 2048)` |
| AP view, preprocessing helper | `(3, 2048)` |
| AP view, dataset default | `(4, 2048)` |
| FFT view, preprocessing helper | `(1, 2048)` |
| FFT view, dataset default | `(2, 2048)` |
| STFT view, `nperseg=128`, `noverlap=96` | `(128, 61)` |

## Latent Extraction Point

The main latent extraction point is `MultiViewModel.forward(iq, ap, freq)` in:

```text
code\2nd\model.py
```

It returns:

- `z_iq`
- `z_ap`
- `z_f`

The dataset already preserves aligned file paths and window indices, so cross-view relation features can be computed without changing the existing baseline implementation.

## New Experiment Area

Use a dedicated folder for follow-up scripts:

```text
code\2nd\cross_view_relation
```

Planned scripts:

| Script | Purpose |
| --- | --- |
| `build_rf_manifest.py` | Rebase or rebuild manifests for the uploaded folder layout |
| `extract_multiview_latents.py` | Save aligned IQ/AP/FFT latents from an existing checkpoint |
| `representation_screening.py` | Compare IQ, AP, FFT, and STFT one-class separability |
| `analyze_cca_cka.py` | Measure cross-view relation using CCA/CKA-style diagnostics |
| `analyze_relation_distributions.py` | Inspect `abs-diff`, product, and cosine relation distributions |
| `evaluate_relational_oneclass.py` | Evaluate train-normal-fitted relation scores at file level |

## Experiment Order

1. Rebase manifests to the current uploaded `data/` root.
2. Run a baseline smoke check using train-normal calibration only.
3. Export IQ/AP/FFT latents from the existing DAGMM-compatible checkpoint.
4. Build cross-view relation features from aligned latents.
5. Fit one-class scoring statistics using Tx1 train-normal only.
6. Evaluate Tx1 holdout versus Tx2-Tx8 at file level.
7. Add STFT screening after the FFT-based path is stable.
8. Compare against the current power-tail baseline and final hybrid reference.

## Experiment Registry

| ID | Status | Purpose | Output to Track |
| --- | --- | --- | --- |
| CVR-00 | Done | Manifest rebase for uploaded folder layout | `code\2nd\cross_view_relation\manifests` |
| CVR-01 | Done | Exploratory representation screening | README summary and compact metric CSV |
| CVR-02 | Planned | IQ/AP/FFT latent export from existing checkpoint | summary only, no latent arrays |
| CVR-03 | Planned | Cross-view relation one-class scoring | file-level metric `.csv` |
| CVR-04 | In CVR-01 | STFT representation screening | compact metric `.csv` |
| CVR-05 | In CVR-01 | CCA/CKA relation diagnostics | compact table or README update |

## First Command

Before running experiments, verify the uploaded data root:

```powershell
cd "C:\Users\Beomm\Desktop\project\모델 관련 자료\project\code\2nd"
.\.venv\Scripts\python.exe -c "from pathlib import Path; root=Path('../../data').resolve(); print(root); print({p.name: len(list(p.glob('*.mat'))) for p in sorted(root.glob('Tx*'))})"
```

The first implementation task is `CVR-00`: create a manifest rebase script that preserves the existing Tx1 split by basename and writes new manifests under the cross-view experiment area.

## CVR-00 Manifest Rebase Result

Manifest rebase completed with zero missing Tx1 basenames.

| Manifest | Count |
| --- | ---: |
| `tx1_train_80_seed42_rebased.txt` | 399 |
| `tx1_test_20_seed42_rebased.txt` | 100 |
| `tx2_all_rebased.txt` | 500 |
| `tx3_all_rebased.txt` | 500 |
| `tx4_all_rebased.txt` | 500 |
| `tx5_all_rebased.txt` | 500 |
| `tx6_all_rebased.txt` | 500 |
| `tx7_all_rebased.txt` | 500 |
| `tx8_all_rebased.txt` | 500 |

## CVR-01 Exploratory Screening Setup

This first screen is not a final model-selection result. It is a feasibility diagnostic for the follow-up idea.

Fitting used Tx1 train-normal only. Tx1 holdout and Tx2-Tx8 were used only after fitting for exploratory evaluation.

Run profile:

| Item | Value |
| --- | --- |
| Tx1 train files | 128 sampled from rebased train manifest |
| Tx1 holdout files | 80 sampled from rebased test manifest |
| Tx2-Tx8 anomaly files | 80 per device |
| windows per file | 16 |
| window size / stride | 2048 / 1024 |
| score threshold | train-normal p95 |
| scoring methods | diagonal z-distance, PCA residual |
| CCA/CKA input | Tx1 train-normal only |

Representations screened:

- IQ summary statistics
- amplitude/phase statistics
- FFT log-magnitude summary
- STFT log-magnitude summary
- cepstral coefficients from FFT log magnitude followed by DCT
- cyclostationary spectral-correlation proxy
- higher-order statistics and cumulants
- reduced-grid bispectrum proxy

The cyclostationary and bispectrum implementations are intentionally lightweight proxies for the first screen. They are meant to test whether signal exists before adding a heavier full spectral-correlation or dense bispectrum estimator.

## CVR-01 Representation Screening Result

Ranking by file-level AUC:

| Rank | Representation | Method | AUC | Min Tx AUC | F1 | FP / 80 normal | FN / 560 anomaly |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Cyclostationary proxy | PCA residual | 0.614241 | 0.531563 | 0.440217 | 14 | 398 |
| 2 | FFT | z-distance | 0.566540 | 0.533594 | 0.178914 | 10 | 504 |
| 3 | Bispectrum proxy | z-distance | 0.550982 | 0.498281 | 0.220126 | 6 | 490 |
| 4 | AP | PCA residual | 0.544464 | 0.512656 | 0.340058 | 16 | 442 |
| 5 | Bispectrum proxy | PCA residual | 0.536629 | 0.465625 | 0.696872 | 44 | 237 |

Interpretation:

- The strongest single representation in the more stable run was the cyclostationary spectral-correlation proxy, but the AUC is still modest.
- AP looked much stronger in the smaller 64-train-file screen, reaching AUC 0.763707, but that signal weakened when the Tx1 train sample was increased to 128 files. This means the AP result should be treated as unstable for now.
- F1 is not the primary criterion here because the exploratory evaluation has many more anomaly files than normal files. AUC and Tx1 holdout false positives are more informative.
- No single handcrafted representation is strong enough yet to claim a new method. The useful signal is more likely in cross-view relation or fusion.

## CVR-01 CKA / CCA Result

Highest linear CKA pairs on Tx1 train-normal:

| Pair | Linear CKA | Split-half CCA mean5 |
| --- | ---: | ---: |
| STFT / Cepstral | 0.867930 | 0.840709 |
| FFT / STFT | 0.805816 | 0.851743 |
| FFT / Cepstral | 0.783354 | 0.927945 |
| AP / Cyclostationary proxy | 0.775124 | 0.759429 |
| AP / Cepstral | 0.708757 | 0.806135 |

Lowest linear CKA pairs on Tx1 train-normal:

| Pair | Linear CKA | Split-half CCA mean5 |
| --- | ---: | ---: |
| IQ / Bispectrum proxy | 0.259008 | 0.486570 |
| IQ / Cyclostationary proxy | 0.298461 | 0.493240 |
| IQ / STFT | 0.307034 | 0.426254 |
| IQ / HOS cumulants | 0.310686 | 0.916320 |
| IQ / Cepstral | 0.329354 | 0.636425 |

Interpretation:

- FFT, STFT, and cepstral features are highly redundant. They should not all be treated as independent evidence without a relation/fusion control.
- AP and cyclostationary proxy have high CKA and were also among the more promising screening signals, so their relationship is a strong first cross-view candidate.
- IQ with bispectrum or cyclostationary proxy has low CKA, suggesting possible complementary information.
- Split-half CCA remains high for several pairs, so CCA should be used as a diagnostic rather than proof of anomaly separability.

## CVR-01 Decision

The follow-up topic is researchable, but not yet proven.

The next experiment should move from single-representation screening to explicit cross-view relation scoring. The first candidates are:

- AP with cyclostationary proxy
- IQ with bispectrum proxy
- IQ with cyclostationary proxy
- FFT/STFT/cepstral as a controlled redundant frequency group

The next scoring features should include cosine distance, absolute difference, elementwise product, PCA residual, and train-normal Mahalanobis-style distance fitted only on Tx1 train-normal.
