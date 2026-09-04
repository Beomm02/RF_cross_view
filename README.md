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

Data check after adding the missing Tx1 file:

| Item | Result |
| --- | --- |
| Tx1 file count | 500 |
| Missing Tx1 iter numbers | none |
| Added file | `RFF_Tx_ANTSDR_1_Boot_01_20265608_020422_iter386.mat` |

The rebased 399/100 Tx1 train/test split remains preserved for comparability. The newly added `iter386` file is not inserted into the existing split unless a fresh 400/100 split is explicitly created later.

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
| CVR-03 | Done | Cross-view relation one-class scoring | file-level metric `.csv` |
| CVR-04 | In CVR-01 | STFT representation screening | compact metric `.csv` |
| CVR-05 | In CVR-01 | CCA/CKA relation diagnostics | compact table or README update |
| CVR-06 | Done | Train-normal rank-calibrated relation fusion | file-level metric `.csv` |
| CVR-07 | Done | Full preserved split relation-fusion threshold sweep | file-level metric `.csv` |

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

## CVR-03 Relation Scoring Setup

This experiment tests whether explicit cross-view relation features improve over the single-representation screen.

Fitting again used Tx1 train-normal only. Evaluation used Tx1 holdout and Tx2-Tx8 only after fitting.

Run profile:

| Item | Value |
| --- | --- |
| Tx1 train files | 128 sampled from rebased train manifest |
| Tx1 holdout files | 80 sampled from rebased test manifest |
| Tx2-Tx8 anomaly files | 80 per device |
| windows per file | 16 |
| per-view projection | train-fitted PCA, 16 components |
| relation features | absolute difference, elementwise product, cosine distance |
| relation scoring | z-distance, PCA residual |
| score threshold | train-normal p95 |

Pairs tested:

- AP / Cyclostationary proxy
- IQ / Bispectrum proxy
- IQ / Cyclostationary proxy
- FFT / STFT
- FFT / Cepstral
- STFT / Cepstral

## CVR-03 Relation Scoring Result

Ranking by file-level AUC:

| Rank | Pair | Method | AUC | Min Tx AUC | F1 | FP / 80 normal | FN / 560 anomaly |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | AP / Cyclostationary proxy | relation PCA residual | 0.610402 | 0.478906 | 0.479791 | 23 | 376 |
| 2 | IQ / Cyclostationary proxy | relation PCA residual | 0.605446 | 0.484844 | 0.403315 | 18 | 414 |
| 3 | IQ / Bispectrum proxy | relation PCA residual | 0.570379 | 0.465313 | 0.409904 | 18 | 411 |
| 4 | IQ / Bispectrum proxy | relation z-distance | 0.569129 | 0.480156 | 0.195827 | 2 | 499 |
| 5 | STFT / Cepstral | relation PCA residual | 0.548795 | 0.477344 | 0.509603 | 22 | 361 |

Interpretation:

- Cross-view relation scoring did not beat the best single-representation screening result, which was Cyclostationary proxy PCA residual at AUC 0.614241.
- AP / Cyclostationary and IQ / Cyclostationary were the strongest relation pairs, which is consistent with the CKA/CCA diagnostic.
- Frequency-only relation pairs such as FFT / STFT and FFT / Cepstral were weak, likely because those views are highly redundant.
- The current relation formulation is not yet strong enough as a final detector. It is still useful as a research direction because pair ranking is coherent with the CKA/CCA structure.

## CVR-03 Decision

Continue the follow-up, but change the next experiment from raw relation PCA residuals to a better-calibrated relation model.

The next candidate should combine:

- the single Cyclostationary proxy score,
- AP / Cyclostationary relation PCA residual,
- IQ / Cyclostationary relation PCA residual,
- IQ / Bispectrum relation with a low-false-positive threshold,
- a redundancy-controlled frequency group rather than separate FFT/STFT/Cepstral relation scores.

The next fitting method should add shrinkage covariance or robust rank calibration using Tx1 train-normal only.

## CVR-06 Rank-Calibrated Relation Fusion Setup

This experiment tests whether the best single score and the strongest relation scores become more useful after train-normal empirical-rank calibration.

All calibration used Tx1 train-normal only.

Components:

- Cyclostationary proxy PCA residual
- AP / Cyclostationary relation PCA residual
- IQ / Cyclostationary relation PCA residual
- IQ / Bispectrum relation z-distance

Fusion methods:

- rank mean
- rank max
- rank top-2 mean
- cyclostationary-weighted rank fusion

Run profile:

| Item | Value |
| --- | --- |
| Tx1 train files | 128 sampled from rebased train manifest |
| Tx1 holdout files | 80 sampled from rebased test manifest |
| Tx2-Tx8 anomaly files | 80 per device |
| windows per file | 16 |
| component calibration | empirical rank against Tx1 train-normal |
| score threshold | fused Tx1 train-normal p95 |

## CVR-06 Rank-Calibrated Relation Fusion Result

Fusion ranking:

| Rank | Fusion | AUC | Min Tx AUC | F1 | FP / 80 normal | FN / 560 anomaly |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | rank mean | 0.635469 | 0.525547 | 0.402219 | 16 | 415 |
| 2 | cyclostationary-weighted rank | 0.633382 | 0.528203 | 0.427793 | 17 | 403 |
| 3 | rank max | 0.625904 | 0.534141 | 0.435724 | 18 | 399 |
| 4 | rank top-2 mean | 0.622768 | 0.511797 | 0.434316 | 24 | 398 |

Best component comparison:

| Score | AUC | F1 | FP / 80 normal | FN / 560 anomaly |
| --- | ---: | ---: | ---: | ---: |
| Cyclostationary proxy PCA residual | 0.614241 | 0.440217 | 14 | 398 |
| AP / Cyclostationary relation PCA residual | 0.610402 | 0.479791 | 23 | 376 |
| IQ / Cyclostationary relation PCA residual | 0.605446 | 0.403315 | 18 | 414 |
| IQ / Bispectrum relation z-distance | 0.569129 | 0.195827 | 2 | 499 |

Per-device AUC for the best fusion, rank mean:

| Device | AUC |
| --- | ---: |
| Tx2 | 0.615859 |
| Tx3 | 0.758516 |
| Tx4 | 0.533438 |
| Tx5 | 0.707422 |
| Tx6 | 0.659687 |
| Tx7 | 0.647813 |
| Tx8 | 0.525547 |

Interpretation:

- Rank-calibrated fusion improved AUC over the best single component, from 0.614241 to 0.635469.
- The improvement is modest but directionally useful, so cross-view relation features should remain in the follow-up study.
- Tx4 and Tx8 remain weak cases, which prevents any strong claim at this stage.
- The best current direction is not relation-only detection. It is calibrated fusion of a cyclostationary single score with selected cross-view relation scores.

## CVR-06 Decision

The next experiment should scale the same protocol before adding more complex modeling.

Recommended next steps:

- run the rank-calibrated fusion on the full preserved 399/100 split and all Tx2-Tx8 files,
- compare p90, p95, and p97 train-normal thresholds,
- add shrinkage covariance scoring for the relation feature space,
- keep Tx4 and Tx8 as explicit failure-analysis targets.

## CVR-07 Full Split Threshold Sweep Setup

This experiment scales CVR-06 to the preserved full split before adding a more complex relation model.

Fitting and calibration used Tx1 train-normal only. Tx1 holdout and Tx2-Tx8 were used only for evaluation.

Run profile:

| Item | Value |
| --- | --- |
| Tx1 train files | 399 |
| Tx1 holdout files | 100 |
| Tx2-Tx8 anomaly files | 500 per device, 3500 total |
| windows per file | 16 |
| components | same as CVR-06 |
| thresholds | train-normal p90, p95, p97 |

## CVR-07 Full Split Threshold Sweep Result

Fusion ranking by AUC:

| Rank | Fusion | AUC | p90 F1 | p90 FP / 100 | p90 FN / 3500 | p95 F1 | p97 F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rank top-2 mean | 0.601646 | 0.346271 | 15 | 2764 | 0.278351 | 0.229914 |
| 2 | rank max | 0.596049 | 0.329608 | 12 | 2807 | 0.264356 | 0.247000 |
| 3 | cyclostationary-weighted rank | 0.593829 | 0.324427 | 12 | 2820 | 0.262766 | 0.240522 |
| 4 | rank mean | 0.591980 | 0.317605 | 12 | 2837 | 0.258016 | 0.233871 |

Best component comparison on the full split:

| Score | AUC | p90 F1 | p90 FP / 100 | p90 FN / 3500 |
| --- | ---: | ---: | ---: | ---: |
| AP / Cyclostationary relation PCA residual | 0.595189 | 0.386327 | 17 | 2658 |
| Cyclostationary proxy PCA residual | 0.594117 | 0.351947 | 12 | 2750 |
| IQ / Bispectrum relation z-distance | 0.568200 | 0.267095 | 10 | 2959 |
| IQ / Cyclostationary relation PCA residual | 0.550620 | 0.324105 | 11 | 2821 |

Per-device AUC for the best full-split fusion, rank top-2 mean:

| Device | AUC | p90 F1 | p90 FN / 500 |
| --- | ---: | ---: | ---: |
| Tx2 | 0.574000 | 0.330632 | 398 |
| Tx3 | 0.620220 | 0.359873 | 387 |
| Tx4 | 0.585340 | 0.308703 | 406 |
| Tx5 | 0.616910 | 0.338710 | 395 |
| Tx6 | 0.600930 | 0.362480 | 386 |
| Tx7 | 0.630140 | 0.380503 | 379 |
| Tx8 | 0.583980 | 0.289037 | 413 |

Interpretation:

- The full-split result is weaker than the sampled CVR-06 result.
- Rank fusion still slightly improves AUC over the best full-split single component, but only from 0.595189 to 0.601646.
- p90 is the best operating threshold among p90, p95, and p97 for F1, but recall remains low.
- The signal is broad but shallow: all anomaly devices are above chance AUC, yet none are cleanly separated.
- Tx8 is still the weakest practical operating case by p90 F1 and false negatives.

## CVR-07 Decision

The current handcrafted relation-fusion path is not strong enough as a standalone detector.

The research should continue only if the next step improves the scoring model itself. The next useful experiment is shrinkage covariance or robust Mahalanobis scoring in the relation feature space, fitted only on Tx1 train-normal, followed by the same full-split p90/p95/p97 evaluation.
