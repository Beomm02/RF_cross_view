# RF Cross-View Feature Relation Study

## 현재 연구 기준

2026-09-04부터 본 저장소의 메인 연구 기준은 다음 주제로 전환한다.

```text
Multi-View Representation 관계를 활용한 RF 송신 장치 이상 탐지
```

핵심 질문은 Tx1 정상 데이터에서 IQ/AP/STFT representation 사이의 관계 구조가 안정적으로 존재하는지, 그리고 Tx2-Tx8 및 Oracle SigMF가 그 Tx1 relation distribution에서 벗어나는지를 확인하는 것이다.

메인 구현 위치:

```text
rf_multiview_relation/
```

초기 문서:

- `CURRENT_CODE_AUDIT.md`
- `IMPLEMENTATION_PLAN.md`

Phase 0 데이터 감사 실행:

```bash
python rf_multiview_relation/scripts/00_audit_dataset.py --config rf_multiview_relation/configs/default.yaml
```

Phase 1 representation 확인 실행:

```bash
python rf_multiview_relation/scripts/01_extract_views.py --config rf_multiview_relation/configs/default.yaml
```

Phase 2 autoencoder pretraining 실행:

```bash
python rf_multiview_relation/scripts/02_train_autoencoders.py --config rf_multiview_relation/configs/default.yaml
```

Phase 0 로컬 산출물:

```text
outputs/dataset_audit.json
outputs/tables/dataset_audit_devices.csv
outputs/tables/dataset_audit_oracle.csv
outputs/splits/tx1_train_320_seed42.txt
outputs/splits/tx1_calibration_80_seed42.txt
outputs/splits/tx1_holdout_100_seed42.txt
```

위 산출물은 `outputs/` 아래에 생성되며 GitHub에는 업로드하지 않는다.

## 현재 진행 상태

| Phase | 상태 | 핵심 확인 |
| --- | --- | --- |
| Phase 0 Dataset Audit | 완료 | Tx1-Tx8 각 500 files, Tx1 train/calibration/holdout 320/80/100 split, Oracle 표준 SigMF 128 files |
| Phase 1 Representation Check | 완료 | IQ `2x2048`, AP `2x2048`, STFT `1x128x31`, sample NaN/Inf 없음 |
| Phase 2 Autoencoder | 구현 및 smoke 완료 | IQ/AP/STFT AE forward/backward/checkpoint 저장 확인, full 30 epoch 학습은 다음 단계 |

Phase 1에서 생성된 sanity figure는 다음 로컬 경로에 저장된다.

```text
outputs/figures/example_iq.png
outputs/figures/example_ap.png
outputs/figures/example_stft.png
outputs/tables/representation_examples.csv
```

AP representation은 현재 기본값 `phase_unwrap: true`를 사용한다. 일부 window에서 unwrap phase가 크게 누적되므로, 메인 실험 이후 `phase_unwrap: false`는 ablation 후보로 둔다.

Phase 2 smoke command:

```bash
python rf_multiview_relation/scripts/02_train_autoencoders.py --config rf_multiview_relation/configs/default.yaml --views iq ap stft --epochs 1 --batch-size 64 --max-train-files 1 --max-calibration-files 1 --run-name smoke
```

Smoke 결과에서 AP reconstruction loss가 IQ/STFT보다 매우 크게 나타났다. 이는 unwrap phase scale이 amplitude보다 훨씬 커지는 영향으로 보이며, 메인 학습은 명세대로 유지하되 이후 ablation에서 `phase_unwrap: false` 또는 AP channel scaling을 비교한다.

## 연구 목적

이 저장소는 RF IQ 원본 신호에서 여러 representation을 만들고, 정상 RF와 unseen/anomaly RF 사이에서 representation 간 관계성이 달라지는지 확인하기 위한 실험 기록이다.

현재 핵심 질문은 다음 두 가지다.

1. 정상 RF, 즉 Tx1에서 서로 다른 representation 사이에 일정한 관계가 존재하는가?
2. Tx2-Tx8 또는 Oracle external RF가 들어오면 Tx1 relation distribution과의 차이로 구분할 수 있는가?

## 평가 원칙

- Train/calibration에는 Tx1만 사용한다.
- Tx2-Tx8은 fitting, feature 선택, threshold 결정, score normalization에 사용하지 않는다.
- Oracle SigMF는 external anomaly evaluation에만 사용한다.
- Tx1 holdout은 normal test로만 사용한다.
- 최종 평가는 file-level로만 본다.
- 이 결과는 unseen-transmitter detection 실험이며, receiver/channel/distance robustness로 해석하지 않는다.
- 실험 로그, raw data, checkpoint, feature cache, `.npy`, `.npz`는 GitHub에 올리지 않는다.

## 데이터 상태

데이터 위치:

```text
data/
```

현재 파일 수:

| Device | Count |
| --- | ---: |
| Tx1 | 500 |
| Tx2 | 500 |
| Tx3 | 500 |
| Tx4 | 500 |
| Tx5 | 500 |
| Tx6 | 500 |
| Tx7 | 500 |
| Tx8 | 500 |

Tx1에서 누락되었던 파일은 추가되었다.

```text
RFF_Tx_ANTSDR_1_Boot_01_20265608_020422_iter386.mat
```

## GitHub 관리 정책

GitHub에는 다음만 올린다.

- README
- 실험 스크립트
- split manifest
- 작은 CSV 결과표

GitHub에는 다음을 올리지 않는다.

- raw RF data
- feature cache
- runtime log
- checkpoint
- local path가 들어간 run config
- `.npy`, `.npz`, `.pt`, `.pth`

## 실험 루트

작업 루트:

```text
C:\Users\Beomm\Desktop\project\모델 관련 자료\project
```

실험 코드:

```text
code\2nd\cross_view_relation
```

사용 Python:

```text
code\2nd\.venv\Scripts\python.exe
```

## 현재 실험

현재 메인 실험은 `rf_multiview_relation/`에서 새로 진행한다. 아래 내용은 2026-09-04 이전에 수행한 Tx1-Tx4 known-normal open-set relation 실험 기록이다.

이번 실험은 Tx1-Tx4를 known-normal로 두고, Tx5-Tx8을 학습에서 보지 않은 unseen/anomaly 송신기로 두는 open-set relation 실험이다.

스크립트:

```text
code\2nd\cross_view_relation\run_known4_relation_generalization_experiment.py
```

결과 위치:

```text
code\2nd\results\cross_view_relation\known4_relation_generalization_seed42_w16
```

Feature cache는 이전 IQ feature 추출 cache를 재사용했지만, cache 파일 자체는 GitHub에 올리지 않는다.

```text
code\2nd\artifacts\cross_view_relation\iq_feature_relation_full_seed42_w16.npz
```

## 실험 설정

| Item | Value |
| --- | --- |
| Known train | Tx1-Tx4, 400 files per Tx, 1600 total |
| Known test | Tx1-Tx4, 100 files per Tx, 400 total |
| Unknown/anomaly test | Tx5-Tx8, 500 files per Tx, 2000 total |
| Split seed | 42 |
| Windows per file | 16 |
| Window size / stride | 2048 / 1024 |
| Normalization | power |
| Thresholds | known train p90, p95, p97 |

## 사용한 Representation

모든 representation은 IQ 원본 신호에서 파생했다.

| Representation | 설명 |
| --- | --- |
| IQ | I/Q 통계와 power 통계 |
| AP | amplitude, phase-diff, circular phase 통계 |
| FFT | FFT log magnitude와 spectral shape |
| STFT | STFT log magnitude의 time-frequency 요약 |
| Cepstral | FFT log magnitude에 DCT 적용 |
| Cyclostationary proxy | shifted spectrum correlation 기반 경량 spectral-correlation proxy |
| HOS / cumulants | higher-order statistics와 cumulant 요약 |
| Bispectrum proxy | reduced-grid bispectrum magnitude proxy |

Cyclostationary와 Bispectrum은 첫 연구 가능성 확인을 위한 경량 proxy이다. full spectral correlation 또는 dense bispectrum estimator는 아직 사용하지 않았다.

## 관계성 진단 방법

Tx1-Tx4 known train feature만 사용해서 representation 간 관계를 진단했다.

- Linear CKA
- split-half CCA
- known Tx별 CKA/CCA 안정성
- representation pair relation feature의 one-class residual score

Relation score는 각 representation을 known train 기준으로 standardize/PCA projection한 뒤, 두 representation 사이의 `abs diff`, `product`, `1 - cosine` 관계 feature를 만들고, 그 relation feature가 known train 관계 분포에서 얼마나 벗어나는지 측정한다. PCA residual score는 test batch 평균이 아니라 known train center를 고정해서 계산한다.

## 정상 관계성 진단 결과

Pooled Tx1-Tx4 known train에서 CKA가 낮은 pair:

| Pair | Linear CKA | Split CCA mean5 |
| --- | ---: | ---: |
| IQ / Bispectrum | 0.254903 | 0.440773 |
| IQ / Cyclostationary | 0.281706 | 0.735932 |
| IQ / HOS | 0.286684 | 0.925653 |
| IQ / STFT | 0.300900 | 0.622141 |
| HOS / Bispectrum | 0.315463 | 0.433688 |

Pooled Tx1-Tx4 known train에서 CKA가 높은 pair:

| Pair | Linear CKA | Split CCA mean5 |
| --- | ---: | ---: |
| STFT / Cepstral | 0.850683 | 0.884585 |
| FFT / STFT | 0.821949 | 0.927842 |
| AP / Cyclostationary | 0.797802 | 0.905448 |
| FFT / Cepstral | 0.775935 | 0.953439 |
| AP / Cepstral | 0.763458 | 0.868737 |

Known Tx별 CKA가 안정적인 pair:

| Pair | Pooled CKA | Known Tx CKA mean | Known Tx CKA std |
| --- | ---: | ---: | ---: |
| STFT / Cepstral | 0.850683 | 0.855232 | 0.008113 |
| FFT / Cyclostationary | 0.530808 | 0.536714 | 0.011484 |
| FFT / Bispectrum | 0.550255 | 0.556473 | 0.012647 |
| AP / FFT | 0.640975 | 0.645511 | 0.012778 |
| IQ / Cyclostationary | 0.281706 | 0.290162 | 0.013366 |

이 결과만 보면 정상 Tx1-Tx4 안에서 representation 간 관계는 존재한다. 특히 STFT/Cepstral, FFT/STFT, AP/Cyclostationary처럼 높은 CKA pair는 여러 known Tx에서 비교적 안정적으로 반복된다.

## Single Representation 결과

상위 single representation score:

| Rank | Representation | Method | AUC | p90 F1 | p90 FP / 400 | p90 FN / 2000 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | FFT | zdist | 0.574055 | 0.205353 | 45 | 1766 |
| 2 | AP | PCA residual | 0.555504 | 0.271072 | 41 | 1680 |
| 3 | AP | shrinkage Mahalanobis | 0.532104 | 0.214192 | 51 | 1754 |
| 4 | FFT | PCA residual | 0.527154 | 0.223958 | 46 | 1742 |
| 5 | Bispectrum proxy | zdist | 0.525696 | 0.209007 | 48 | 1761 |

## Relation Pair 결과

상위 relation score:

| Rank | Pair | Method | AUC | p90 F1 | p90 FP / 400 | p90 FN / 2000 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | FFT / Bispectrum | relation shrinkage Mahalanobis | 0.530048 | 0.234738 | 53 | 1727 |
| 2 | Cepstral / Bispectrum | relation PCA residual | 0.528199 | 0.228002 | 44 | 1737 |
| 3 | FFT / Bispectrum | relation zdist | 0.528170 | 0.205689 | 50 | 1765 |
| 4 | FFT / Bispectrum | relation PCA residual | 0.522836 | 0.243278 | 58 | 1715 |
| 5 | Cepstral / Bispectrum | relation shrinkage Mahalanobis | 0.522575 | 0.204655 | 44 | 1767 |

## Best Relation Pair 세부 결과

Best relation score는 FFT / Bispectrum relation shrinkage Mahalanobis이다.

p90 threshold에서 known Tx별 false reject:

| Known Tx | False reject / 100 |
| --- | ---: |
| Tx1 | 15 |
| Tx2 | 7 |
| Tx3 | 17 |
| Tx4 | 14 |

p90 threshold에서 unknown Tx별 탐지:

| Unknown Tx | AUC | F1 | Recall | FN / 500 |
| --- | ---: | ---: | ---: | ---: |
| Tx5 | 0.493865 | 0.180921 | 0.110 | 445 |
| Tx6 | 0.541470 | 0.227564 | 0.142 | 429 |
| Tx7 | 0.564740 | 0.250000 | 0.158 | 421 |
| Tx8 | 0.520115 | 0.219002 | 0.136 | 432 |

## Fusion 결과

상위 fusion score:

| Fusion | AUC | p90 F1 | p90 FP / 400 | p90 FN / 2000 |
| --- | ---: | ---: | ---: | ---: |
| single PCA rank mean | 0.522226 | 0.226350 | 53 | 1738 |
| low-CKA relation PCA rank mean | 0.497518 | 0.214099 | 52 | 1754 |
| all relation PCA rank mean | 0.496154 | 0.211806 | 60 | 1756 |
| low-CKA relation Mahalanobis rank mean | 0.490196 | 0.195423 | 50 | 1778 |
| high-CKA relation PCA rank mean | 0.483333 | 0.203493 | 57 | 1767 |

## 해석

핵심 질문 1에 대한 답:

정상 RF Tx1-Tx4에서 representation 간 관계는 관찰된다. 높은 CKA pair와 낮은 Tx별 CKA 표준편차를 보면, 일부 관계는 특정 Tx 하나에만 생기는 우연이라기보다 known-normal 그룹 안에서 반복되는 구조로 볼 수 있다.

핵심 질문 2에 대한 답:

현재 feature와 residual scoring 방식만으로는 Tx5-Tx8 anomaly를 강하게 구분하지 못했다. Best relation AUC는 0.530048이고, best single representation인 FFT zdist AUC 0.574055보다 낮다. Fusion도 최고 AUC 0.522226 수준이라 relation score 조합이 성능을 끌어올리지 못했다.

현재 결론:

정상 representation relation은 존재하지만, Tx1-Tx4를 모두 normal로 묶으면 그 관계가 Tx5-Tx8에서도 크게 깨지지 않는다. 즉 현재 relation feature는 anomaly-specific 차이보다 RF 신호의 공통 구조를 더 많이 잡고 있을 가능성이 크다.

## 다음 방향

이번 결과 기준으로는 다음 보강이 필요하다.

1. CFO representation을 명시적으로 추가한다.
2. Tx1-Tx4를 하나의 pooled normal로만 보지 말고, Tx별 relation template 또는 mixture relation model을 둔다.
3. Relation residual 대신 Tx identity를 쓰지 않는 self-supervised consistency loss를 검토한다.
4. Cyclostationary proxy와 Bispectrum proxy를 full spectral correlation, bicoherence 계열로 확장한다.
5. McAFF의 IQ/CFO/FFT/STFT multi-channel 아이디어는 가져오되, supervised classifier가 아니라 unseen rejection 목적의 relation/consistency score로 바꾼다.
