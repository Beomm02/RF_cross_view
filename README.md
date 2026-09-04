# RF Cross-View Feature Relation Study

## 연구 목적

이 저장소는 RF IQ 원본 신호에서 여러 표현(feature representation)을 만들고, 그 표현들 사이의 관계성이 Tx1 정상 송신기와 Tx2-Tx8 타 송신기 사이에서 달라지는지 확인하기 위한 실험 기록이다.

핵심 질문은 다음과 같다.

Tx1 정상 데이터에서 안정적인 feature 간 관계가 존재하고, Tx2-Tx8에서는 그 관계가 깨지는가?

## 평가 원칙

- Train/calibration에는 Tx1 train-normal만 사용한다.
- Tx1 holdout과 Tx2-Tx8은 fitting, feature 선택, threshold 결정, score normalization에 사용하지 않는다.
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

## 단일 파이프라인 실험

이번 정리에서는 이전 개별 실험 결과를 제거하고, 하나의 파이프라인으로 다시 수행했다.

스크립트:

```text
code\2nd\cross_view_relation\run_iq_feature_relation_experiment.py
```

이 스크립트는 다음을 한 번에 수행한다.

1. Tx1 500개에서 seed 42 기준 fresh 400/100 split 생성
2. Tx2-Tx8 전체 anomaly manifest 생성
3. IQ 원본에서 여러 representation feature 추출
4. Tx1 train-normal 기준 CKA/CCA relation 진단
5. single representation one-class scoring
6. 모든 representation pair에 대한 relation scoring
7. train-normal empirical rank fusion

## 실험 설정

| Item | Value |
| --- | --- |
| Tx1 train | 400 files |
| Tx1 holdout | 100 files |
| Tx2-Tx8 anomaly | 500 files per Tx, 3500 total |
| windows per file | 16 |
| window size / stride | 2048 / 1024 |
| normalization | power |
| fitting data | Tx1 train only |
| thresholds | Tx1 train p90, p95, p97 |

## 사용한 Feature Representation

모든 feature는 IQ 원본 신호에서 파생했다.

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

Tx1 train-normal feature만 사용해서 representation 간 관계를 진단했다.

- Linear CKA
- split-half CCA

이 값들은 anomaly detection 성능 그 자체가 아니라, representation들이 얼마나 중복되거나 상보적인지 보는 진단값이다.

## CKA / CCA 결과

낮은 CKA pair:

| Pair | Linear CKA | Split CCA mean5 |
| --- | ---: | ---: |
| IQ / Bispectrum | 0.251270 | 0.417163 |
| IQ / HOS | 0.272731 | 0.792399 |
| IQ / STFT | 0.283481 | 0.454875 |
| IQ / Cyclostationary | 0.286236 | 0.450929 |
| HOS / Bispectrum | 0.294336 | 0.449063 |

높은 CKA pair:

| Pair | Linear CKA | Split CCA mean5 |
| --- | ---: | ---: |
| STFT / Cepstral | 0.863303 | 0.885524 |
| FFT / STFT | 0.803549 | 0.935830 |
| AP / Cyclostationary | 0.782317 | 0.834140 |
| FFT / Cepstral | 0.765221 | 0.946848 |
| AP / Cepstral | 0.736148 | 0.856276 |

## Single Representation 결과

상위 single representation score:

| Rank | Representation | Method | AUC | p90 F1 | p90 FP / 100 | p90 FN / 3500 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Cyclostationary proxy | PCA residual | 0.580211 | 0.366426 | 13 | 2712 |
| 2 | FFT | PCA residual | 0.579683 | 0.427518 | 20 | 2543 |
| 3 | Cepstral | shrinkage Mahalanobis | 0.568717 | 0.379922 | 18 | 2675 |
| 4 | FFT | shrinkage Mahalanobis | 0.566163 | 0.363721 | 18 | 2718 |

## Relation Pair 결과

상위 relation score:

| Rank | Pair | Method | AUC | p90 F1 | p90 FP / 100 | p90 FN / 3500 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | AP / HOS | relation PCA residual | 0.651763 | 0.477302 | 22 | 2396 |
| 2 | Cepstral / Bispectrum | relation PCA residual | 0.611949 | 0.375463 | 9 | 2689 |
| 3 | FFT / Cyclostationary | relation PCA residual | 0.583309 | 0.394161 | 20 | 2636 |
| 4 | Cepstral / Bispectrum | relation shrinkage Mahalanobis | 0.582231 | 0.388991 | 12 | 2652 |
| 5 | IQ / HOS | relation PCA residual | 0.578534 | 0.338208 | 23 | 2783 |

## Best Relation Pair Per-Device

Best overall relation score: AP / HOS relation PCA residual.

| Device | AUC | p90 F1 | p90 FN / 500 |
| --- | ---: | ---: | ---: |
| Tx2 | 0.663240 | 0.363636 | 384 |
| Tx3 | 0.541640 | 0.307942 | 405 |
| Tx4 | 0.658440 | 0.307942 | 405 |
| Tx5 | 0.473360 | 0.268657 | 419 |
| Tx6 | 0.802300 | 0.706320 | 215 |
| Tx7 | 0.614420 | 0.342857 | 392 |
| Tx8 | 0.808940 | 0.765957 | 176 |

## Fusion 결과

상위 fusion score:

| Fusion | AUC | p90 F1 | p90 FP / 100 | p90 FN / 3500 |
| --- | ---: | ---: | ---: | ---: |
| single PCA rank mean | 0.558096 | 0.417957 | 24 | 2569 |
| high-CKA relation PCA rank mean | 0.542416 | 0.378229 | 16 | 2680 |
| all relation PCA rank mean | 0.537249 | 0.386712 | 21 | 2656 |
| low-CKA relation Mahalanobis rank mean | 0.499743 | 0.275270 | 15 | 2939 |
| low-CKA relation PCA rank mean | 0.485277 | 0.328188 | 20 | 2809 |

## 해석

이번 실험은 "feature 간 관계성"이 실제로 의미가 있는지 확인하는 데 초점을 둔다.

핵심 관찰:

- Single representation 최고 AUC는 Cyclostationary proxy의 0.580211이다.
- Relation pair 최고 AUC는 AP / HOS의 0.651763이다.
- 따라서 단일 feature보다 feature 간 관계를 보는 쪽에서 더 강한 신호가 관찰됐다.
- 다만 AP / HOS는 Tx6, Tx8에서는 강하지만 Tx5에서는 AUC 0.473360으로 실패한다.
- 전체 fusion은 relation pair 최고값보다 낮았다.
- FFT, STFT, Cepstral은 CKA가 높아 서로 중복성이 강하다.
- IQ와 Bispectrum, IQ와 Cyclostationary는 CKA가 낮아 상보성 후보이지만 현재 scoring에서는 강한 성능으로 이어지지 않았다.

## 현재 결론

연구 가능성은 있다.

가장 중요한 결과는 AP / HOS relation이 single representation보다 더 높은 AUC를 보였다는 점이다. 이는 Tx1 정상에서의 amplitude/phase 구조와 higher-order statistics 사이 관계가 일부 타 송신기에서 달라질 수 있음을 시사한다.

하지만 아직 standalone detector로 주장하기에는 부족하다. Tx5에서 실패하고, 전체 recall도 낮다. 다음 단계는 AP / HOS relation을 중심으로 안정성을 높이는 것이다.

## 다음 실험 후보

- AP / HOS relation feature를 더 자세히 분해해서 어떤 항이 성능을 만드는지 확인
- Tx5 failure analysis
- full cyclostationary spectral correlation 구현
- dense bispectrum 또는 bicoherence feature 구현
- relation feature에 robust covariance / rank calibration / per-family normalization 적용
- 기존 power-tail baseline과 score-level fusion 비교
