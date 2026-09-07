# RF Multi-View Relation One-Class Anomaly Detection

## 연구 기준

본 저장소의 메인 연구는 다음 명세를 기준으로 한다.

```text
Multi-View Representation 관계를 활용한 RF 송신 장치 이상 탐지
```

목표는 Tx1 정상 데이터만 이용해 IQ/AP/STFT representation의 absolute latent feature와 explicit relation feature를 모델링하고, Tx2-Tx8 및 Oracle SigMF를 unseen anomaly로 평가하는 것이다.

중심 비교:

```text
Concat
vs
CCA Relation
vs
Concat + Relation
```

핵심 질문:

1. Tx1에서 multi-view relation distribution이 안정적인가?
2. Tx2-Tx8 및 Oracle은 Tx1 relation distribution에서 유의하게 이탈하는가?
3. Relation score는 absolute latent score에 없는 complementary anomaly information을 제공하는가?

## One-Class 원칙

- 모델 fitting에는 Tx1만 사용한다.
- Tx2-Tx8은 최종 anomaly evaluation에만 사용한다.
- Oracle SigMF는 external anomaly evaluation에만 사용한다.
- Encoder, CCA, scaler, covariance, threshold, alpha 선택에는 Tx2-Tx8/Oracle을 사용하지 않는다.
- Train/test split은 반드시 file level로 수행한다.

## 프로젝트 구조

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
├── utils/
└── requirements.txt
```

Raw RF dataset은 로컬 `data/`에 둔다. `data/`, `outputs/`, checkpoint, latent cache, runtime log는 GitHub에 올리지 않는다.

## 현재 완료 단계

| Phase | 상태 | 핵심 확인 |
| --- | --- | --- |
| Repository cleanup | 완료 | 새 명세와 무관한 legacy 코드/문서/출력물 제거, raw `data/` 보존 |
| Phase 0 Dataset Audit | 완료 | Tx1-Tx8 각 500 files, Tx1 split 320/80/100, Oracle 표준 SigMF 128 files |
| Phase 1 Representation Check | 완료 | IQ `2x2048`, AP `2x2048`, STFT `1x128x31`, sample NaN/Inf 없음 |
| Phase 2 Autoencoder | 완료 | Tx1 train only GPU 학습, IQ/AP/STFT checkpoint 저장 |
| Phase 3 Latent Extraction | 완료 | Tx1 split, Tx2-Tx8, Oracle paired latent 저장 |
| Phase 4 Relation Analysis | 완료 | Tx1-only CCA fitting, CKA table 생성 |
| Phase 5 Relation Model | 완료 | Tx1-only covariance/threshold fitting |
| Phase 6 Evaluation | 완료 | Tx2-Tx8 closed test 및 Oracle external test 완료 |

## Seed 42 실행 결과

### 학습 설정

- Tx1 train 320 files, calibration 80 files, holdout 100 files
- Tx2-Tx8 각 500 files
- Oracle SigMF 128 files
- Window size 2048, stride 1024, max windows per file 256
- Encoders: IQ/AP/STFT autoencoder pretraining, latent dim 64
- CCA components 16
- File score aggregation: p60
- Threshold: Tx1 calibration file score p95

### Autoencoder 학습

| View | Best epoch | Best calibration loss | Early stop epoch |
| --- | ---: | ---: | ---: |
| IQ | 4 | 0.408929 | 9 |
| AP | 8 | 2718.610694 | 13 |
| STFT | 10 | 0.001738 | 15 |

AP loss가 큰 이유는 현재 main setting에서 unwrapped phase를 그대로 reconstruction하기 때문이다. 이 부분은 이후 phase scaling 또는 raw phase ablation이 필요하다.

### Relation 존재 여부

Tx1 train 기준 CCA mean canonical correlation:

| Pair | Mean corr |
| --- | ---: |
| IQ-AP | 0.3594 |
| IQ-STFT | 0.5119 |
| AP-STFT | 0.4479 |

Linear CKA에서는 Tx1의 AP-STFT 관계가 가장 뚜렷했다.

| Dataset | IQ-AP | IQ-STFT | AP-STFT |
| --- | ---: | ---: | ---: |
| Tx1 train | 0.0129 | 0.0110 | 0.2814 |
| Tx1 holdout | 0.0138 | 0.0075 | 0.3056 |
| Tx2 | 0.0112 | 0.0097 | 0.2969 |
| Tx4 | 0.0080 | 0.0096 | 0.2360 |
| Tx8 | 0.0108 | 0.0098 | 0.2563 |
| Oracle | 0.0118 | 0.0013 | 0.0314 |

현재 결과에서는 IQ가 포함된 relation보다 AP-STFT relation이 더 의미 있는 축으로 보인다.

### Closed Dataset Test

Normal은 Tx1 holdout 100 files, anomaly는 Tx2-Tx8 3500 files combined.

| Method | AUROC | AUPRC | Precision | Recall | F1 | FPR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ-only | 0.5908 | 0.9733 | 0.8861 | 0.0200 | 0.0391 | 0.0900 | 0.9100 |
| Concat | 0.6193 | 0.9736 | 0.8765 | 0.0203 | 0.0397 | 0.1000 | 0.9000 |
| Raw Relation | 0.5646 | 0.9732 | 0.9635 | 0.0603 | 0.1135 | 0.0800 | 0.9200 |
| CCA Relation, signed residual + MD | 0.6290 | 0.9760 | 0.9416 | 0.0414 | 0.0794 | 0.0900 | 0.9100 |
| CCA Compact Relation | 0.6531 | 0.9782 | 0.9483 | 0.0471 | 0.0898 | 0.0900 | 0.9100 |
| AP-STFT CCA Cosine Direct | 0.7831 | 0.9916 | 0.9950 | 0.4523 | 0.6219 | 0.0800 | 0.9200 |
| AP-STFT CCA L2 Direct | 0.6627 | 0.9831 | 0.9892 | 0.0786 | 0.1456 | 0.0300 | 0.9700 |
| AP-STFT CCA Abs Mean Direct | 0.6668 | 0.9833 | 0.9911 | 0.1900 | 0.3189 | 0.0600 | 0.9400 |
| Concat + Relation | 0.6165 | 0.9744 | 0.8947 | 0.0243 | 0.0473 | 0.1000 | 0.9000 |
| Score Fusion | 0.6303 | 0.9748 | 0.9020 | 0.0263 | 0.0511 | 0.1000 | 0.9000 |

### Device-Wise Result

`AP-STFT CCA Cosine Direct` 기준:

| Anomaly | AUROC | Recall | F1 |
| --- | ---: | ---: | ---: |
| Tx2 | 0.7337 | 0.3100 | 0.4676 |
| Tx3 | 0.7172 | 0.2440 | 0.3873 |
| Tx4 | 0.8389 | 0.6500 | 0.7803 |
| Tx5 | 0.7559 | 0.2900 | 0.4441 |
| Tx6 | 0.7194 | 0.2840 | 0.4369 |
| Tx7 | 0.8565 | 0.6520 | 0.7818 |
| Tx8 | 0.8601 | 0.7360 | 0.8402 |

Mann-Whitney U test도 모든 Tx2-Tx8에서 유의했다. Cliff's delta는 Tx2 0.4674, Tx3 0.4345, Tx4 0.6779, Tx5 0.5118, Tx6 0.4388, Tx7 0.7129, Tx8 0.7202였다.

### Oracle External Test

Normal은 Tx1 holdout 100 files, anomaly는 Oracle SigMF 128 files.

| Method | AUROC | AUPRC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Concat | 0.7599 | 0.6907 | 0.0000 | 0.0000 | 0.0000 |
| CCA Relation, signed residual + MD | 0.8588 | 0.7801 | 0.6786 | 0.1484 | 0.2436 |
| CCA Compact Relation | 0.9314 | 0.9116 | 0.9159 | 0.7656 | 0.8340 |
| AP-STFT CCA Cosine Direct | 0.9988 | 0.9991 | 0.9412 | 1.0000 | 0.9697 |
| AP-STFT CCA L2 Direct | 0.9603 | 0.9753 | 0.9730 | 0.8438 | 0.9038 |
| AP-STFT CCA Abs Mean Direct | 0.9713 | 0.9820 | 0.9520 | 0.9297 | 0.9407 |

### 현재 해석

원래 main proposal인 `3-pair signed residual + Mahalanobis`만 고집하면 성능이 약하다. 하지만 AP-STFT CCA 관계, 특히 CCA cosine을 방향성 있는 score로 직접 사용하는 경우 Tx2-Tx8과 Oracle에서 뚜렷한 anomaly signal이 확인된다.

따라서 다음 논문 방향은 다음처럼 재정립하는 것이 좋다.

```text
전체 multi-view relation을 무작정 결합하기보다,
Tx1에서 안정적으로 형성되는 view-pair relation을 선별하고,
CCA-aligned AP-STFT relation shift를 one-class anomaly score로 사용한다.
```

## 실행 명령

Repository root 기준:

```bash
python rf_multiview_relation/scripts/00_audit_dataset.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/01_verify_representations.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/02_train_autoencoders.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/03_extract_latents.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/04_analyze_relations.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/05_fit_relation_model.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/06_evaluate.py --config rf_multiview_relation/configs/default.yaml
```

최종 목표:

```bash
python rf_multiview_relation/scripts/08_run_all.py --config rf_multiview_relation/configs/default.yaml
```

`rf_multiview_relation/` 내부 기준:

```bash
python scripts/08_run_all.py --config configs/default.yaml
```

## 로컬 산출물

현재까지 생성된 산출물은 모두 `outputs/` 아래에 저장된다.

```text
outputs/dataset_audit.json
outputs/tables/dataset_audit_devices.csv
outputs/tables/dataset_audit_oracle.csv
outputs/tables/representation_examples.csv
outputs/tables/cca_results.csv
outputs/tables/cka_results.csv
outputs/tables/main_results.csv
outputs/tables/device_results.csv
outputs/scores/file_scores.csv
outputs/figures/example_iq.png
outputs/figures/example_ap.png
outputs/figures/example_stft.png
outputs/figures/score_distribution.png
outputs/figures/roc_curve.png
outputs/figures/relation_heatmap.png
outputs/figures/oracle_distribution.png
```

이 파일들은 재현 가능한 코드로 다시 만들 수 있으므로 GitHub에는 올리지 않는다.
