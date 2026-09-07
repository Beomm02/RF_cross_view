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
| Phase 2 Autoencoder | 구현 및 smoke 완료 | IQ/AP/STFT AE forward/backward/checkpoint 저장 확인 |
| Phase 3 Latent Extraction | 구현 및 minimal smoke 완료 | paired `file_id/window_id` latent 저장 확인 |
| Phase 4 Relation Analysis | 구현 및 minimal smoke 완료 | Tx1-only CCA fitting, CKA table 생성 확인 |
| Phase 5 Relation Model | 구현 및 minimal smoke 완료 | Tx1-only covariance/threshold fitting 확인 |
| Phase 6 Evaluation | 구현 및 minimal smoke 완료 | file score, metrics, figures 생성 확인 |

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
