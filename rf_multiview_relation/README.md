# RF Multi-View Relation

This project implements Tx1-only one-class RF transmitter anomaly detection
using relationships between IQ, amplitude-phase, and STFT representations.

The first runnable phase is dataset audit:

```bash
python rf_multiview_relation/scripts/00_audit_dataset.py --config rf_multiview_relation/configs/default.yaml
```

Representation sanity extraction:

```bash
python rf_multiview_relation/scripts/01_verify_representations.py --config rf_multiview_relation/configs/default.yaml
```

Tx1-only autoencoder pretraining:

```bash
python rf_multiview_relation/scripts/02_train_autoencoders.py --config rf_multiview_relation/configs/default.yaml
```

Latent extraction, relation analysis, fitting, and evaluation:

```bash
python rf_multiview_relation/scripts/03_extract_latents.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/04_analyze_relations.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/05_fit_relation_model.py --config rf_multiview_relation/configs/default.yaml
python rf_multiview_relation/scripts/06_evaluate.py --config rf_multiview_relation/configs/default.yaml
```

Generated experiment outputs are written under `outputs/`, which is ignored by
Git to keep raw data, logs, checkpoints, latent arrays, and local result files
out of GitHub.

Current local sanity outputs:

```text
outputs/figures/example_iq.png
outputs/figures/example_ap.png
outputs/figures/example_stft.png
outputs/tables/representation_examples.csv
outputs/tables/cca_results.csv
outputs/tables/cka_results.csv
outputs/tables/main_results.csv
outputs/tables/device_results.csv
outputs/scores/file_scores.csv
```

Validated smoke command:

```bash
python rf_multiview_relation/scripts/02_train_autoencoders.py --config rf_multiview_relation/configs/default.yaml --views iq ap stft --epochs 1 --batch-size 64 --max-train-files 1 --max-calibration-files 1 --run-name smoke
```

Canonical final target from inside this directory:

```bash
python scripts/08_run_all.py --config configs/default.yaml
```

The central experimental comparison is:

```text
Concat vs CCA Relation vs Concat + Relation
```
