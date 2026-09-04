# RF Multi-View Relation

This project implements Tx1-only one-class RF transmitter anomaly detection
using relationships between IQ, amplitude-phase, and STFT representations.

The first runnable phase is dataset audit:

```bash
python rf_multiview_relation/scripts/00_audit_dataset.py --config rf_multiview_relation/configs/default.yaml
```

Representation sanity extraction:

```bash
python rf_multiview_relation/scripts/01_extract_views.py --config rf_multiview_relation/configs/default.yaml
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
```
