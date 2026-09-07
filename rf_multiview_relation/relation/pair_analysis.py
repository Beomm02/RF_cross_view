from __future__ import annotations

import numpy as np

from rf_multiview_relation.relation.cka import linear_cka


PAIR_KEYS = (
    ("iq_ap", "iq", "ap"),
    ("iq_stft", "iq", "stft"),
    ("ap_stft", "ap", "stft"),
)


def cka_rows(device: str, latents: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows = []
    for pair_name, left, right in PAIR_KEYS:
        rows.append(
            {
                "device": device,
                "pair": pair_name,
                "cka": linear_cka(latents[left], latents[right]),
            }
        )
    return rows
