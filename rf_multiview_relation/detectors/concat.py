from __future__ import annotations

import numpy as np


def concat_latents(z_iq: np.ndarray, z_ap: np.ndarray, z_stft: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(z_iq, dtype=np.float64),
            np.asarray(z_ap, dtype=np.float64),
            np.asarray(z_stft, dtype=np.float64),
        ],
        axis=1,
    )
