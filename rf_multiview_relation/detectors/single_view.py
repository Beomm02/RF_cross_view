from __future__ import annotations

import numpy as np


def single_view_features(latents: dict[str, np.ndarray], view_name: str) -> np.ndarray:
    if view_name not in latents:
        raise KeyError(f"missing latent view: {view_name}")
    return np.asarray(latents[view_name], dtype=np.float64)
