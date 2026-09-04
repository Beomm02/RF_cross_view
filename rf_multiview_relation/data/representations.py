from __future__ import annotations

import numpy as np
from scipy import signal


EPS = 1e-8


def energy_normalize_iq_window(iq_window: np.ndarray, eps: float = EPS) -> np.ndarray:
    window = np.asarray(iq_window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != 2:
        raise ValueError("iq_window must have shape [N, 2]")
    power = float(np.mean(np.sum(window * window, axis=1)))
    return (window / np.sqrt(power + eps)).astype(np.float32)


def build_iq_view(iq_window: np.ndarray) -> np.ndarray:
    window = np.asarray(iq_window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != 2:
        raise ValueError("iq_window must have shape [N, 2]")
    view = np.stack([window[:, 0], window[:, 1]], axis=0)
    return np.nan_to_num(view, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_ap_view(iq_window: np.ndarray, phase_unwrap: bool = True) -> np.ndarray:
    window = np.asarray(iq_window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != 2:
        raise ValueError("iq_window must have shape [N, 2]")
    i_data = window[:, 0]
    q_data = window[:, 1]
    amplitude = np.sqrt(i_data * i_data + q_data * q_data)
    phase = np.arctan2(q_data, i_data)
    if phase_unwrap:
        phase = np.unwrap(phase)
    view = np.stack([amplitude, phase], axis=0)
    return np.nan_to_num(view, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_stft_view(
    iq_window: np.ndarray,
    n_fft: int = 128,
    win_length: int = 128,
    hop_length: int = 64,
    window: str = "hann",
    log_magnitude: bool = True,
) -> np.ndarray:
    iq = np.asarray(iq_window, dtype=np.float32)
    if iq.ndim != 2 or iq.shape[1] != 2:
        raise ValueError("iq_window must have shape [N, 2]")
    complex_signal = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    win_length = min(int(win_length), int(complex_signal.shape[0]))
    n_fft = max(int(n_fft), win_length)
    hop_length = int(hop_length)
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    noverlap = max(0, win_length - hop_length)
    _, _, zxx = signal.stft(
        complex_signal,
        fs=1.0,
        window=window,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    magnitude = np.fft.fftshift(np.abs(zxx), axes=0)
    if log_magnitude:
        magnitude = np.log1p(magnitude)
    view = magnitude[np.newaxis, :, :]
    return np.nan_to_num(view, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_all_views(iq_window: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    normalized = energy_normalize_iq_window(iq_window)
    stft_cfg = config["representations"]["stft"]
    return {
        "iq": build_iq_view(normalized),
        "ap": build_ap_view(normalized, phase_unwrap=bool(config["representations"]["phase_unwrap"])),
        "stft": build_stft_view(
            normalized,
            n_fft=int(stft_cfg["n_fft"]),
            win_length=int(stft_cfg["win_length"]),
            hop_length=int(stft_cfg["hop_length"]),
            window=str(stft_cfg["window"]),
            log_magnitude=bool(stft_cfg["log_magnitude"]),
        ),
    }
