import os
import numpy as np
from scipy import signal
from scipy.io import loadmat


def load_iq_from_mat(mat_path: str, key: str = "rxData") -> np.ndarray:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    mat = loadmat(mat_path)
    if key not in mat:
        raise KeyError(f"'{key}' not found in {mat_path}")

    rx = np.squeeze(mat[key])

    if not np.iscomplexobj(rx):
        raise ValueError(f"{mat_path} -> '{key}' is not complex data")

    if rx.ndim != 1:
        rx = rx.reshape(-1)

    iq = np.stack([rx.real, rx.imag], axis=1).astype(np.float32)
    return iq


def validate_iq(iq: np.ndarray, min_len: int) -> None:
    if not isinstance(iq, np.ndarray):
        raise TypeError("iq must be numpy.ndarray")
    if iq.ndim != 2 or iq.shape[1] != 2:
        raise ValueError("iq must have shape (N, 2)")
    if np.isnan(iq).any() or np.isinf(iq).any():
        raise ValueError("iq contains NaN/Inf")
    if iq.shape[0] < min_len:
        raise ValueError(f"iq length is smaller than required minimum ({min_len})")


def normalize_iq(iq: np.ndarray, mode: str = "power") -> np.ndarray:
    x = iq.astype(np.float32).copy()

    if mode == "none":
        pass
    elif mode == "zscore":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - mean) / std
    elif mode == "minmax":
        x_min = x.min(axis=0, keepdims=True)
        x_max = x.max(axis=0, keepdims=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
    elif mode == "power":
        power = np.mean(np.sum(x**2, axis=1), keepdims=True)
        x = x / np.sqrt(power + 1e-8)
    elif mode == "power_dc":
        x = x - x.mean(axis=0, keepdims=True)
        power = np.mean(np.sum(x**2, axis=1), keepdims=True)
        x = x / np.sqrt(power + 1e-8)
    elif mode == "dc_only":
        mean = x.mean(axis=0, keepdims=True)
        x = x - mean
    else:
        raise ValueError("mode must be 'none', 'zscore', 'minmax', 'power', 'power_dc', or 'dc_only'")

    return x


def make_windows(iq: np.ndarray, window_size: int = 2048, stride: int = 1024) -> np.ndarray:
    n = iq.shape[0]
    windows = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(iq[start:end])

    if not windows:
        raise ValueError("No windows generated")

    return np.stack(windows, axis=0).astype(np.float32)


def build_iq_view(iq_window: np.ndarray) -> np.ndarray:
    return np.stack([iq_window[:, 0], iq_window[:, 1]], axis=0).astype(np.float32)


def build_ap_view(iq_window: np.ndarray) -> np.ndarray:
    I = iq_window[:, 0]
    Q = iq_window[:, 1]

    amp = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    phase_unwrap = np.unwrap(phase)
    phase_diff = np.diff(phase_unwrap, prepend=phase_unwrap[0])

    return np.stack([amp, phase, phase_diff], axis=0).astype(np.float32)


def _zscore_sample(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(keepdims=True)) / (x.std(keepdims=True) + eps)


def build_freq_view(
    iq_window: np.ndarray,
    use_log_mag: bool = True,
    mode: str = "fft",
    stft_nperseg: int = 128,
    stft_noverlap: int = 96,
    stft_window: str = "hann",
    stft_zscore: bool = True,
) -> np.ndarray:
    """
    Build a frequency representation from one IQ window.

    FFT mode preserves the original global spectrum summary with shape
    (1, window_size). STFT mode preserves local time-frequency variation
    with shape (freq_bins, time_frames), so freq_bins can be used as the
    channel dimension for the existing Conv1d frequency encoder.
    """
    complex_signal = iq_window[:, 0] + 1j * iq_window[:, 1]

    if mode == "fft":
        fft_val = np.fft.fftshift(np.fft.fft(complex_signal))
        mag = np.abs(fft_val)

        if use_log_mag:
            mag = np.log1p(mag)

        return np.nan_to_num(mag[np.newaxis, :], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if mode != "stft":
        raise ValueError("mode must be 'fft' or 'stft'")

    nperseg = min(int(stft_nperseg), int(complex_signal.shape[0]))
    noverlap = min(int(stft_noverlap), max(0, nperseg - 1))
    _, _, zxx = signal.stft(
        complex_signal,
        fs=1.0,
        window=stft_window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    mag = np.fft.fftshift(np.abs(zxx), axes=0)

    if use_log_mag:
        mag = np.log1p(mag)
    if stft_zscore:
        mag = _zscore_sample(mag)

    return np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
