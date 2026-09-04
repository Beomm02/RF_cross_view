from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


RGB_WHITE = np.array([255, 255, 255], dtype=np.uint8)
RGB_GRAY = np.array([220, 224, 229], dtype=np.uint8)
RGB_BLUE = np.array([34, 97, 214], dtype=np.uint8)
RGB_RED = np.array([217, 74, 74], dtype=np.uint8)
RGB_GREEN = np.array([45, 160, 100], dtype=np.uint8)
RGB_ORANGE = np.array([228, 133, 45], dtype=np.uint8)


def save_iq_examples(examples: list[np.ndarray], path: str | Path) -> None:
    image = _line_grid(
        examples,
        channels=(0, 1),
        colors=(RGB_BLUE, RGB_RED),
        height_per_example=72,
        width=1000,
    )
    write_png(path, image)


def save_ap_examples(examples: list[np.ndarray], path: str | Path) -> None:
    image = _line_grid(
        examples,
        channels=(0, 1),
        colors=(RGB_GREEN, RGB_ORANGE),
        height_per_example=72,
        width=1000,
    )
    write_png(path, image)


def save_stft_examples(examples: list[np.ndarray], path: str | Path) -> None:
    tiles = [_heatmap_tile(np.asarray(example)[0], scale_y=2, scale_x=8) for example in examples]
    image = _tile_grid(tiles, columns=2, gap=12, background=RGB_WHITE)
    write_png(path, image)


def save_loss_curves(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    views = sorted({str(row["view"]) for row in rows})
    height = max(180, 140 * len(views))
    width = 900
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = RGB_WHITE
    panel_h = height // len(views)
    for panel_idx, view in enumerate(views):
        top = panel_idx * panel_h
        bottom = min(height - 1, top + panel_h - 1)
        view_rows = [row for row in rows if str(row["view"]) == view]
        train = np.asarray([float(row["train_loss"]) for row in view_rows], dtype=np.float64)
        val = np.asarray([float(row["calibration_loss"]) for row in view_rows], dtype=np.float64)
        values = np.concatenate([train, val])
        y_top = top + 18
        y_bottom = bottom - 18
        x_left = 36
        x_right = width - 24
        _draw_rect(image, x_left, y_top, x_right, y_bottom, RGB_GRAY)
        _draw_horizontal(image, y_bottom, x_left, x_right, RGB_GRAY)
        x_values = np.linspace(x_left, x_right, num=max(len(train), 1)).astype(np.int32)
        if len(train) == 1:
            x_values = np.asarray([(x_left + x_right) // 2], dtype=np.int32)
        train_y = _values_to_y(train, values, y_top + 4, y_bottom - 4).astype(np.int32)
        val_y = _values_to_y(val, values, y_top + 4, y_bottom - 4).astype(np.int32)
        _draw_polyline_or_point(image, x_values, train_y, RGB_BLUE)
        _draw_polyline_or_point(image, x_values, val_y, RGB_RED)
    write_png(path, image)


def write_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(image, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    height, width, _ = img.shape
    raw_rows = [b"\x00" + img[row].tobytes() for row in range(height)]
    raw = b"".join(raw_rows)
    chunks = [
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw, level=6)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _line_grid(
    examples: list[np.ndarray],
    channels: tuple[int, int],
    colors: tuple[np.ndarray, np.ndarray],
    height_per_example: int,
    width: int,
) -> np.ndarray:
    margin_x = 24
    gap_y = 8
    height = len(examples) * height_per_example + max(0, len(examples) - 1) * gap_y
    image = np.empty((max(height, 1), width, 3), dtype=np.uint8)
    image[:] = RGB_WHITE
    plot_width = width - 2 * margin_x

    for row_idx, example in enumerate(examples):
        top = row_idx * (height_per_example + gap_y)
        bottom = top + height_per_example - 1
        _draw_horizontal(image, top + height_per_example // 2, margin_x, width - margin_x - 1, RGB_GRAY)
        _draw_rect(image, margin_x, top, width - margin_x - 1, bottom, RGB_GRAY)
        for channel, color in zip(channels, colors):
            y_values = _series_to_y(np.asarray(example[channel]), top + 6, bottom - 6)
            x_values = np.linspace(margin_x, margin_x + plot_width - 1, num=y_values.size)
            _draw_polyline(image, x_values.astype(np.int32), y_values.astype(np.int32), color)
    return image


def _series_to_y(series: np.ndarray, top: int, bottom: int) -> np.ndarray:
    data = np.asarray(series, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.quantile(data, 0.01))
    high = float(np.quantile(data, 0.99))
    if high <= low:
        high = low + 1.0
    clipped = np.clip(data, low, high)
    normalized = (clipped - low) / (high - low)
    return bottom - normalized * (bottom - top)


def _values_to_y(values: np.ndarray, reference: np.ndarray, top: int, bottom: int) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    reference = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(reference))
    high = float(np.max(reference))
    if high <= low:
        high = low + 1.0
    normalized = (np.asarray(values, dtype=np.float64) - low) / (high - low)
    return bottom - np.clip(normalized, 0.0, 1.0) * (bottom - top)


def _draw_horizontal(image: np.ndarray, y: int, x0: int, x1: int, color: np.ndarray) -> None:
    if 0 <= y < image.shape[0]:
        image[y, max(0, x0) : min(image.shape[1], x1 + 1)] = color


def _draw_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray) -> None:
    _draw_horizontal(image, y0, x0, x1, color)
    _draw_horizontal(image, y1, x0, x1, color)
    if 0 <= x0 < image.shape[1]:
        image[max(0, y0) : min(image.shape[0], y1 + 1), x0] = color
    if 0 <= x1 < image.shape[1]:
        image[max(0, y0) : min(image.shape[0], y1 + 1), x1] = color


def _draw_polyline(image: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, color: np.ndarray) -> None:
    for idx in range(len(x_values) - 1):
        _draw_line(image, int(x_values[idx]), int(y_values[idx]), int(x_values[idx + 1]), int(y_values[idx + 1]), color)


def _draw_polyline_or_point(image: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, color: np.ndarray) -> None:
    if len(x_values) == 1:
        x = int(x_values[0])
        y = int(y_values[0])
        image[max(0, y - 2) : min(image.shape[0], y + 3), max(0, x - 2) : min(image.shape[1], x + 3)] = color
        return
    _draw_polyline(image, x_values, y_values, color)


def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            image[y, x] = color
        if x == x1 and y == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x += sx
        if err2 <= dx:
            err += dx
            y += sy


def _heatmap_tile(array: np.ndarray, scale_y: int, scale_x: int) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.quantile(data, 0.01))
    high = float(np.quantile(data, 0.99))
    if high <= low:
        high = low + 1.0
    normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
    rgb = _colormap(normalized)
    rgb = np.repeat(np.repeat(rgb, int(scale_y), axis=0), int(scale_x), axis=1)
    return rgb.astype(np.uint8)


def _colormap(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [35, 30, 70],
            [45, 95, 150],
            [45, 160, 120],
            [230, 185, 65],
            [250, 245, 190],
        ],
        dtype=np.float64,
    )
    scaled = values * (len(stops) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.clip(left + 1, 0, len(stops) - 1)
    alpha = (scaled - left)[..., np.newaxis]
    return stops[left] * (1.0 - alpha) + stops[right] * alpha


def _tile_grid(
    tiles: list[np.ndarray],
    columns: int,
    gap: int,
    background: np.ndarray,
) -> np.ndarray:
    if not tiles:
        return np.full((1, 1, 3), background, dtype=np.uint8)
    columns = max(1, int(columns))
    rows = int(np.ceil(len(tiles) / columns))
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    height = rows * tile_h + (rows + 1) * gap
    width = columns * tile_w + (columns + 1) * gap
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = background
    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        top = gap + row * (tile_h + gap)
        left = gap + col * (tile_w + gap)
        image[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
    return image
