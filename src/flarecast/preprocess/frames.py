"""Br frame preprocessing and per-frame storage layout (decisions.md §6).

NaN -> 0, clip +-clip_gauss, divide by clip_gauss, zero-pad to a centred square, resize with area averaging, cast.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["harpnum", "time", "status", "raw_h", "raw_w", "nan_fraction", "error"]


def read_br(path: Path) -> np.ndarray:
    """Return the first 2-D image HDU of a FITS file as a native-endian float32 array."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        for hdu in hdul:
            data = hdu.data
            if data is not None and data.ndim == 2:
                return np.array(data, dtype=np.float32)  # copies and removes FITS big-endian byte order
    raise ValueError(f"{Path(path).name}: no 2-D image HDU found")


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Zero-pad a 2-D array to a centred square (aspect ratio preserved)."""
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {img.shape}")
    h, w = img.shape
    s = max(h, w)
    out = np.zeros((s, s), dtype=img.dtype)
    top, left = (s - h) // 2, (s - w) // 2
    out[top:top + h, left:left + w] = img
    return out


def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix whose rows average the input pixels covered by each output pixel (exact overlap).

    Rows sum to 1. For n_in < n_out this degenerates to pixel replication.
    """
    edges = np.linspace(0.0, n_in, n_out + 1)
    lo, hi = np.arange(n_in), np.arange(1, n_in + 1)
    overlap = np.clip(np.minimum(hi[None, :], edges[1:, None]) - np.maximum(lo[None, :], edges[:-1, None]), 0.0, None)
    return (overlap / overlap.sum(axis=1, keepdims=True)).astype(np.float32)


def area_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2-D array to (size, size) by area averaging."""
    h, w = img.shape
    return _area_weights(h, size) @ img.astype(np.float32, copy=False) @ _area_weights(w, size).T


def preprocess_br(raw: np.ndarray, size: int = 128, clip_gauss: float = 1500.0,
                  dtype: str = "float16") -> tuple[np.ndarray, float]:
    """Preprocess one Br frame. Returns (array of shape (size, size), NaN fraction of the raw frame)."""
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError(f"expected a non-empty 2-D Br array, got shape {raw.shape}")
    raw = raw.astype(np.float32, copy=False)
    nan_mask = ~np.isfinite(raw)
    nan_fraction = float(nan_mask.mean())
    x = np.where(nan_mask, 0.0, raw)
    x = np.clip(x, -clip_gauss, clip_gauss) / clip_gauss
    x = area_resize(pad_to_square(x), size)
    out = np.clip(x, -1.0, 1.0).astype(dtype)  # clip guards float rounding only; averaging cannot leave [-1, 1]
    if out.shape != (size, size):
        raise AssertionError(f"preprocess produced shape {out.shape}, expected {(size, size)}")
    return out, nan_fraction


def frame_path(frames_dir: Path, harpnum: int, time: pd.Timestamp) -> Path:
    """data/processed/frames/<harpnum>/<YYYYMMDD_HHMM>.npy"""
    return Path(frames_dir) / str(int(harpnum)) / f"{pd.Timestamp(time):%Y%m%d_%H%M}.npy"


def save_frame(path: Path, arr: np.ndarray) -> None:
    """Atomic save (temp file + rename) so an interrupted run never leaves a truncated .npy behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    tmp.replace(path)


class Manifest:
    """Append-only CSV manifest, flushed per row so Ctrl+C never loses recorded frames."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=MANIFEST_COLUMNS)
        if new:
            self._w.writeheader()
            self._f.flush()

    def add(self, harpnum: int, time: pd.Timestamp, status: str, raw_h=None, raw_w=None,
            nan_fraction=None, error: str | None = None) -> None:
        self._w.writerow({"harpnum": int(harpnum), "time": pd.Timestamp(time).strftime("%Y-%m-%d %H:%M:%S"),
                          "status": status, "raw_h": raw_h, "raw_w": raw_w,
                          "nan_fraction": None if nan_fraction is None else round(float(nan_fraction), 6),
                          "error": error})
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def chunk_frames(frames: pd.DataFrame, max_records: int) -> list[pd.DataFrame]:
    """Split frames into per-HARP chunks of at most ``max_records`` rows (sorted by time within a HARP)."""
    if max_records < 1:
        raise ValueError("max_records must be >= 1")
    out = []
    for _, g in frames.sort_values(["harpnum", "time"]).groupby("harpnum", sort=True):
        for i in range(0, len(g), max_records):
            out.append(g.iloc[i:i + max_records])
    return out
