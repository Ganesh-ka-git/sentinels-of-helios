"""Packed float16 frame cache (decisions.md Change Log 14).

All frames referenced by the sample index are copied into ONE array ``frames_f16.npy`` [N,128,128] plus
``frames_index.csv`` (harpnum, time, row), ``time`` being the frame id ``YYYYMMDDTHHMM``. The per-frame ``.npy`` files
stay the source of truth: the cache is rebuildable and read-only for training.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .flare_dataset import CACHE_FILE, CACHE_INDEX, IMG, frame_filename

log = logging.getLogger(__name__)


def unique_frames(samples: pd.DataFrame) -> pd.DataFrame:
    """Sorted, de-duplicated (harpnum, time) of every frame of every sample (all splits)."""
    pairs = {(int(h), f) for h, ids in zip(samples["harpnum"], samples["frame_ids"]) for f in ids}
    df = pd.DataFrame(sorted(pairs), columns=["harpnum", "time"])
    df["row"] = np.arange(len(df))
    return df


def _read(frames_dir: Path, h: int, f: str) -> np.ndarray:
    arr = np.load(Path(frames_dir) / str(h) / frame_filename(f))
    if arr.shape != (IMG, IMG) or arr.dtype != np.float16:
        raise ValueError(f"{h}/{f}: expected ({IMG},{IMG}) float16, got {arr.shape} {arr.dtype}")
    return arr


def build_cache(samples: pd.DataFrame, frames_dir: Path, cache_dir: Path, workers: int = 8,
                progress_every: int = 20000) -> pd.DataFrame:
    """Write the packed array and index; fails (writing nothing final) if any source frame is missing/bad."""
    frames_dir, cache_dir = Path(frames_dir), Path(cache_dir)
    idx = unique_frames(samples)
    missing = [(h, f) for h, f in zip(idx["harpnum"], idx["time"])
               if not (frames_dir / str(h) / frame_filename(f)).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source frames missing (first: {missing[0]}); cache not built")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / (CACHE_FILE + ".tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(idx), IMG, IMG))
    harp, time_ = idx["harpnum"].to_numpy(), idx["time"].to_numpy()

    def work(i: int) -> None:
        out[i] = _read(frames_dir, int(harp[i]), time_[i])

    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(work, range(len(idx))), 1):   # iterating re-raises worker exceptions
            if n % progress_every == 0:
                log.info("cached %d / %d frames", n, len(idx))
    out.flush()
    del out
    os.replace(tmp, cache_dir / CACHE_FILE)
    idx.to_csv(cache_dir / CACHE_INDEX, index=False)
    return idx


def verify_cache(frames_dir: Path, cache_dir: Path, n: int = 1000, seed: int = 0) -> int:
    """Compare ``n`` random cache rows with their source files; raises on any difference. Returns rows checked."""
    cache_dir = Path(cache_dir)
    idx = pd.read_csv(cache_dir / CACHE_INDEX, dtype={"harpnum": int, "time": str, "row": int})
    arr = np.load(cache_dir / CACHE_FILE, mmap_mode="r")
    if arr.shape != (len(idx), IMG, IMG) or arr.dtype != np.float16:
        raise ValueError(f"cache shape/dtype {arr.shape} {arr.dtype} does not match the index ({len(idx)} rows)")
    pick = np.random.default_rng(seed).choice(len(idx), size=min(n, len(idx)), replace=False)
    for i in pick:
        r = idx.iloc[i]
        if not np.array_equal(arr[int(r["row"])], _read(frames_dir, int(r["harpnum"]), r["time"])):
            raise AssertionError(f"cache row {r['row']} (HARP {r['harpnum']}, {r['time']}) differs from its source")
    return len(pick)
