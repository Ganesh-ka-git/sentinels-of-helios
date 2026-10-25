"""FlareDataset, availability filter, loaders and class weights (decisions.md §4, §7, §8).

Frames are read-only inputs: ``<frames_dir>/<harpnum>/<YYYYMMDD_HHMM>.npy`` (128x128 float16, written by Module 3).
A sample's ``frames`` column holds 31 ids ``YYYYMMDDTHHMM``, oldest -> newest, the last one being t. Nothing later
than t is ever read.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

N_FRAMES = 31
COARSE = slice(0, None, 2)   # 16 frames, 12 h @ 48 min, ends at t
FINE = slice(15, None)       # 16 frames, last 6 h @ 24 min, ends at t
MODES = ("static", "single", "dual")
SPLITS = ("train", "val", "test")
N_CLASSES = 3
IMG = 128


def frame_filename(frame_id: str) -> str:
    """'20240101T0024' -> '20240101_0024.npy'"""
    if len(frame_id) != 13 or frame_id[8] != "T":
        raise ValueError(f"Bad frame id {frame_id!r}; expected YYYYMMDDTHHMM")
    return f"{frame_id[:8]}_{frame_id[9:]}.npy"


def load_samples(samples_csv: Path) -> pd.DataFrame:
    """Read the sample index; keep included rows with 31 frame ids. Adds ``frame_ids`` (list[str])."""
    df = pd.read_csv(samples_csv, parse_dates=["t"])
    df = df[df["included"].astype(bool)].copy()
    df["frame_ids"] = df["frames"].str.split(";")
    bad = df[df["frame_ids"].map(len) != N_FRAMES]
    if len(bad):
        raise ValueError(f"{len(bad)} included samples do not have {N_FRAMES} frames (first: HARP "
                         f"{bad['harpnum'].iloc[0]} t={bad['t'].iloc[0]})")
    if not (df["frame_ids"].str[-1] == df["t"].dt.strftime("%Y%m%dT%H%M")).all():
        raise ValueError("For some samples the last frame is not t; refusing to build inputs.")
    for col in ("label", "persistence_label"):
        df[col] = df[col].astype(int)
    return df.sort_values(["t", "harpnum"]).reset_index(drop=True)


def filter_available(samples: pd.DataFrame, frames_dir: Path) -> pd.DataFrame:
    """Return the samples whose 31 frame files all exist (one directory listing per HARP; read-only)."""
    frames_dir = Path(frames_dir)
    listing: dict[int, set[str]] = {}
    for h in samples["harpnum"].unique():
        d = frames_dir / str(int(h))
        listing[int(h)] = {e.name for e in os.scandir(d)} if d.is_dir() else set()
    ok = [all(frame_filename(f) in listing[int(h)] for f in ids)
          for h, ids in zip(samples["harpnum"], samples["frame_ids"])]
    return samples[np.asarray(ok, dtype=bool)].reset_index(drop=True)


def availability_report(all_samples: pd.DataFrame, available: pd.DataFrame, print_fn=print) -> pd.DataFrame:
    """Print available / total per split with class counts (0/1/2). Returns the table."""
    rows = []
    for sp in SPLITS:
        tot, av = all_samples[all_samples["split"] == sp], available[available["split"] == sp]
        rows.append({"split": sp, "available": len(av), "total": len(tot),
                     "pct": round(100 * len(av) / len(tot), 1) if len(tot) else float("nan"),
                     **{f"avail_c{c}": int((av["label"] == c).sum()) for c in range(N_CLASSES)},
                     **{f"total_c{c}": int((tot["label"] == c).sum()) for c in range(N_CLASSES)}})
    table = pd.DataFrame(rows)
    for r in rows:
        print_fn(f"{r['split']:<5} available {r['available']:>5}/{r['total']:<5} ({r['pct']}%)  classes 0/1/2 "
                 f"available {r['avail_c0']}/{r['avail_c1']}/{r['avail_c2']}  of total "
                 f"{r['total_c0']}/{r['total_c1']}/{r['total_c2']}")
    return table


def class_weights(samples: pd.DataFrame, n_classes: int = N_CLASSES) -> torch.Tensor:
    """w_c = N / (K * N_c) computed on the TRAIN split only (rows of other splits are ignored)."""
    y = samples.loc[samples["split"] == "train", "label"].to_numpy()
    if len(y) == 0:
        raise ValueError("No training samples: cannot compute class weights.")
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if (counts == 0).any():
        raise ValueError(f"Class weight undefined: classes {list(np.where(counts == 0)[0])} have no training "
                         f"samples (train counts {counts.astype(int).tolist()}).")
    return torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32)


class FlareDataset(Dataset):
    """Returns (inputs, label, meta).

    mode="static": frame t -> [1,128,128]
    mode="single": 31 frames -> [31,1,128,128]
    mode="dual":   (coarse [16,1,128,128] = frames[0::2], fine [16,1,128,128] = frames[15:])
    Tensors are float32 (files are float16). meta = {"harpnum", "t", "persistence_label"}.
    """

    def __init__(self, samples: pd.DataFrame, frames_dir: Path, mode: str):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.frames_dir = mode, Path(frames_dir)
        self.harp = samples["harpnum"].astype(int).to_numpy()
        self.ids = list(samples["frame_ids"])
        self.label = samples["label"].astype(int).to_numpy()
        self.persistence = samples["persistence_label"].astype(int).to_numpy()
        self.t = [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M") for x in samples["t"]]
        if any(len(i) != N_FRAMES for i in self.ids):
            raise ValueError(f"Every sample needs {N_FRAMES} frame ids")
        idx = np.arange(N_FRAMES)
        self._wanted = {"static": idx[-1:], "single": idx,
                        "dual": np.union1d(idx[COARSE], idx[FINE])}[mode]

    def __len__(self) -> int:
        return len(self.label)

    def _load(self, i: int) -> dict[int, np.ndarray]:
        out = {}
        for k in self._wanted:
            p = self.frames_dir / str(self.harp[i]) / frame_filename(self.ids[i][k])
            arr = np.load(p)  # a missing/corrupt file raises: never substitute silently
            if arr.shape != (IMG, IMG) or arr.dtype != np.float16:
                raise ValueError(f"{p}: expected ({IMG},{IMG}) float16, got {arr.shape} {arr.dtype}")
            out[int(k)] = arr
        return out

    def __getitem__(self, i: int):
        fr = self._load(i)

        def stack(sel) -> torch.Tensor:
            ks = np.arange(N_FRAMES)[sel]
            return torch.from_numpy(np.stack([fr[int(k)] for k in ks]).astype(np.float32)).unsqueeze(1)

        if self.mode == "static":
            x = stack(slice(N_FRAMES - 1, None))[0]          # [1,128,128]
        elif self.mode == "single":
            x = stack(slice(None))                          # [31,1,128,128]
        else:
            x = (stack(COARSE), stack(FINE))                # 2 x [16,1,128,128]
        meta = {"harpnum": int(self.harp[i]), "t": self.t[i], "persistence_label": int(self.persistence[i])}
        return x, torch.tensor(int(self.label[i]), dtype=torch.long), meta


def make_loaders(mode: str, batch_size: int, num_workers: int, *, samples: pd.DataFrame, frames_dir: Path,
                 seed: int = 0) -> dict[str, DataLoader]:
    """DataLoaders for train/val/test built from ``samples`` (already availability-filtered).

    Shuffle only for train; pin_memory when CUDA exists; persistent workers when num_workers > 0. Splits with no
    samples are omitted with a warning; an empty train split raises. Callers using num_workers > 0 on Windows need
    an ``if __name__ == "__main__":`` guard (spawn).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    loaders = {}
    for sp in SPLITS:
        sub = samples[samples["split"] == sp]
        if sub.empty:
            if sp == "train":
                raise ValueError("No available training samples.")
            log.warning("Split %r has no available samples; no loader created for it.", sp)
            continue
        gen = torch.Generator().manual_seed(seed) if sp == "train" else None
        loaders[sp] = DataLoader(FlareDataset(sub, frames_dir, mode), batch_size=batch_size,
                                 shuffle=(sp == "train"), num_workers=num_workers,
                                 pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                                 drop_last=False, generator=gen)
    return loaders
