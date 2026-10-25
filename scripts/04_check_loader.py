"""Module 4 — check the dataset/loader on whatever frames are downloaded so far (read-only on data/processed).

Prints availability per split, class weights, shapes and seconds/batch for each mode with num_workers 0 and 4, and
saves a PNG of the 31 frames of one M/X-class training sample (outputs/figures/sample_sequence.png).

Run from the project root (PowerShell):
    python scripts/04_check_loader.py
    python scripts/04_check_loader.py --batches 20 --workers 0 4 --modes static single dual
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.dataset import (FlareDataset, availability_report, class_weights, filter_available,  # noqa: E402
                               load_samples, make_loaders)

log = logging.getLogger("check_loader")
CLASS_NAMES = {0: "<M", 1: "M", 2: "X"}


def shapes_of(x) -> str:
    return " + ".join(str(tuple(t.shape)) for t in x) if isinstance(x, (list, tuple)) else str(tuple(x.shape))


def time_loader(mode: str, workers: int, batch_size: int, n_batches: int, samples: pd.DataFrame,
                frames_dir: Path) -> dict:
    """Iterate up to n_batches train batches; report shapes and seconds/batch (first batch separately)."""
    loader = make_loaders(mode, batch_size, workers, samples=samples, frames_dir=frames_dir)["train"]
    stamps, shapes = [], None
    tic = time.perf_counter()
    for i, (x, y, meta) in enumerate(loader):
        stamps.append(time.perf_counter() - tic)
        if i == 0:
            shapes = f"x {shapes_of(x)} {(x[0] if isinstance(x, (list, tuple)) else x).dtype}, y {tuple(y.shape)} {y.dtype}"
        if i + 1 >= n_batches:
            break
    n = len(stamps)
    del loader
    gc.collect()  # lets persistent workers shut down before the next configuration
    return {"mode": mode, "workers": workers, "batches": n, "shapes": shapes,
            "first_batch_s": round(stamps[0], 2), "s_per_batch_total": round(stamps[-1] / n, 3),
            "s_per_batch_steady": round((stamps[-1] - stamps[0]) / (n - 1), 3) if n > 1 else None}


def save_sequence_figure(samples: pd.DataFrame, frames_dir: Path, out: Path) -> str | None:
    """Plot the 31 frames of the first available M/X training sample. Returns a description, or None if none."""
    cand = samples[(samples["split"] == "train") & (samples["label"] >= 1)]
    if cand.empty:
        return None
    row = cand.iloc[[0]]
    x, label, meta = FlareDataset(row, frames_dir, "single")[0]
    t = pd.Timestamp(row["t"].iloc[0])
    ids = row["frame_ids"].iloc[0]
    fig, axes = plt.subplots(4, 8, figsize=(16, 8.6), constrained_layout=True)
    im = None
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= len(ids):
            continue
        im = ax.imshow(x[k, 0].numpy(), cmap="gray", vmin=-1, vmax=1, origin="lower")
        dt = int(round((t - pd.to_datetime(ids[k], format="%Y%m%dT%H%M")).total_seconds() / 60))
        ax.set_title("t" if dt == 0 else f"t−{dt // 60}h{dt % 60:02d}", fontsize=9)
    desc = (f"HARP {int(row['harpnum'].iloc[0])}, t = {t:%Y-%m-%d %H:%M} UTC, class {CLASS_NAMES[int(label)]}"
            f" (max flare in [t+6h, t+24h): {row['max_class'].iloc[0]})")
    fig.suptitle(f"Input sequence of one training sample: 31 Br frames, 24 min apart, oldest to newest\n{desc}",
                 fontsize=11)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, location="right", pad=0.01)
    cb.set_label("Br / 1500 G (clipped, 128×128)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return desc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--modes", nargs="+", default=["static", "single", "dual"])
    ap.add_argument("--fig", default=None, help="default <outputs>/figures/sample_sequence.png")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    bs = args.batch_size or cfg["loader"]["batch_size"]
    frames_dir = cfg["paths"]["frames_dir"]

    all_samples = load_samples(cfg["paths"]["samples_csv"])
    avail = filter_available(all_samples, frames_dir)
    print(f"Included samples in index: {len(all_samples)}; with all 31 frames on disk: {len(avail)}")
    availability_report(all_samples, avail)
    try:
        print("Class weights (train, N/(K*N_c)):", [round(float(w), 3) for w in class_weights(avail)])
    except ValueError as e:
        print(f"Class weights not available yet: {e}")

    if avail[avail["split"] == "train"].empty:
        print("No available training samples yet; loader timing skipped.")
        return 1
    n_train = int((avail["split"] == "train").sum())
    print(f"\nTiming up to {args.batches} train batches of {bs} ({n_train} train samples available)")
    for mode in args.modes:
        for w in args.workers:
            r = time_loader(mode, w, bs, args.batches, avail, frames_dir)
            print(f"  {r['mode']:<7} workers={r['workers']}  {r['batches']} batches  {r['shapes']}  "
                  f"first batch {r['first_batch_s']} s  steady {r['s_per_batch_steady']} s/batch  "
                  f"(incl. first {r['s_per_batch_total']})")

    fig_path = Path(args.fig) if args.fig else cfg["paths"]["outputs_dir"] / "figures" / "sample_sequence.png"
    desc = save_sequence_figure(avail, frames_dir, fig_path)
    if desc is None:
        print("\nNo M/X training sample has all 31 frames on disk yet; figure not created (re-run later).")
    else:
        print(f"\nFigure saved: {fig_path}\n  {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
