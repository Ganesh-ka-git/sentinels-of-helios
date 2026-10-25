"""Module 4b — build the packed float16 frame cache from the per-frame .npy files (read-only on the sources).

Reads every frame listed by the included rows of outputs/index/samples_final.csv (all splits, de-duplicated) into
data/processed/cache/frames_f16.npy [N,128,128] + frames_index.csv (harpnum, time, row), then verifies 1,000 random
rows against the source files (exact equality). Rebuildable at any time.

Run from the project root (PowerShell):
    conda run -n flare --no-capture-output python scripts/04b_build_cache.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.dataset import load_samples  # noqa: E402
from flarecast.dataset.cache import build_cache, verify_cache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8, help="reader threads")
    ap.add_argument("--verify", type=int, default=1000, help="random rows compared with the source files")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    frames_dir, cache_dir = cfg["paths"]["frames_dir"], cfg["paths"]["cache_dir"]
    samples = load_samples(cfg["paths"]["samples_csv"])
    print(f"{len(samples)} included samples")
    tic = time.perf_counter()
    idx = build_cache(samples, frames_dir, cache_dir, a.workers)
    print(f"Built {len(idx):,} frames in {time.perf_counter() - tic:.0f} s -> {cache_dir}")
    n = verify_cache(frames_dir, cache_dir, a.verify)
    print(f"Verified {n} random rows: identical to the per-frame files")
    mb = (cache_dir / "frames_f16.npy").stat().st_size / 2**20
    print(f"frames_f16.npy: {mb:,.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
