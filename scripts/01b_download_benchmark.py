"""Module 1b — Download throughput benchmark (no sample/label logic).

Takes the first N rows of outputs/feasibility/<YYYY-MM>/frames_needed.csv, exports their Br segment with
method="url_quick", protocol="as-is", then downloads the files with requests + a thread pool for each worker count
(default 1, 4, 8). Each worker count writes to its own folder so runs do not skip each other's files.

Run from the project root (PowerShell):
    python scripts/01b_download_benchmark.py
    python scripts/01b_download_benchmark.py --workers 1 4 --n-frames 30
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config, require_email  # noqa: E402
from flarecast.data.jsoc import build_record_sets, download_all, export_urls  # noqa: E402

log = logging.getLogger("download_benchmark")


def summarise(results: list[dict], wall_s: float, workers: int) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    new_mb = sum(r["bytes"] for r in ok) / 1e6
    total_mb = new_mb + sum(r["bytes"] for r in skipped) / 1e6
    n_have = len(ok) + len(skipped)
    return {
        "workers": workers,
        "files_ok": len(ok),
        "files_skipped_existing": len(skipped),
        "files_failed": len(failed),
        "failed_errors": {r["filename"]: r["error"] for r in failed},
        "total_mb": round(total_mb, 3),
        "mb_per_file": round(total_mb / n_have, 4) if n_have else None,
        "wall_seconds": round(wall_s, 1),
        "files_per_s": round(len(ok) / wall_s, 3) if wall_s > 0 else None,
        "mb_per_s": round(new_mb / wall_s, 3) if wall_s > 0 else None,
    }


def check_fits(path: Path) -> dict:
    """Open a FITS file and return shape/dtype/NaN fraction of its first 2-D image HDU."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        for i, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None and data.ndim == 2:
                arr = np.asarray(data)
                nan_frac = float(np.isnan(arr).mean()) if arr.dtype.kind == "f" else 0.0
                return {"file": path.name, "hdu": i, "shape": list(arr.shape), "dtype": str(arr.dtype),
                        "nan_fraction": round(nan_frac, 4)}
    raise ValueError(f"{path.name}: no 2-D image HDU found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--fresh", action="store_true", help="delete existing benchmark folders first")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    email = require_email(cfg)
    dl_cfg = cfg["download"]
    start = pd.Timestamp(cfg["pilot"]["start"])
    feas_dir = Path(cfg["paths"]["outputs_dir"]) / "feasibility"
    need_csv = feas_dir / start.strftime("%Y-%m") / "frames_needed.csv"
    frames = pd.read_csv(need_csv, parse_dates=["time"]).head(args.n_frames)
    if len(frames) < args.n_frames:
        log.warning("Only %d rows available in %s", len(frames), need_csv)

    import drms
    client = drms.Client(email=email)
    record_sets = build_record_sets(frames, cfg["jsoc"]["series"], cfg["jsoc"]["segment"])
    tic = time.time()
    urls = export_urls(client, record_sets, email, cfg["jsoc"]["segment"])  # JSOC errors propagate: fail loudly
    export_s = round(time.time() - tic, 1)
    if len(urls) != len(frames):
        log.warning("Requested %d frames but JSOC returned %d files", len(frames), len(urls))
    items = list(zip(urls["filename"], urls["url"]))

    base = Path(cfg["paths"]["data_dir"]) / "benchmark_fits"
    runs = []
    for w in args.workers:
        out_dir = base / f"workers_{w}"
        if args.fresh and out_dir.exists():
            shutil.rmtree(out_dir)
        log.info("Downloading %d files with %d worker(s) -> %s", len(items), w, out_dir)
        tic = time.time()
        results = download_all(items, out_dir, workers=w, timeout_s=dl_cfg["timeout_s"], retries=dl_cfg["retries"])
        runs.append(summarise(results, time.time() - tic, w))

    report = {
        "protocol": "url_quick / as-is",
        "n_frames_requested": int(len(frames)),
        "n_files_exported": int(len(urls)),
        "n_record_sets": len(record_sets),
        "export_seconds": export_s,
        "timeout_s": dl_cfg["timeout_s"],
        "retries": dl_cfg["retries"],
        "runs": runs,
    }

    sample_files = sorted((base / f"workers_{args.workers[0]}").glob("*.fits"))[:3]
    fits_checks = []
    for p in sample_files:
        fits_checks.append(check_fits(p))
    if len(fits_checks) < 3:
        log.warning("Only %d FITS files available for verification", len(fits_checks))
    report["fits_checks"] = fits_checks

    out = feas_dir / "download_benchmark.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n================ DOWNLOAD BENCHMARK ================")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWritten to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
