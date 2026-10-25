"""Module 3 — Download Br frames from JSOC and preprocess them to 128x128 float16 .npy files.

For every month in the frames CSV (chronological order), every HARP's frames are exported (url_quick / as-is, <=300
records per request), downloaded with a thread pool, preprocessed immediately and the raw FITS deleted:
    data/processed/frames/<harpnum>/<YYYYMMDD_HHMM>.npy
Every frame is appended to data/processed/manifest.csv. Re-running skips frames whose .npy exists; Ctrl+C is safe.

Run from the project root (PowerShell):
    python scripts/03_download_preprocess.py                                   # everything in frames_needed_final.csv
    python scripts/03_download_preprocess.py --months 2024-05 --limit 100      # quick end-to-end test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config, require_email  # noqa: E402
from flarecast.data.jsoc import download_file, export_urls  # noqa: E402
from flarecast.preprocess.frames import Manifest  # noqa: E402
from flarecast.preprocess.pipeline import (PreprocessSettings, Progress, pending_frames,  # noqa: E402
                                           process_frames, process_frames_tar, retry_export)

log = logging.getLogger("download_preprocess")


def load_frames(path: Path, months: list[str] | None, limit: int | None) -> pd.DataFrame:
    """Read the frames CSV, keep the requested months, sort chronologically by month then HARP/time, apply --limit."""
    if not path.exists():
        raise FileNotFoundError(f"Frames CSV not found: {path} (pass --frames)")
    df = pd.read_csv(path, parse_dates=["time"])
    missing = {"harpnum", "time"} - set(df.columns)
    if missing:
        raise KeyError(f"{path} lacks required columns: {sorted(missing)}")
    if "month" not in df.columns:
        df["month"] = df["time"].dt.strftime("%Y-%m")
    df["month"] = df["month"].astype(str)
    df = df.drop_duplicates(subset=["harpnum", "time"])
    if months:
        unknown = set(months) - set(df["month"])
        if unknown:
            raise ValueError(f"Months not present in {path.name}: {sorted(unknown)}")
        df = df[df["month"].isin(months)]
    df = df.sort_values(["month", "harpnum", "time"]).reset_index(drop=True)
    return df.head(limit) if limit else df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--frames", default=None, help="frames CSV (harpnum, time, month); "
                    "default <outputs>/index/frames_needed_final.csv")
    ap.add_argument("--months", nargs="+", default=None, help="restrict to these YYYY-MM months")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N frames (after month filter)")
    ap.add_argument("--workers", type=int, default=None, help="override download.workers")
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--method", choices=["url-quick", "url-tar"], default="url-quick",
                    help="url-quick: one URL per file (default); url-tar: one staged .tar per HARP chunk")
    ap.add_argument("--tar-workers", type=int, default=None, help="url-tar: concurrent chunks (download.tar_workers)")
    ap.add_argument("--out-dir", default=None, help="write frames/, manifest.csv and raw_tmp/ under this directory "
                    "instead of <data_dir>/processed and <data_dir>/raw_tmp (use for benchmarks)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    email = require_email(cfg)
    dl, pp, jc = cfg["download"], cfg["preprocess"], cfg["jsoc"]
    data_dir = cfg["paths"]["data_dir"]
    if args.out_dir:
        out = Path(args.out_dir)
        frames_dir, raw_dir, manifest_path = out / "frames", out / "raw_tmp", out / "manifest.csv"
    else:
        frames_dir, raw_dir = data_dir / "processed" / "frames", data_dir / "raw_tmp"
        manifest_path = data_dir / "processed" / "manifest.csv"
    frames_csv = Path(args.frames) if args.frames else cfg["paths"]["outputs_dir"] / "index" / "frames_needed_final.csv"

    st = PreprocessSettings(series=jc["series"], segment=jc["segment"], size=pp["image_size"],
                            clip_gauss=pp["clip_gauss"], dtype=pp["dtype"],
                            workers=args.workers or dl["workers"], chunk_records=dl["export_chunk_records"])
    frames = load_frames(frames_csv, args.months, args.limit)
    months = list(dict.fromkeys(frames["month"]))
    per_month = {m: pending_frames(frames[frames["month"] == m], frames_dir) for m in months}
    total = sum(len(p) for p, _ in per_month.values())
    n_skip = sum(s for _, s in per_month.values())
    log.info("%d frames in %d month(s) from %s: %d already done (skipped), %d to process, %d workers",
             len(frames), len(months), frames_csv.name, n_skip, total, st.workers)
    if total == 0:
        log.info("Nothing to do.")
        return 0

    import drms
    import requests
    client = drms.Client(email=email)
    local = threading.local()

    def download_fn(url: str, dest: Path) -> dict:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        return download_file(url, dest, local.session, dl["timeout_s"], dl["retries"])

    stop_waiting = threading.Event()  # set on Ctrl+C so a retry wait does not delay shutdown

    def export_fn(record_sets: list[str]) -> pd.DataFrame:
        return export_urls(client, record_sets, email, st.segment)

    export_fn = retry_export(export_fn, stop=stop_waiting)  # 30/60/120/240/150 s waits, then the chunk is failed

    export_lock = threading.Lock()

    def tar_export_fn(record_set: str) -> str:
        """Stage one url-tar export and return the absolute tar URL; retries transient JSOC timeouts."""
        last = None
        for attempt in range(1, dl["tar_retries"] + 2):
            try:
                with export_lock:  # submit one request at a time; the wait for staging runs unlocked
                    req = client.export(record_set, method="url-tar", protocol="as-is", email=email)
                req.wait()
                urls = req.urls["url"]
                if len(urls) != 1:
                    raise RuntimeError(f"expected 1 tar URL, JSOC returned {len(urls)}: {list(urls)[:3]}")
                return str(urls.iloc[0])
            except Exception as e:
                last = e
                log.warning("tar export attempt %d/%d failed: %r", attempt, dl["tar_retries"] + 1, e)
                if attempt <= dl["tar_retries"]:
                    time.sleep(5 * 2 ** (attempt - 1))
        raise last

    def tar_download_fn(url: str, dest: Path) -> dict:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        return download_file(url, dest, local.session, dl["tar_timeout_s"], dl["tar_retries"])

    progress = Progress(total, every=args.progress_every)
    totals = {"ok": 0, "failed": 0, "missing": 0}
    tar_stats = {"n_chunks": 0, "staging_s": 0.0, "download_s": 0.0, "tar_bytes": 0, "n_files": 0}
    interrupted = False
    wall0 = time.monotonic()
    with Manifest(manifest_path) as manifest:
        try:
            for m in months:
                pending, skipped = per_month[m]
                log.info("=== %s: %d to process (%d already done) ===", m, len(pending), skipped)
                if args.method == "url-tar":
                    counts, s = process_frames_tar(
                        pending, frames_dir=frames_dir, raw_dir=raw_dir, manifest=manifest,
                        tar_export_fn=tar_export_fn, tar_download_fn=tar_download_fn, st=st, progress=progress,
                        chunk_workers=args.tar_workers or dl["tar_workers"])
                    for k, v in s.items():
                        tar_stats[k] += v
                else:
                    counts = process_frames(pending, frames_dir=frames_dir, raw_dir=raw_dir, manifest=manifest,
                                            export_fn=export_fn, download_fn=download_fn, st=st, progress=progress)
                for k, v in counts.items():
                    totals[k] += v
                log.info("=== %s done: %s ===", m, counts)
        except KeyboardInterrupt:
            stop_waiting.set()
            interrupted = True
            log.warning("Interrupted (Ctrl+C). Finished frames are saved; re-run the same command to resume.")
    wall = time.monotonic() - wall0

    print(f"\n{progress.done}/{total} frames processed: {totals}; overall {progress.overall_rate():.2f} files/s "
          f"({st.workers} workers); {n_skip} skipped as already done")
    if args.method == "url-tar":
        n_ok = totals["ok"]
        print("TAR BENCHMARK " + json.dumps({
            "method": "url-tar", "tar_workers": args.tar_workers or dl["tar_workers"],
            "chunk_records": st.chunk_records, "n_chunks": tar_stats["n_chunks"],
            "frames_ok": n_ok, "frames_failed": totals["failed"], "frames_missing": totals["missing"],
            "staging_seconds_sum": round(tar_stats["staging_s"], 1),
            "download_seconds_sum": round(tar_stats["download_s"], 1),
            "tar_mb": round(tar_stats["tar_bytes"] / 1e6, 2),
            "mb_per_frame": round(tar_stats["tar_bytes"] / 1e6 / max(tar_stats["n_files"], 1), 4),
            "wall_seconds": round(wall, 1),
            "effective_files_per_s_incl_staging": round(n_ok / wall, 3) if wall > 0 else None}))
    print(f"Manifest: {manifest_path}")
    if interrupted:
        return 130
    return 1 if totals["failed"] else 0  # 'missing' = JSOC has no such record; listed in the manifest


if __name__ == "__main__":
    sys.exit(main())
