"""Read-only live dashboard for scripts/03_download_preprocess.py. Safe to start/stop at any time; never touches the run.

It only READS data/processed/manifest.csv, the frames CSV and disk usage (no JSOC requests, nothing written).
Open a second PowerShell window in the project root:
    python scripts/watch_download.py
    python scripts/watch_download.py --deadline "2026-10-25 12:00"     # also shows the speed needed to hit a deadline
    python scripts/watch_download.py --once                            # print one snapshot and exit
Ctrl+C closes only the watcher.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.preprocess.monitor import (ManifestTail, RateWindow, Tracker, eta_seconds, fmt_duration,  # noqa: E402
                                          frame_key)

BAR = 24
NPY_KB = 33.0  # size of one 128x128 float16 .npy incl. header


def bar(frac: float, width: int = BAR) -> str:
    n = int(round(width * max(0.0, min(1.0, frac))))
    return "#" * n + "-" * (width - n)


def find_download_process() -> str:
    """Describe the running 03_download_preprocess.py process (read-only PowerShell query)."""
    cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '03_download_preprocess' -and "
           "$_.Name -match 'python' -and $_.CommandLine -notmatch 'conda' } | "
           "ForEach-Object { \"$($_.ProcessId)|$($_.CreationDate.ToString('yyyy-MM-dd HH:mm:ss'))|$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as e:
        return f"unknown ({type(e).__name__})"
    if not out:
        return "NOT RUNNING"
    pid, started, line = out[0].split("|", 2)
    method = "url-tar" if "url-tar" in line else "url-quick"
    return f"running  PID {pid}  started {started}  ({method})"


def load_needed(frames_csv: Path) -> dict[str, str]:
    df = pd.read_csv(frames_csv, parse_dates=["time"])
    if "month" not in df.columns:
        df["month"] = df["time"].dt.strftime("%Y-%m")
    ts = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {frame_key(h, t): m for h, t, m in zip(df["harpnum"], ts, df["month"].astype(str))}


def render(tr: Tracker, rates: RateWindow, proc: str, manifest: Path, data_dir: Path, deadline: datetime | None,
           watch_start: float, last_row_time: float | None) -> str:
    total, done = len(tr.needed), tr.done
    remaining = total - done
    now = time.time()
    r1, _ = rates.rate(60)
    r10, s10 = rates.rate(600)
    r60, s60 = rates.rate(3600)
    rall, sall = rates.rate(1e9)
    basis = r10 or rall
    eta = eta_seconds(remaining, basis)
    L = []
    L.append(f"DOWNLOAD MONITOR (read-only)   {datetime.now():%Y-%m-%d %H:%M:%S}   refresh Ctrl+C to close")
    L.append("=" * 100)
    L.append(f"process : {proc}")
    age = None if last_row_time is None else now - last_row_time
    stall = ""
    if age is not None and age > 180:
        stall = "   <-- WARNING: no new manifest rows for > 3 min (waiting on JSOC retry, or stalled)"
    L.append(f"manifest: last row {'n/a' if age is None else fmt_duration(age) + ' ago'}{stall}")
    L.append("")
    L.append(f"OVERALL [{bar(done / total, 40)}] {done:,}/{total:,}  ({100 * done / total:.1f}%)   remaining {remaining:,}")
    fmt_r = lambda r: "n/a" if r is None else f"{r:5.2f}"  # noqa: E731
    L.append(f"speed   : last 1 min {fmt_r(r1)} | last 10 min {fmt_r(r10)} | last 60 min {fmt_r(r60)} | "
             f"since watcher start {fmt_r(rall)}  frames/s")
    L.append(f"ETA     : {fmt_duration(eta)}  ->  "
             f"{(datetime.now() + timedelta(seconds=eta)):%a %d %b %H:%M}" if eta else "ETA     : n/a (no progress measured yet)")
    if eta and r60 and s60 > 900 and abs(r60 - (basis or 0)) > 0.05:
        e2 = eta_seconds(remaining, r60)
        L.append(f"          (using the 60-min average instead: {fmt_duration(e2)} -> "
                 f"{(datetime.now() + timedelta(seconds=e2)):%a %d %b %H:%M})")
    if deadline:
        left = (deadline - datetime.now()).total_seconds()
        need = remaining / left if left > 0 else float("inf")
        verdict = "ON TRACK" if basis and basis >= need else "BEHIND"
        L.append(f"deadline: {deadline:%a %d %b %H:%M} -> need {need:.2f} frames/s from now; current {fmt_r(basis)}  [{verdict}]")
    L.append("")
    nf = tr.outstanding_failed()
    L.append(f"frames not yet downloaded after a failure: {nf:,}  (retried automatically only on the next re-run)   "
             f"missing at JSOC: {len(tr.missing):,}")
    if tr.new_rows:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(tr.new_rows.items()))
        L.append(f"new rows since watcher start: {parts}")
        for kind, n in tr.new_errors.most_common(5):
            L.append(f"    failures: {n:>4} x {kind}")
    L.append("")
    L.append("PER MONTH   (> = month being processed now)")
    by_month = tr.done_by_month()
    cur = tr.current_month
    for m in sorted(tr.total_by_month):
        d, t = by_month.get(m, 0), tr.total_by_month[m]
        mark = ">" if m == cur and d < t else " "
        state = "DONE" if d >= t else ""
        L.append(f" {mark} {m}  [{bar(d / t, 20)}] {d:>6,}/{t:<6,} {100 * d / t:5.1f}%  {state}")
    L.append("")
    try:
        free = shutil.disk_usage(data_dir).free / 1e9
        L.append(f"disk    : ~{done * NPY_KB / 1e6:.2f} GB of frames written (expected total ~{total * NPY_KB / 1e6:.1f} GB); "
                 f"{free:.0f} GB free on the data drive")
    except OSError as e:
        L.append(f"disk    : n/a ({e})")
    L.append(f"watching: {manifest}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--frames", default=None, help="default <outputs>/index/frames_needed_final.csv")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--deadline", default=None, help='e.g. "2026-10-25 12:00"')
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["paths"]["data_dir"]
    manifest = data_dir / "processed" / "manifest.csv"
    frames_csv = Path(args.frames) if args.frames else cfg["paths"]["outputs_dir"] / "index" / "frames_needed_final.csv"
    deadline = datetime.fromisoformat(args.deadline) if args.deadline else None

    os.system("")  # enables ANSI escape handling in the Windows console
    print(f"Loading {frames_csv.name} and {manifest.name} ...", flush=True)
    tr = Tracker(load_needed(frames_csv))
    tail = ManifestTail(manifest)
    tr.update(tail.poll())   # history before the watcher started
    tr.mark_baseline()
    rates = RateWindow()
    watch_start = time.time()
    last_row = manifest.stat().st_mtime if manifest.exists() else None
    proc, proc_t = find_download_process(), time.time()
    try:
        while True:
            new = tail.poll()
            tr.update(new)
            if new:
                last_row = time.time()
            elif manifest.exists():
                last_row = max(last_row or 0, manifest.stat().st_mtime)
            now = time.time()
            rates.add(now, tr.done)
            if now - proc_t > 30:
                proc, proc_t = find_download_process(), now
            text = render(tr, rates, proc, manifest, data_dir, deadline, watch_start, last_row)
            if args.once:
                print(text)
                return 0
            print("\x1b[H\x1b[J" + text, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped (the download is unaffected).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
