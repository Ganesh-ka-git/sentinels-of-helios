"""Read-only progress bar for scripts/02_build_index.py. Safe to start/stop at any time; never touches the build.

Run in a second PowerShell window from the project root:
    conda run -n flare --no-capture-output python scripts/watch_index.py
Ctrl+C closes only the watcher.

A month is DONE when its folder has sharp_keywords.csv, flares_associated.csv and samples.csv. A folder that exists
but is not complete is the month currently being processed (or a failed one).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.labels.index import month_starts, priority_order  # noqa: E402

CACHE_FILES = ("sharp_keywords.csv", "flares_associated.csv", "samples.csv")
BAR_WIDTH = 30


def month_summary(d: Path) -> str:
    s = pd.read_csv(d / "samples.csv", keep_default_na=False, na_values=[""])
    inc = s[s["included"].astype(bool)]
    c = inc["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    fr = d / "frames_needed.csv"
    n_frames = len(pd.read_csv(fr)) if fr.exists() else "?"
    return f"included={len(inc)}/{len(s)} classes(0/1/2)={c[0]}/{c[1]}/{c[2]} frames_needed={n_frames}"


def fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def main() -> int:
    cfg = load_config()
    index_dir = Path(cfg["paths"]["outputs_dir"]) / "index"
    order = [m.strftime("%Y-%m") for m in priority_order(
        month_starts(pd.Timestamp(cfg["period"]["start"]), pd.Timestamp(cfg["period"]["end"])),
        list(cfg["period"].get("priority_years", [])))]
    printed: set[str] = set()
    t0 = time.time()
    last_change = t0
    n_done_prev = -1
    try:
        while True:
            done = [m for m in order if all((index_dir / m / f).exists() for f in CACHE_FILES)]
            for m in done:
                if m not in printed:
                    printed.add(m)
                    print(f"\r{' ' * 100}\rMONTH {m}: {month_summary(index_dir / m)}", flush=True)
            if len(done) != n_done_prev:
                n_done_prev, last_change = len(done), time.time()
            current = next((m for m in order if m not in done and (index_dir / m).exists()), None)
            frac = len(done) / len(order)
            fill = int(BAR_WIDTH * frac)
            bar = "#" * fill + "-" * (BAR_WIDTH - fill)
            now = f"working on {current}" if current else "waiting"
            print(f"\r[{bar}] {len(done)}/{len(order)} months ({frac:5.1%}) | {now} | "
                  f"{fmt(time.time() - last_change)} since last month finished   ", end="", flush=True)
            if len(done) == len(order):
                print("\nAll months complete.")
                return 0
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nWatcher stopped (the build is unaffected).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
