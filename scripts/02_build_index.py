"""Module 2 — Full sample index and HARP-grouped split for the observation period (metadata only, no images).

Per month (cached in outputs/index/<YYYY-MM>/, completed months are skipped so reruns resume):
  SHARP keywords  [month_start - look-back, month_end + 1 day)
  HEK flares      [month_start - 1 day,     month_end + 1 day)
  flares associated with ALL loaded keyword records; samples built for t in [month_start, month_end).
Then: concatenate, assign splits per HARP, write samples_all.csv, frames_needed_all.csv and index_report.json.

Run from the project root (PowerShell):
    python scripts/02_build_index.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config, require_email  # noqa: E402
from flarecast.data.goes import fetch_hek_flares  # noqa: E402
from flarecast.data.jsoc import query_sharp_keywords  # noqa: E402
from flarecast.labels.index import (assign_splits, build_month, build_report, frames_needed_all,  # noqa: E402
                                    month_starts, next_month, priority_order)
from flarecast.labels.samples import SampleConfig, frames_needed  # noqa: E402

log = logging.getLogger("build_index")
CACHE_FILES = ("sharp_keywords.csv", "flares_associated.csv", "samples.csv")  # samples.csv is written last


def is_cached(d: Path) -> bool:
    return all((d / f).exists() for f in CACHE_FILES)


def read_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["t"], keep_default_na=False, na_values=[""])
    df["excluded_reason"] = df["excluded_reason"].fillna("")
    df["max_class"] = df["max_class"].fillna("")
    df["included"] = df["included"].astype(bool)
    return df


def process_month(ms: pd.Timestamp, cfg: dict, scfg: SampleConfig, client, out_dir: Path) -> None:
    me = next_month(ms)
    lookback = pd.Timedelta(minutes=scfg.frame_step_minutes * (scfg.n_frames_full - 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    sharp = query_sharp_keywords(client, cfg["jsoc"]["series"], ms - lookback, me + pd.Timedelta(days=1),
                                 chunk_days=cfg["jsoc"]["query_chunk_days"])
    raw = fetch_hek_flares(ms - pd.Timedelta(days=1), me + pd.Timedelta(days=1),
                           timeout_s=cfg["flares"].get("hek_timeout_s"))
    samples, flares = build_month(sharp, raw, scfg, cfg["flares"]["frm_name"],
                                  cfg["flares"]["association_window_hours"], ms)
    sharp.drop(columns=["noaa_set"]).assign(NOAA_ARS=sharp["NOAA_ARS"].astype(str)).to_csv(
        out_dir / "sharp_keywords.csv", index=False)
    flares.assign(candidates=flares["candidates"].map(lambda c: ",".join(map(str, c)))).to_csv(
        out_dir / "flares_associated.csv", index=False)
    frames_needed(samples).to_csv(out_dir / "frames_needed.csv", index=False)  # usable alone for downloads
    samples.to_csv(out_dir / "samples.csv", index=False)  # last: marks the month as complete


def month_line(label: str, d: Path) -> str:
    """One-line summary of a finished month, read from its cached files."""
    s = read_samples(d / "samples.csv")
    inc = s[s["included"]]
    c = inc["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    n_frames = len(pd.read_csv(d / "frames_needed.csv"))
    return (f"MONTH {label}: included={len(inc)}/{len(s)} classes(0/1/2)={c[0]}/{c[1]}/{c[2]} "
            f"frames_needed={n_frames}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    email = require_email(cfg)
    scfg = SampleConfig.from_dict(cfg["samples"])
    start, end = pd.Timestamp(cfg["period"]["start"]), pd.Timestamp(cfg["period"]["end"])
    pattern = list(cfg["split"]["pattern"])
    index_dir = Path(cfg["paths"]["outputs_dir"]) / "index"
    months = month_starts(start, end)
    order = priority_order(months, list(cfg["period"].get("priority_years", [])))
    log.info("Period %s -> %s: %d months, split pattern %s", start, end, len(months), pattern)
    log.info("Processing order: %s", [m.strftime("%Y-%m") for m in order])

    import drms
    client = drms.Client(email=email)
    failed: list[str] = []
    tic_all = time.time()
    for i, ms in enumerate(order, 1):
        label = ms.strftime("%Y-%m")
        d = index_dir / label
        if is_cached(d):
            if not (d / "frames_needed.csv").exists():  # month cached before per-month frame lists existed
                frames_needed(read_samples(d / "samples.csv")).to_csv(d / "frames_needed.csv", index=False)
            log.info("[%d/%d] %s cached, skipping", i, len(order), label)
            print(month_line(label, d), flush=True)
            continue
        for attempt in (1, 2):
            tic = time.time()
            try:
                process_month(ms, cfg, scfg, client, d)
                log.info("[%d/%d] %s done in %.0f s", i, len(order), label, time.time() - tic)
                print(month_line(label, d), flush=True)
                break
            except Exception:  # log with traceback, retry once, then list as failed
                log.error("[%d/%d] %s attempt %d FAILED:\n%s", i, len(months), label, attempt, traceback.format_exc())
        else:
            failed.append(label)

    done = [m for m in months if is_cached(index_dir / m.strftime("%Y-%m"))]
    if not done:
        log.error("No months completed.")
        return 1

    # concatenate; first record per HARP over everything loaded
    samples_by_month, firsts = {}, []
    for ms in done:
        d = index_dir / ms.strftime("%Y-%m")
        samples_by_month[ms.strftime("%Y-%m")] = read_samples(d / "samples.csv")
        firsts.append(pd.read_csv(d / "sharp_keywords.csv", usecols=["HARPNUM", "time"], parse_dates=["time"])
                      .groupby("HARPNUM")["time"].min())
    first_record = pd.concat(firsts).groupby(level=0).min()
    split_by_harp = assign_splits(first_record, start, pattern)

    samples = pd.concat(samples_by_month.values(), ignore_index=True).sort_values(["t", "harpnum"]).reset_index(drop=True)
    samples["split"] = samples["harpnum"].map(split_by_harp)
    if samples["split"].isna().any():
        raise RuntimeError("Some samples belong to HARPs without a first record.")
    samples.to_csv(index_dir / "samples_all.csv", index=False)
    frames = frames_needed_all(samples_by_month, split_by_harp)
    frames.to_csv(index_dir / "frames_needed_all.csv", index=False)

    report = {"period_start": str(start), "period_end_exclusive": str(end), "split_pattern": pattern,
              "months_completed": len(done), "months_expected": len(months), "failed_months": failed,
              **build_report(samples, frames), "total_runtime_seconds": round(time.time() - tic_all, 1)}
    if failed:
        report["warnings"].append(f"INCOMPLETE: months failed and are missing from all outputs: {failed}")
    (index_dir / "index_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n================ INDEX REPORT ================")
    print(json.dumps(report, indent=2, default=str))
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"\nOutputs written to: {index_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
