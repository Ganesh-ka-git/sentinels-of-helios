"""Module 2b — Catalogue-completeness filter and final split, from cached per-month files (no requery).

  1. For every indexed month: share of its >=M flares that have a NOAA number; keep it only if >= completeness.min_share.
  2. STOP (exit 2, nothing written) if the kept months differ from the expected range in the config.
  3. Rebuild the HARP-grouped split over kept months only (pattern by month index within the kept period).
  4. Write outputs/index/samples_final.csv, frames_needed_final.csv, index_report_final.json.

Run from the project root (PowerShell):
    python scripts/02b_finalize_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.labels.index import (build_report, check_expected_kept, completeness_table,  # noqa: E402
                                    frames_needed_all, month_starts, rebuild_split)

CACHE_FILES = ("sharp_keywords.csv", "flares_associated.csv", "samples.csv")


def read_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["t"], keep_default_na=False, na_values=[""])
    df["excluded_reason"] = df["excluded_reason"].fillna("")
    df["max_class"] = df["max_class"].fillna("")
    df["included"] = df["included"].astype(bool)
    return df


def main() -> int:
    cfg = load_config()
    comp = cfg["completeness"]
    index_dir = Path(cfg["paths"]["outputs_dir"]) / "index"
    months = [m.strftime("%Y-%m") for m in month_starts(pd.Timestamp(cfg["period"]["start"]),
                                                       pd.Timestamp(cfg["period"]["end"]))]
    missing = [m for m in months if not all((index_dir / m / f).exists() for f in CACHE_FILES)]
    if missing:
        print(f"STOP: months not indexed yet: {missing}. Run scripts/02_build_index.py first.")
        return 2

    flares = {m: pd.read_csv(index_dir / m / "flares_associated.csv", parse_dates=["start"]) for m in months}
    samples = {m: read_samples(index_dir / m / "samples.csv") for m in months}
    table = completeness_table(flares, samples, comp["min_share_with_noaa"])

    pd.set_option("display.width", 200)
    print(f"\n=== CATALOGUE COMPLETENESS (keep if >= {comp['min_share_with_noaa']:.0%} of >=M flares have a NOAA number) ===")
    print(table.to_string(index=False))
    problems = check_expected_kept(table, comp["expected_kept_first"], comp["expected_kept_last"])
    if problems:
        print("\nSTOP: kept months differ from the expected range "
              f"{comp['expected_kept_first']} ... {comp['expected_kept_last']}. Nothing written.")
        for p in problems:
            print("  -", p)
        return 2

    kept = table.loc[table["kept"], "month"].tolist()
    first_record = (pd.concat([pd.read_csv(index_dir / m / "sharp_keywords.csv", usecols=["HARPNUM", "time"],
                                           parse_dates=["time"]).groupby("HARPNUM")["time"].min() for m in kept])
                    .groupby(level=0).min())
    split_by_harp = rebuild_split(kept, first_record, list(cfg["split"]["pattern"]))

    final = pd.concat([samples[m] for m in kept], ignore_index=True).sort_values(["t", "harpnum"]).reset_index(drop=True)
    final["split"] = final["harpnum"].map(split_by_harp)
    if final["split"].isna().any():
        raise RuntimeError("Some samples belong to HARPs without a first record.")
    frames = frames_needed_all({m: samples[m] for m in kept}, split_by_harp)
    final.to_csv(index_dir / "samples_final.csv", index=False)
    frames.to_csv(index_dir / "frames_needed_final.csv", index=False)

    report = {"completeness_min_share": comp["min_share_with_noaa"], "kept_months": kept,
              "dropped_months": [m for m in months if m not in kept],
              "completeness_table": table.to_dict(orient="records"),
              "split_pattern": list(cfg["split"]["pattern"]), **build_report(final, frames)}
    (index_dir / "index_report_final.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n=== FINAL INDEX: {kept[0]} ... {kept[-1]} ({len(kept)} months) ===")
    for sp in ("train", "val", "test"):
        r = report["per_split"][sp]
        cc = r["class_counts"]
        print(f"{sp:5s} n_included={r['n_included']:5d} (candidates {r['n_candidates']}) "
              f"classes 0/1/2={cc['0']}/{cc['1']}/{cc['2']} HARPs={r['n_harps_included']}")
    print("\nframes needed per month:")
    for m, n in report["frames_needed_per_month"].items():
        print(f"  {m}: {n}")
    print(f"TOTAL frames needed: {report['frames_needed_total']}")
    print("estimated download hours:", report["estimated_download_hours"])
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"\nWritten to {index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
