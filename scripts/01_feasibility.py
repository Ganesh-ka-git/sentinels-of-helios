"""Module 1 — Feasibility pilot on one month of data.

What it does (no model, no bulk download):
  1. Queries SHARP keywords for every HARP in the pilot month (cadence, coverage, QUALITY, LON_FWT, NOAA_ARS).
  2. Queries GOES flare events from HEK (one day either side), reports catalogue sources, keeps one.
  3. Associates flares with HARPs via NOAA numbers and reports associated / unassociated / ambiguous counts.
  4. Builds the (HARP, t) sample index with labels, persistence labels and exclusion reasons.
  5. Optionally exports a few Br frames to measure MB/frame and download speed.
  6. Writes CSVs + report.json + report.md to outputs/feasibility/<YYYY-MM>/ and prints a summary.

Run from the project root (PowerShell):
    python scripts/01_feasibility.py
    python scripts/01_feasibility.py --no-download      # skip the export test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config, require_email  # noqa: E402
from flarecast.data.goes import clean_flares, fetch_hek_flares  # noqa: E402
from flarecast.data.jsoc import query_sharp_keywords, test_export  # noqa: E402
from flarecast.labels.association import associate_flares  # noqa: E402
from flarecast.labels.samples import SampleConfig, build_samples, frames_needed  # noqa: E402

log = logging.getLogger("feasibility")


def cadence_stats(sharp: pd.DataFrame) -> dict:
    d = sharp.groupby("HARPNUM")["time"].diff().dropna().dt.total_seconds() / 60
    return {
        "median_minutes": float(d.median()),
        "share_exactly_12min": round(float((d == 12).mean()), 4),
        "share_gaps_over_1h": round(float((d > 60).mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-download", action="store_true", help="skip the Br export/download test")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    email = require_email(cfg)
    scfg = SampleConfig.from_dict(cfg["samples"])

    start = pd.Timestamp(cfg["pilot"]["start"])
    end = pd.Timestamp(cfg["pilot"]["end"])
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "feasibility" / start.strftime("%Y-%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"pilot_start": str(start), "pilot_end_exclusive": str(end)}
    tic_all = time.time()

    # 1. SHARP keywords (include 12 h before start so early samples have full input windows)
    import drms
    client = drms.Client(email=email)
    lookback = pd.Timedelta(minutes=scfg.frame_step_minutes * (scfg.n_frames_full - 1))
    tic = time.time()
    sharp = query_sharp_keywords(client, cfg["jsoc"]["series"], start - lookback, end,
                                 chunk_days=cfg["jsoc"]["query_chunk_days"])
    report["sharp_query_seconds"] = round(time.time() - tic, 1)
    sharp.drop(columns=["noaa_set"]).assign(NOAA_ARS=sharp["NOAA_ARS"].astype(str)).to_csv(
        out_dir / "sharp_keywords.csv", index=False)
    in_month = sharp[(sharp["time"] >= start) & (sharp["time"] < end)]
    q = in_month["QUALITY"].value_counts()
    report["sharp"] = {
        "n_records": int(len(in_month)),
        "n_harps": int(in_month["HARPNUM"].nunique()),
        "n_harps_with_noaa": int(in_month.loc[in_month["noaa_set"].map(len) > 0, "HARPNUM"].nunique()),
        "cadence": cadence_stats(in_month),
        "quality_top_values_hex": {hex(int(k)): int(v) for k, v in q.head(10).items()},
        "share_quality_zero": round(float((in_month["QUALITY"] == 0).mean()), 4),
        "share_within_limb": round(float((in_month["LON_FWT"].abs() <= scfg.limb_max_deg).mean()), 4),
    }

    # 2. Flares (one day either side so forecast/persistence windows are covered)
    fl_start, fl_end = start - pd.Timedelta(days=1), end + pd.Timedelta(days=1)
    raw = fetch_hek_flares(fl_start, fl_end)
    report["flares_raw_by_frm_name"] = raw["frm_name"].value_counts().to_dict()
    flares = clean_flares(raw, cfg["flares"]["frm_name"])

    # 3. Association
    flares = associate_flares(flares, sharp, cfg["flares"]["association_window_hours"])
    flares.assign(candidates=flares["candidates"].map(lambda c: ",".join(map(str, c)))).to_csv(
        out_dir / "flares_associated.csv", index=False)
    fm = flares[(flares["start"] >= start) & (flares["start"] < end)]
    report["flares_in_month"] = {
        "by_class_letter": fm["class_letter"].value_counts().to_dict(),
        "association_by_status": fm["status"].value_counts().to_dict(),
        "association_geM_by_status": fm[fm["label"] >= 1]["status"].value_counts().to_dict(),
    }

    # 4. Samples (t restricted so [t+6h, t+24h) lies inside the flare query range)
    samples = build_samples(sharp, flares, scfg, t_min=start, t_max=end)
    samples.to_csv(out_dir / "samples.csv", index=False)
    inc = samples[samples["included"]]
    need = frames_needed(samples)
    need.to_csv(out_dir / "frames_needed.csv", index=False)
    mb_proc = len(need) * (cfg["preprocess"]["image_size"] ** 2) * 2 / 1e6
    report["samples"] = {
        "n_candidates": int(len(samples)),
        "n_included": int(len(inc)),
        "excluded_by_reason": samples.loc[~samples["included"], "excluded_reason"].value_counts().to_dict(),
        "included_class_counts": inc["label"].value_counts().sort_index().to_dict(),
        "included_persistence_class_counts": inc["persistence_label"].value_counts().sort_index().to_dict(),
        "n_substituted_distribution": inc["n_substituted"].value_counts().sort_index().to_dict(),
        "unique_frames_needed": int(len(need)),
        "processed_storage_mb_at_image_size": round(mb_proc, 1),
    }

    # 5. Download test
    if cfg["pilot"]["download_test"]["enabled"] and not args.no_download and len(inc):
        h = int(inc["harpnum"].value_counts().index[0])
        t0 = inc.loc[inc["harpnum"] == h, "t"].iloc[0]
        try:
            dl = test_export(client, cfg["jsoc"]["series"], cfg["jsoc"]["segment"], h, t0,
                             cfg["pilot"]["download_test"]["n_hours"],
                             Path(cfg["paths"]["data_dir"]) / "pilot_fits", email)
            dl["est_raw_download_gb_for_month"] = round(dl["mb_per_frame"] * len(need) / 1e3, 2)
            dl["est_hours_for_month_at_measured_rate"] = round(
                dl["seconds_per_frame_incl_staging"] * len(need) / 3600, 1)
            report["download_test"] = dl
        except Exception as e:  # report, do not hide
            log.error("Download test failed: %s", e)
            report["download_test"] = {"error": repr(e)}

    report["total_runtime_seconds"] = round(time.time() - tic_all, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n================ FEASIBILITY REPORT ================")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nOutputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
