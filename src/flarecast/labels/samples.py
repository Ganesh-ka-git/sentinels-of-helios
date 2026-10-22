"""Build the (HARP, t) sample index: input frame lists, labels and exclusions (decisions.md §4, §5, §7)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .association import AMBIGUOUS, ASSOCIATED, NO_AR, NO_HARP

CADENCE = pd.Timedelta(minutes=12)


@dataclass(frozen=True)
class SampleConfig:
    grid_hours: int = 6
    forecast_start_hours: float = 6
    forecast_end_hours: float = 24
    persistence_hours: float = 24
    frame_step_minutes: int = 24
    n_frames_full: int = 31
    max_substitutions: int = 3
    limb_max_deg: float = 68.0
    quality_good_values: tuple = (0,)
    require_noaa: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "SampleConfig":
        d = dict(d)
        d["quality_good_values"] = tuple(d.get("quality_good_values", (0,)))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def nominal_frame_times(t: pd.Timestamp, cfg: SampleConfig) -> list[pd.Timestamp]:
    """Nominal input times t - k*step for k = n-1..0, oldest first; never later than t."""
    step = pd.Timedelta(minutes=cfg.frame_step_minutes)
    return [t - k * step for k in range(cfg.n_frames_full - 1, -1, -1)]


def _resolve_frames(t, recs: dict, cfg: SampleConfig):
    """Return (used_times, n_substituted, reason). reason is None when the sample is usable.

    recs maps time -> (lon, quality) for one HARP. Only records <= t inside the input window are ever used.
    """
    nominal = nominal_frame_times(t, cfg)
    window_start = nominal[0]
    good_q = set(cfg.quality_good_values)

    def is_good(ts):
        r = recs.get(ts)
        return (r is not None and r[1] in good_q
                and not np.isnan(r[0]) and abs(r[0]) <= cfg.limb_max_deg)

    # limb rule applies to every existing frame on the nominal grid
    for ts in nominal:
        r = recs.get(ts)
        if r is not None and (np.isnan(r[0]) or abs(r[0]) > cfg.limb_max_deg):
            return None, 0, "limb"
    if not is_good(t):
        return None, 0, "frame_t_missing_or_bad"

    used, n_sub = [], 0
    for ts in nominal:
        if is_good(ts):
            used.append(ts)
            continue
        # nearest earlier good record within the window (12-min steps)
        cand = ts - CADENCE
        while cand >= window_start and not is_good(cand):
            cand -= CADENCE
        if cand < window_start:
            return None, n_sub, "unfillable_gap"
        used.append(cand)
        n_sub += 1
        if n_sub > cfg.max_substitutions:
            return None, n_sub, "too_many_substitutions"
    assert all(u <= t for u in used), "future frame in input window"
    return used, n_sub, None


def build_samples(sharp: pd.DataFrame, flares: pd.DataFrame, cfg: SampleConfig,
                  t_min: pd.Timestamp | None = None, t_max: pd.Timestamp | None = None) -> pd.DataFrame:
    """Create one row per candidate (HARP, t) with its label or exclusion reason.

    sharp  : output of query_sharp_keywords (HARPNUM, time, noaa_set, LON_FWT, QUALITY)
    flares : output of associate_flares (start, label, status, harpnum, candidates)
    Samples with t outside [t_min, t_max) are not created (so forecast windows stay inside the flare query range).
    """
    fs = pd.Timedelta(hours=cfg.forecast_start_hours)
    fe = pd.Timedelta(hours=cfg.forecast_end_hours)
    ph = pd.Timedelta(hours=cfg.persistence_hours)

    assoc = flares[flares["status"] == ASSOCIATED]
    unassoc_m = flares[flares["status"].isin([NO_AR, NO_HARP]) & (flares["label"] >= 1)]
    ambig = flares[flares["status"] == AMBIGUOUS]
    assoc_by_harp = {h: g for h, g in assoc.groupby("harpnum")}

    rows = []
    for harp, g in sharp.groupby("HARPNUM"):
        recs = {ts: (lon, q) for ts, lon, q in zip(g["time"], g["LON_FWT"], g["QUALITY"])}
        noaa_at = dict(zip(g["time"], g["noaa_set"]))
        grid = g["time"][(g["time"].dt.hour % cfg.grid_hours == 0) & (g["time"].dt.minute == 0)
                         & (g["time"].dt.second == 0)]
        hf = assoc_by_harp.get(harp, assoc.iloc[0:0])
        for t in grid:
            if t_min is not None and t < t_min:
                continue
            if t_max is not None and t >= t_max:
                continue
            row = {"harpnum": int(harp), "t": t, "n_noaa": len(noaa_at[t])}
            w0, w1 = t + fs, t + fe
            in_w = hf[(hf["start"] >= w0) & (hf["start"] < w1)]
            row["label"] = int(in_w["label"].max()) if len(in_w) else 0
            row["max_class"] = in_w.loc[in_w["peak_flux"].idxmax(), "goes_class"] if len(in_w) else ""
            past = hf[(hf["start"] >= t - ph) & (hf["start"] < t)]
            row["persistence_label"] = int(past["label"].max()) if len(past) else 0

            reason = None
            if cfg.require_noaa and len(noaa_at[t]) == 0:
                reason = "no_noaa_region"
            if reason is None:
                used, n_sub, reason = _resolve_frames(t, recs, cfg)
                row["n_substituted"] = n_sub
                if reason is None:
                    row["frames"] = ";".join(u.strftime("%Y%m%dT%H%M") for u in used)
            if reason is None:
                um = unassoc_m[(unassoc_m["start"] >= w0) & (unassoc_m["start"] < w1)]
                if len(um):
                    reason = "unassociated_geM_in_window"
            if reason is None:
                am = ambig[(ambig["start"] >= w0) & (ambig["start"] < w1)]
                if any(harp in c for c in am["candidates"]):
                    reason = "ambiguous_flare_in_window"
            row["excluded_reason"] = reason or ""
            row["included"] = reason is None
            rows.append(row)
    cols = ["harpnum", "t", "n_noaa", "label", "max_class", "persistence_label", "n_substituted",
            "included", "excluded_reason", "frames"]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out:
            out[c] = np.nan
    return out[cols].sort_values(["t", "harpnum"]).reset_index(drop=True)


def frames_needed(samples: pd.DataFrame) -> pd.DataFrame:
    """Unique (HARPNUM, time) records required by included samples — the download list."""
    inc = samples[samples["included"]]
    pairs = {(h, ts) for h, fr in zip(inc["harpnum"], inc["frames"]) for ts in fr.split(";")}
    df = pd.DataFrame(sorted(pairs), columns=["harpnum", "time"])
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%dT%H%M")
    return df
