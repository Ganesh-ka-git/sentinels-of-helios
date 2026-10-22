"""Associate GOES flares with SHARP HARPs through NOAA active-region numbers (decisions.md §5)."""
from __future__ import annotations

import pandas as pd

ASSOCIATED = "associated"
NO_AR = "unassociated_no_ar"
NO_HARP = "unassociated_no_harp"
AMBIGUOUS = "ambiguous"


def explode_noaa(sharp: pd.DataFrame) -> pd.DataFrame:
    """One row per (HARPNUM, time, noaa) from the per-record noaa_set column."""
    rows = sharp[["HARPNUM", "time", "noaa_set"]].copy()
    rows["noaa"] = rows["noaa_set"].map(list)
    rows = rows.explode("noaa").dropna(subset=["noaa"])
    rows["noaa"] = rows["noaa"].astype(int)
    return rows[["HARPNUM", "time", "noaa"]].reset_index(drop=True)


def associate_flares(flares: pd.DataFrame, sharp: pd.DataFrame, window_hours: float = 1.0) -> pd.DataFrame:
    """Assign each flare to a HARP.

    A flare is associated with a HARP if its (normalised) NOAA number appears in that HARP's NOAA_ARS in any record
    within ±window_hours of the flare start. Returns the flare table with columns:
      status  : associated | unassociated_no_ar | unassociated_no_harp | ambiguous
      harpnum : the HARP for associated flares, else -1
      candidates : tuple of all matching HARPs (len > 1 for ambiguous)
    """
    exploded = explode_noaa(sharp)
    by_noaa = {k: g for k, g in exploded.groupby("noaa")}
    w = pd.Timedelta(hours=window_hours)

    status, harp, cands = [], [], []
    for _, f in flares.iterrows():
        if f["noaa"] <= 0:
            status.append(NO_AR); harp.append(-1); cands.append(())
            continue
        g = by_noaa.get(int(f["noaa"]))
        if g is None:
            status.append(NO_HARP); harp.append(-1); cands.append(())
            continue
        near = g[(g["time"] >= f["start"] - w) & (g["time"] <= f["start"] + w)]
        hs = tuple(sorted(near["HARPNUM"].unique().tolist()))
        if len(hs) == 0:
            status.append(NO_HARP); harp.append(-1)
        elif len(hs) == 1:
            status.append(ASSOCIATED); harp.append(hs[0])
        else:
            status.append(AMBIGUOUS); harp.append(-1)
        cands.append(hs)
    out = flares.copy()
    out["status"] = status
    out["harpnum"] = harp
    out["candidates"] = cands
    return out
