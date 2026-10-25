"""Month-by-month sample index and HARP-grouped split (decisions.md §5, §10). Pure functions, no network."""
from __future__ import annotations

import logging

import pandas as pd

from ..data.goes import clean_flares
from .association import associate_flares
from .samples import SampleConfig, build_samples, frames_needed

log = logging.getLogger(__name__)


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """First instants of every calendar month in [start, end). start and end must be month starts."""
    for name, ts in (("start", start), ("end", end)):
        if not (ts.day == 1 and ts == ts.normalize()):
            raise ValueError(f"period {name} must be the first instant of a month, got {ts}")
    return list(pd.date_range(start, end, freq="MS", inclusive="left"))


def priority_order(months: list[pd.Timestamp], priority_years: list[int]) -> list[pd.Timestamp]:
    """Process months of priority_years first (in the given year order), then the rest in calendar order."""
    rank = {y: i for i, y in enumerate(priority_years)}
    return sorted(months, key=lambda m: (rank.get(m.year, len(rank)), m))


def next_month(ts: pd.Timestamp) -> pd.Timestamp:
    return ts + pd.offsets.MonthBegin(1)


def build_month(sharp: pd.DataFrame, raw_flares: pd.DataFrame, scfg: SampleConfig, frm_name: str,
                association_window_hours: float, month_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Associate flares with ALL loaded keyword records, then build samples with t in [month_start, month_end).

    Returns (samples with a 'month' column, associated flares).
    """
    month_end = next_month(month_start)
    flares = clean_flares(raw_flares, frm_name)
    flares = associate_flares(flares, sharp, association_window_hours)
    samples = build_samples(sharp, flares, scfg, t_min=month_start, t_max=month_end)
    if len(samples) and not ((samples["t"] >= month_start) & (samples["t"] < month_end)).all():
        raise AssertionError(f"samples with t outside [{month_start}, {month_end})")
    samples["month"] = month_start.strftime("%Y-%m")
    return samples, flares


def split_for_month_index(month_index: int, pattern: list[str]) -> str:
    return pattern[month_index % len(pattern)]


def assign_splits(first_record: pd.Series, period_start: pd.Timestamp, pattern: list[str]) -> pd.Series:
    """Split per HARP from the calendar month of its first loaded record.

    first_record : Series indexed by HARPNUM holding the timestamp of the HARP's first record.
    month_index  = months since period_start; HARPs first seen before period_start (inside the 12 h look-back of
    the first month) get month_index 0. Returns a Series indexed by HARPNUM with the split names.
    """
    idx = (first_record.dt.year - period_start.year) * 12 + (first_record.dt.month - period_start.month)
    n_clamped = int((idx < 0).sum())
    if n_clamped:
        log.info("%d HARPs first seen before the period start (look-back) -> month_index 0", n_clamped)
    idx = idx.clip(lower=0)
    out = idx.map(lambda i: split_for_month_index(int(i), pattern))
    out.name = "split"
    return out


def frames_needed_all(samples_by_month: dict[str, pd.DataFrame], split_by_harp: pd.Series) -> pd.DataFrame:
    """De-duplicated (harpnum, time, month, split); month = calendar month of the frame time."""
    parts = [frames_needed(s) for s in samples_by_month.values() if s["included"].any()]
    if not parts:
        return pd.DataFrame(columns=["harpnum", "time", "month", "split"])
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["harpnum", "time"]).sort_values(["harpnum", "time"])
    df["month"] = df["time"].dt.strftime("%Y-%m")
    df["split"] = df["harpnum"].map(split_by_harp)
    return df.reset_index(drop=True)


def build_report(samples: pd.DataFrame, frames: pd.DataFrame, rates_files_per_s=(0.5, 1.0, 1.9)) -> dict:
    """Summaries for index_report.json. samples must have a 'split' column."""
    inc = samples[samples["included"]]
    per_split = {}
    for sp in ("train", "val", "test"):
        a, b = samples[samples["split"] == sp], inc[inc["split"] == sp]
        per_split[sp] = {
            "n_candidates": int(len(a)),
            "n_included": int(len(b)),
            "n_harps_included": int(b["harpnum"].nunique()),
            "class_counts": {str(k): int(v) for k, v in b["label"].value_counts().reindex([0, 1, 2], fill_value=0).items()},
            "persistence_class_counts": {str(k): int(v) for k, v in
                                         b["persistence_label"].value_counts().reindex([0, 1, 2], fill_value=0).items()},
        }
    exc = samples[~samples["included"]]
    n_frames = int(len(frames))
    report = {
        "n_candidates": int(len(samples)),
        "n_included": int(len(inc)),
        "per_split": per_split,
        "excluded_by_reason_total": exc["excluded_reason"].value_counts().to_dict(),
        "excluded_by_reason_per_month": {m: g["excluded_reason"].value_counts().to_dict()
                                         for m, g in exc.groupby("month")},
        "frames_needed_per_month": frames.groupby("month").size().to_dict() if n_frames else {},
        "frames_needed_total": n_frames,
        "estimated_download_hours": {f"{r}_files_per_s": round(n_frames / r / 3600, 1) for r in rates_files_per_s},
        "warnings": [],
    }
    n_test_x = per_split["test"]["class_counts"]["2"]
    if n_test_x < 10:
        report["warnings"].append(
            f"TEST split has only {n_test_x} X-class samples (< 10): per decisions.md §11 report X descriptively only.")
    return report


# ---------------- catalogue-completeness rule and final split (decisions.md §10) ----------------
def completeness_table(flares_by_month: dict[str, pd.DataFrame], samples_by_month: dict[str, pd.DataFrame],
                       min_share: float) -> pd.DataFrame:
    """Per month: share of >=M flares that carry a NOAA number, kept flag, included samples and class counts.

    flares_by_month : month label -> flares_associated table (columns start, label, noaa); only flares with start
                      inside the calendar month are counted (the cached tables also hold margin days).
    A month with no >=M flares has an undefined share and is NOT kept (flagged in column 'note').
    """
    rows = []
    for label in sorted(flares_by_month):
        ms = pd.Timestamp(label + "-01")
        f = flares_by_month[label]
        f = f[(f["start"] >= ms) & (f["start"] < next_month(ms)) & (f["label"] >= 1)]
        n = int(len(f))
        share = float((f["noaa"] > 0).mean()) if n else float("nan")
        kept = bool(n > 0 and share >= min_share)
        s = samples_by_month[label]
        inc = s[s["included"]]
        c = inc["label"].value_counts().reindex([0, 1, 2], fill_value=0)
        rows.append({"month": label, "n_geM_flares": n, "share_with_noaa": round(share, 4) if n else None,
                     "kept": kept, "included_samples": int(len(inc)),
                     "class0": int(c[0]), "class1": int(c[1]), "class2": int(c[2]),
                     "note": "" if n else "no >=M flares: share undefined, not kept"})
    return pd.DataFrame(rows)


def check_expected_kept(table: pd.DataFrame, first: str, last: str) -> list[str]:
    """Differences between the kept months and the expected contiguous range first..last (empty list = as expected)."""
    expected = {m.strftime("%Y-%m") for m in pd.date_range(first + "-01", last + "-01", freq="MS")}
    kept = set(table.loc[table["kept"], "month"])
    problems = [f"{m} expected kept but FAILS the rule" for m in sorted(expected - kept) if m in set(table["month"])]
    problems += [f"{m} expected kept but is not indexed" for m in sorted(expected - set(table["month"]))]
    problems += [f"{m} passes the rule but was not expected to" for m in sorted(kept - expected)]
    return problems


def rebuild_split(kept_months: list[str], first_record: pd.Series, pattern: list[str]) -> pd.Series:
    """Split per HARP over the kept period only.

    kept_months must be contiguous calendar months; month_index counts from the first kept month, and HARPs whose first
    loaded record lies before it (alive at the start of the kept period) are assigned to the first kept month.
    first_record must be computed from the kept months' keyword records only.
    """
    ms = pd.DatetimeIndex([pd.Timestamp(m + "-01") for m in sorted(kept_months)])
    if len(ms) == 0:
        raise ValueError("No kept months.")
    if not (ms == pd.date_range(ms[0], periods=len(ms), freq="MS")).all():
        raise ValueError(f"Kept months are not contiguous: {[m.strftime('%Y-%m') for m in ms]}")
    return assign_splits(first_record, ms[0], pattern)
