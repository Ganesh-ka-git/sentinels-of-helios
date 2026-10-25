"""Unit tests for Module 2 (month index + split), synthetic data, no network."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.labels.index import (assign_splits, build_month, build_report, check_expected_kept,  # noqa: E402
                                    completeness_table, frames_needed_all, month_starts, priority_order,
                                    rebuild_split, split_for_month_index)
from flarecast.labels.samples import SampleConfig  # noqa: E402

PATTERN = ["train", "train", "train", "val", "test"]
SCFG = SampleConfig()
MS = pd.Timestamp("2024-05-01")
ME = pd.Timestamp("2024-06-01")


def make_sharp(harp, start, end, noaa=(13664,)):
    times = pd.date_range(start, end, freq="12min")
    return pd.DataFrame({"HARPNUM": harp, "time": times, "LON_FWT": 0.0, "QUALITY": 0,
                         "noaa_set": [frozenset(noaa)] * len(times)})


def raw_flares(rows):
    df = pd.DataFrame(rows, columns=["start", "goes_class", "ar_raw"])
    df["peak"] = df["start"]; df["end"] = df["start"]; df["frm_name"] = "SWPC"
    return df


def test_pattern_cycles_through_blocks():
    got = [split_for_month_index(i, PATTERN) for i in range(12)]
    assert got == ["train", "train", "train", "val", "test"] * 2 + ["train", "train"]


def test_priority_order_2024_then_2023_then_rest():
    ms = month_starts(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-02-01"))
    got = [m.strftime("%Y-%m") for m in priority_order(ms, [2024, 2023])]
    assert got[:12] == [f"2024-{m:02d}" for m in range(1, 13)]
    assert got[12:24] == [f"2023-{m:02d}" for m in range(1, 13)]
    assert got[24:] == ["2025-01"] and len(got) == 25


def test_month_starts_count_for_period():
    ms = month_starts(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-02-01"))
    assert len(ms) == 25 and ms[0] == pd.Timestamp("2023-01-01") and ms[-1] == pd.Timestamp("2025-01-01")
    with pytest.raises(ValueError):
        month_starts(pd.Timestamp("2023-01-15"), pd.Timestamp("2025-02-01"))


def test_assign_splits_by_first_record_month_and_one_split_per_harp():
    ps = pd.Timestamp("2023-01-01")
    first = pd.Series({1: pd.Timestamp("2023-01-31 18:00"), 2: pd.Timestamp("2023-02-01 00:00"),
                       3: pd.Timestamp("2023-04-10"), 4: pd.Timestamp("2023-05-01"),
                       5: pd.Timestamp("2023-06-15"), 6: pd.Timestamp("2022-12-31 12:00")})
    got = assign_splits(first, ps, PATTERN)
    assert got.loc[[1, 2, 3, 4, 5]].tolist() == ["train", "train", "val", "test", "train"]
    assert got.loc[6] == "train"  # look-back record before period start -> month_index 0
    assert got.index.is_unique  # a HARP maps to exactly one split


def test_harp_never_in_two_splits_across_months():
    # one long-lived HARP spanning two months gets a single split on all its samples
    ps = pd.Timestamp("2024-05-01")
    s_may, _ = build_month(make_sharp(100, "2024-04-30 12:00", "2024-06-02"),
                           raw_flares([(pd.Timestamp("2024-05-10"), "C1.0", 3664)]), SCFG, "SWPC", 1.0, MS)
    s_jun, _ = build_month(make_sharp(100, "2024-05-31 12:00", "2024-07-02"),
                           raw_flares([(pd.Timestamp("2024-06-10"), "C1.0", 3664)]), SCFG, "SWPC", 1.0, ME)
    allsamp = pd.concat([s_may, s_jun], ignore_index=True)
    first = pd.Series({100: pd.Timestamp("2024-04-30 12:00")})
    allsamp["split"] = allsamp["harpnum"].map(assign_splits(first, ps, PATTERN))
    assert allsamp.groupby("harpnum")["split"].nunique().max() == 1
    assert set(allsamp["month"]) == {"2024-05", "2024-06"}


def test_flare_after_month_end_uses_lookahead_records():
    t = ME - pd.Timedelta(hours=6)  # last grid time of May; window [t+6h, t+24h) = [Jun 1 00:00, Jun 1 18:00)
    flare = raw_flares([(ME + pd.Timedelta(hours=2), "M2.0", 3664)])
    with_look = make_sharp(100, MS - pd.Timedelta(hours=12), ME + pd.Timedelta(days=1))
    s, fl = build_month(with_look, flare, SCFG, "SWPC", 1.0, MS)
    row = s[s["t"] == t].iloc[0]
    assert fl.iloc[0]["status"] == "associated"
    assert row["included"] and row["label"] == 1

    # control: without look-ahead records the same flare cannot be associated -> sample excluded
    truncated = make_sharp(100, MS - pd.Timedelta(hours=12), ME - pd.Timedelta(minutes=12))
    s2, fl2 = build_month(truncated, flare, SCFG, "SWPC", 1.0, MS)
    assert fl2.iloc[0]["status"] == "unassociated_no_harp"
    assert s2[s2["t"] == t].iloc[0]["excluded_reason"] == "unassociated_geM_in_window"


def test_no_sample_outside_its_month():
    s, _ = build_month(make_sharp(100, MS - pd.Timedelta(hours=12), ME + pd.Timedelta(days=1)),
                       raw_flares([(pd.Timestamp("2024-05-10"), "C1.0", 3664)]), SCFG, "SWPC", 1.0, MS)
    assert len(s) > 0 and (s["t"] >= MS).all() and (s["t"] < ME).all()
    assert (s["month"] == "2024-05").all()


def test_frames_dedup_and_report_x_warning():
    s, _ = build_month(make_sharp(100, MS - pd.Timedelta(hours=12), ME + pd.Timedelta(days=1)),
                       raw_flares([(pd.Timestamp("2024-05-10"), "C1.0", 3664)]), SCFG, "SWPC", 1.0, MS)
    split = pd.Series({100: "test"})
    frames = frames_needed_all({"2024-05": s, "again": s}, split)
    assert not frames.duplicated(["harpnum", "time"]).any() and (frames["split"] == "test").all()
    s["split"] = "test"
    rep = build_report(s, frames)
    assert rep["per_split"]["test"]["n_included"] == int(s["included"].sum())
    assert any("X-class" in w for w in rep["warnings"])
    assert rep["estimated_download_hours"]["1.0_files_per_s"] == round(len(frames) / 3600, 1)


def test_hek_timeout_is_applied_and_restored(monkeypatch):
    import socket

    from sunpy.net import Fido

    from flarecast.data.goes import fetch_hek_flares

    seen = {}

    def fake_search(*a, **k):
        seen["timeout"] = socket.getdefaulttimeout()
        raise socket.timeout("stalled")

    monkeypatch.setattr(Fido, "search", fake_search)
    before = socket.getdefaulttimeout()
    with pytest.raises(socket.timeout):
        fetch_hek_flares(pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01"), timeout_s=123)
    assert seen["timeout"] == 123 and socket.getdefaulttimeout() == before  # applied during, restored after


# ---------------- completeness rule and final split ----------------
def _flares(month, noaas, label=1):
    ms = pd.Timestamp(month + "-01")
    return pd.DataFrame({"start": [ms + pd.Timedelta(days=2 + i) for i in range(len(noaas))],
                         "label": label, "noaa": noaas})


def _samples(n_inc=3):
    return pd.DataFrame({"included": [True] * n_inc + [False], "label": [0, 1, 2][:n_inc] + [0]})


def test_completeness_threshold_boundary_and_counts():
    fl = {"2023-10": _flares("2023-10", [0, 0, 13451, 0, 0]),              # 20%  -> dropped
          "2024-03": _flares("2024-03", [13500] * 4 + [0]),                # exactly 80% -> kept
          "2024-04": _flares("2024-04", [13500] * 3 + [0, 0]),             # 60%  -> dropped
          "2024-05": _flares("2024-05", [13500, 13501])}                   # 100% -> kept
    sm = {m: _samples() for m in fl}
    t = completeness_table(fl, sm, 0.8).set_index("month")
    assert t["kept"].to_dict() == {"2023-10": False, "2024-03": True, "2024-04": False, "2024-05": True}
    assert t.loc["2024-03", "share_with_noaa"] == 0.8 and t.loc["2024-03", "n_geM_flares"] == 5
    assert t.loc["2024-05", ["included_samples", "class0", "class1", "class2"]].tolist() == [3, 1, 1, 1]


def test_completeness_ignores_margin_days_and_sub_m_flares():
    f = pd.concat([_flares("2024-03", [13500, 13500]),
                   pd.DataFrame({"start": [pd.Timestamp("2024-02-29 12:00"), pd.Timestamp("2024-04-01 06:00")],
                                 "label": 1, "noaa": 0}),                       # margin-day >=M flares without NOAA
                   _flares("2024-03", [0] * 5, label=0)], ignore_index=True)    # C-class flares without NOAA
    t = completeness_table({"2024-03": f}, {"2024-03": _samples()}, 0.8)
    assert t.loc[0, "n_geM_flares"] == 2 and t.loc[0, "kept"]


def test_completeness_month_without_geM_flares_is_not_kept():
    t = completeness_table({"2024-03": _flares("2024-03", [0], label=0)}, {"2024-03": _samples()}, 0.8)
    assert not t.loc[0, "kept"] and "undefined" in t.loc[0, "note"]


def test_check_expected_kept_flags_unexpected_months():
    tab = pd.DataFrame({"month": ["2023-12", "2024-01", "2024-02"], "kept": [True, True, False]})
    problems = check_expected_kept(tab, "2024-01", "2024-02")
    assert any("2023-12 passes" in p for p in problems) and any("2024-02 expected kept but FAILS" in p for p in problems)
    ok = pd.DataFrame({"month": ["2023-12", "2024-01", "2024-02"], "kept": [False, True, True]})
    assert check_expected_kept(ok, "2024-01", "2024-02") == []


def test_rebuild_split_indexes_from_first_kept_month():
    kept = [f"2024-{m:02d}" for m in range(1, 8)] 
    first = pd.Series({1: pd.Timestamp("2023-11-20"),       # alive before the kept period -> month_index 0
                       2: pd.Timestamp("2023-12-31 12:00"),  # look-back record -> month_index 0
                       3: pd.Timestamp("2024-01-15"), 4: pd.Timestamp("2024-04-02"),   # index 0 and 3
                       5: pd.Timestamp("2024-05-31"), 6: pd.Timestamp("2024-06-01"),   # index 4 and 5
                       7: pd.Timestamp("2024-07-20")})                                 # index 6
    got = rebuild_split(kept, first, PATTERN)
    assert got.tolist() == ["train", "train", "train", "val", "test", "train", "train"]
    # same HARP, different period start -> pattern is anchored at the first kept month, not at 2023-01
    assert rebuild_split(kept[1:], first, PATTERN).loc[4] == "train"  # 2024-04 is index 2 from 2024-02


def test_rebuild_split_rejects_gaps_and_empty():
    first = pd.Series({1: pd.Timestamp("2024-01-15")})
    with pytest.raises(ValueError, match="contiguous"):
        rebuild_split(["2024-01", "2024-03"], first, PATTERN)
    with pytest.raises(ValueError):
        rebuild_split([], first, PATTERN)
