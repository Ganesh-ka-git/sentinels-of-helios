"""Unit tests for Module 1 logic using synthetic SHARP records and flares (no network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.data.goes import class_to_label, clean_flares, normalise_noaa, parse_goes_class  # noqa: E402
from flarecast.data.jsoc import parse_noaa_ars  # noqa: E402
from flarecast.labels.association import (AMBIGUOUS, ASSOCIATED, NO_AR, NO_HARP,  # noqa: E402
                                          associate_flares)
from flarecast.labels.samples import SampleConfig, build_samples, nominal_frame_times  # noqa: E402

T = pd.Timestamp("2024-05-10 12:00")
CFG = SampleConfig()


def make_sharp(harp=100, noaa=(13664,), start=T - pd.Timedelta(hours=14), end=T + pd.Timedelta(hours=26),
               lon=0.0, drop=(), quality=None):
    times = pd.date_range(start, end, freq="12min")
    df = pd.DataFrame({"HARPNUM": harp, "time": times, "LON_FWT": lon, "QUALITY": 0,
                       "noaa_set": [frozenset(noaa)] * len(times)})
    df = df[~df["time"].isin(list(drop))]
    if quality:
        for ts, qv in quality.items():
            df.loc[df["time"] == ts, "QUALITY"] = qv
    return df.reset_index(drop=True)


def make_flares(rows):
    df = pd.DataFrame(rows, columns=["start", "goes_class", "ar_raw"])
    df["peak"] = df["start"]; df["end"] = df["start"]; df["frm_name"] = "SWPC"
    return clean_flares(df, "SWPC")


# ---------------- parsing ----------------
@pytest.mark.parametrize("s,letter,flux", [("M1.2", "M", 1.2e-5), ("X10", "X", 1e-3), ("c3.4", "C", 3.4e-6)])
def test_parse_goes_class(s, letter, flux):
    l, f = parse_goes_class(s)
    assert l == letter and np.isclose(f, flux)


def test_parse_bad_class():
    letter, flux = parse_goes_class("")
    assert letter is None and np.isnan(flux)


def test_class_to_label():
    assert [class_to_label(c) for c in "ABCMX"] == [0, 0, 0, 1, 2]


def test_normalise_noaa():
    assert normalise_noaa(3664) == 13664 and normalise_noaa(13664) == 13664
    assert normalise_noaa(0) == 0 and normalise_noaa(None) == 0 and normalise_noaa("nan") == 0


def test_parse_noaa_ars():
    assert parse_noaa_ars("13664,13668") == {13664, 13668}
    assert parse_noaa_ars("3664") == {13664}
    assert parse_noaa_ars(np.nan) == frozenset()


# ---------------- association ----------------
def test_association_statuses():
    sharp = pd.concat([make_sharp(100, (13664,)), make_sharp(200, (13670,)), make_sharp(300, (13670,))])
    fl = make_flares([(T, "M1.0", 3664), (T, "C1.0", 0), (T, "X1.0", 13999), (T, "M2.0", 3670)])
    a = associate_flares(fl, sharp).set_index("goes_class")
    assert a.loc["M1.0", "status"] == ASSOCIATED and a.loc["M1.0", "harpnum"] == 100
    assert a.loc["C1.0", "status"] == NO_AR
    assert a.loc["X1.0", "status"] == NO_HARP
    assert a.loc["M2.0", "status"] == AMBIGUOUS


# ---------------- labels & windows ----------------
def _label_for(flare_offset_hours, cls="M1.0"):
    sharp = make_sharp()
    fl = associate_flares(make_flares([(T + pd.Timedelta(hours=flare_offset_hours), cls, 13664)]), sharp)
    s = build_samples(sharp, fl, CFG)
    return s.set_index("t").loc[T]


def test_label_window_boundaries():
    assert _label_for(6)["label"] == 1                 # t+6h inclusive
    assert _label_for(23.99)["label"] == 1
    assert _label_for(24)["label"] == 0                # t+24h exclusive
    assert _label_for(5.99)["label"] == 0              # before window
    assert _label_for(10, "X2.0")["label"] == 2


def test_max_class_wins():
    sharp = make_sharp()
    fl = associate_flares(make_flares([(T + pd.Timedelta(hours=8), "M5.0", 13664),
                                       (T + pd.Timedelta(hours=9), "X1.1", 13664)]), sharp)
    row = build_samples(sharp, fl, CFG).set_index("t").loc[T]
    assert row["label"] == 2 and row["max_class"] == "X1.1"


def test_persistence_label():
    row = _label_for(-3)
    assert row["persistence_label"] == 1 and row["label"] == 0


def test_unassociated_m_excludes():
    sharp = make_sharp()
    fl = associate_flares(make_flares([(T + pd.Timedelta(hours=8), "M1.0", 0)]), sharp)
    row = build_samples(sharp, fl, CFG).set_index("t").loc[T]
    assert not row["included"] and row["excluded_reason"] == "unassociated_geM_in_window"


def test_unassociated_c_does_not_exclude():
    sharp = make_sharp()
    fl = associate_flares(make_flares([(T + pd.Timedelta(hours=8), "C9.0", 0)]), sharp)
    assert build_samples(sharp, fl, CFG).set_index("t").loc[T, "included"]


# ---------------- temporal ordering & missing frames ----------------
def test_frame_times_never_after_t():
    times = nominal_frame_times(T, CFG)
    assert len(times) == 31 and times[-1] == T and times[0] == T - pd.Timedelta(hours=12)
    assert all(a < b for a, b in zip(times, times[1:]))


def test_substitution_uses_earlier_frame():
    missing = T - pd.Timedelta(minutes=48)
    sharp = make_sharp(drop=[missing])
    fl = associate_flares(make_flares([(T + pd.Timedelta(days=5), "C1.0", 13664)]), sharp)
    row = build_samples(sharp, fl, CFG).set_index("t").loc[T]
    assert row["included"] and row["n_substituted"] == 1
    used = pd.to_datetime(row["frames"].split(";"), format="%Y%m%dT%H%M")
    assert (missing - pd.Timedelta(minutes=12)) in set(used) and all(u <= T for u in used)


def test_too_many_substitutions():
    drops = [T - pd.Timedelta(minutes=24 * k) for k in (1, 3, 5, 7)]
    drops += [d - pd.Timedelta(minutes=12) for d in drops]  # remove neighbours too -> still fillable from earlier
    sharp = make_sharp(drop=drops)
    fl = associate_flares(make_flares([(T + pd.Timedelta(days=5), "C1.0", 13664)]), sharp)
    row = build_samples(sharp, fl, CFG).set_index("t").loc[T]
    assert not row["included"] and row["excluded_reason"] == "too_many_substitutions"


def test_frame_t_required():
    sharp = make_sharp(drop=[T])
    fl = associate_flares(make_flares([(T + pd.Timedelta(days=5), "C1.0", 13664)]), sharp)
    s = build_samples(sharp, fl, CFG)
    assert T not in set(s["t"])  # no record at t -> no candidate at all


def test_bad_quality_at_t_excluded():
    sharp = make_sharp(quality={T: 0x400})
    fl = associate_flares(make_flares([(T + pd.Timedelta(days=5), "C1.0", 13664)]), sharp)
    row = build_samples(sharp, fl, CFG).set_index("t").loc[T]
    assert row["excluded_reason"] == "frame_t_missing_or_bad"


def test_limb_excluded():
    sharp = make_sharp(lon=70.0)
    fl = associate_flares(make_flares([(T + pd.Timedelta(days=5), "C1.0", 13664)]), sharp)
    assert build_samples(sharp, fl, CFG).set_index("t").loc[T, "excluded_reason"] == "limb"


def test_no_noaa_excluded():
    sharp = make_sharp(noaa=())
    fl = associate_flares(make_flares([(T + pd.Timedelta(days=5), "C1.0", 13664)]), sharp)
    assert build_samples(sharp, fl, CFG).set_index("t").loc[T, "excluded_reason"] == "no_noaa_region"
