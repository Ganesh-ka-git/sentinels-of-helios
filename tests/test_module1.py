"""Unit tests for Module 1 logic using synthetic SHARP records and flares (no network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.data.goes import class_to_label, clean_flares, normalise_noaa, parse_goes_class  # noqa: E402
from flarecast.data.jsoc import parse_noaa_ars, parse_quality  # noqa: E402
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


def test_parse_quality_hex_strings():
    assert parse_quality("0x00000000") == 0
    assert parse_quality("0x00000400") == 0x400
    assert parse_quality(0) == 0 and parse_quality("0") == 0
    assert parse_quality(np.nan) == -1 and parse_quality("garbage") == -1


class _FakeClient:
    """Mimics drms.Client.query(convert_numeric=False): every value comes back as a string, QUALITY in hex."""
    def keys(self, series):
        return ["HARPNUM", "T_REC", "NOAA_AR", "NOAA_NUM", "NOAA_ARS", "LON_FWT", "LAT_FWT", "QUALITY"]

    def query(self, q, key, convert_numeric=True):
        assert convert_numeric is False, "numeric conversion must be disabled (it destroys hex QUALITY)"
        return pd.DataFrame({
            "HARPNUM": ["11149", "11149"], "T_REC": ["2024.05.01_00:00:00_TAI", "2024.05.01_00:12:00_TAI"],
            "NOAA_AR": ["13664", "13664"], "NOAA_NUM": ["1", "1"], "NOAA_ARS": ["13664", "13664,13668"],
            "LON_FWT": ["-12.5", "-12.4"], "LAT_FWT": ["-18.1", "-18.1"], "QUALITY": ["0x00000000", "0x00000400"]})


def test_query_sharp_keywords_parses_strings():
    if "drms" not in sys.modules:
        try:
            import drms  # noqa: F401
        except ImportError:  # minimal stand-in so the test runs without drms installed
            import types
            stub = types.ModuleType("drms")
            stub.to_datetime = lambda s: pd.to_datetime(pd.Series(s).str.replace("_TAI", ""), format="%Y.%m.%d_%H:%M:%S")
            sys.modules["drms"] = stub
    from flarecast.data.jsoc import query_sharp_keywords
    df = query_sharp_keywords(_FakeClient(), "hmi.sharp_cea_720s", pd.Timestamp("2024-05-01"),
                              pd.Timestamp("2024-05-02"), chunk_days=7)
    assert df["QUALITY"].tolist() == [0, 0x400]
    assert df["HARPNUM"].dtype.kind == "i" and df["LON_FWT"].dtype.kind == "f"
    assert df["noaa_set"].iloc[1] == {13664, 13668}
    assert df["time"].iloc[0] == pd.Timestamp("2024-05-01 00:00")


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


# ---------------- download benchmark helpers ----------------
from flarecast.data.jsoc import build_record_sets, download_all, download_file, export_urls, local_filename  # noqa: E402


def test_build_record_sets_per_harp():
    frames = pd.DataFrame({"harpnum": [200, 100, 100, 100],
                           "time": pd.to_datetime(["2024-05-01 00:24", "2024-05-01 00:24", "2024-05-01 00:00",
                                                   "2024-05-01 00:24"])})
    sets = build_record_sets(frames, "hmi.sharp_cea_720s", "Br")
    assert sets == [
        "hmi.sharp_cea_720s[100][2024.05.01_00:00:00_TAI,2024.05.01_00:24:00_TAI]{Br}",
        "hmi.sharp_cea_720s[200][2024.05.01_00:24:00_TAI]{Br}",
    ]
    with pytest.raises(ValueError):
        build_record_sets(frames.iloc[0:0], "s", "Br")


class _Resp:
    def __init__(self, body: bytes, status: int = 200, length: int | None = None, fail_midway: bool = False):
        self.body, self.status, self.fail_midway = body, status, fail_midway
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def raise_for_status(self):
        import requests
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def iter_content(self, chunk_size=1):
        import requests
        yield self.body[: len(self.body) // 2]
        if self.fail_midway:
            raise requests.ConnectionError("dropped")
        yield self.body[len(self.body) // 2:]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Session:
    """Scripted session: `responses` is a list consumed by successive .get calls (callable or response)."""
    def __init__(self, responses, head_length=None):
        self.responses, self.head_length, self.gets, self.heads = list(responses), head_length, 0, 0

    def get(self, url, stream=True, timeout=None):
        self.gets += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def head(self, url, timeout=None, allow_redirects=True):
        self.heads += 1
        return _Resp(b"", length=self.head_length)


def test_download_file_retries_then_succeeds(tmp_path):
    import requests
    sleeps = []
    s = _Session([requests.Timeout("t"), _Resp(b"abcdef", fail_midway=True), _Resp(b"abcdef")])
    res = download_file("http://x/f.fits", tmp_path / "f.fits", s, timeout_s=60, retries=3, backoff_s=2.0,
                        sleep=sleeps.append)
    assert res["status"] == "ok" and res["attempts"] == 3 and res["bytes"] == 6
    assert sleeps == [2.0, 4.0]  # exponential backoff
    assert (tmp_path / "f.fits").read_bytes() == b"abcdef"
    assert not (tmp_path / "f.fits.part").exists()


def test_download_file_fails_loudly_after_retries(tmp_path):
    import requests
    s = _Session([requests.Timeout("t")] * 3)
    res = download_file("http://x/f.fits", tmp_path / "f.fits", s, timeout_s=60, retries=2, sleep=lambda _: None)
    assert res["status"] == "failed" and res["attempts"] == 3 and "Timeout" in res["error"]
    assert not (tmp_path / "f.fits").exists()


def test_download_file_rejects_truncated_body(tmp_path):
    s = _Session([_Resp(b"abcdef", length=10)])
    res = download_file("http://x/f.fits", tmp_path / "f.fits", s, timeout_s=60, retries=0, sleep=lambda _: None)
    assert res["status"] == "failed" and "incomplete" in res["error"]
    assert not (tmp_path / "f.fits").exists()


def test_download_file_skips_existing_with_correct_size(tmp_path):
    dest = tmp_path / "f.fits"
    dest.write_bytes(b"abcdef")
    s = _Session([], head_length=6)
    res = download_file("http://x/f.fits", dest, s, timeout_s=60, retries=3)
    assert res["status"] == "skipped" and s.gets == 0
    # wrong size on disk -> re-download
    dest.write_bytes(b"abc")
    s2 = _Session([_Resp(b"abcdef")], head_length=6)
    res2 = download_file("http://x/f.fits", dest, s2, timeout_s=60, retries=3)
    assert res2["status"] == "ok" and dest.read_bytes() == b"abcdef"


def test_download_all_parallel(tmp_path):
    items = [(f"f{i}.fits", f"http://x/{i}") for i in range(6)]

    class Factory:
        def __call__(self):
            class S:
                def get(self, url, stream=True, timeout=None):
                    return _Resp(url.encode())

                def head(self, *a, **k):
                    raise AssertionError("no HEAD expected for fresh files")
            return S()

    res = download_all(items, tmp_path, workers=4, timeout_s=60, retries=1, session_factory=Factory())
    assert [r["status"] for r in res] == ["ok"] * 6
    assert (tmp_path / "f3.fits").read_bytes() == b"http://x/3"


def test_local_filename_unique_per_record():
    a = local_filename("hmi.sharp_cea_720s[11117][2024.05.01_00:24:00_TAI]", "Br")
    b = local_filename("hmi.sharp_cea_720s[11117][2024.05.01_00:48:00_TAI]", "Br")
    assert a == "hmi.sharp_cea_720s.11117.20240501_002400_TAI.Br.fits" and a != b
    with pytest.raises(ValueError):
        local_filename("garbage", "Br")


class _FakeExportClient:
    def __init__(self, records):
        self.records = records

    def export(self, q, method, protocol, email):
        assert (method, protocol) == ("url_quick", "as-is")
        recs = self.records

        class Req:
            urls = pd.DataFrame({"record": recs, "filename": ["Br.fits"] * len(recs),  # as-is: all named Br.fits
                                 "url": [f"http://x/{i}/Br.fits" for i in range(len(recs))]})

            def wait(self):
                pass
        return Req()


def test_export_urls_gives_unique_names_despite_as_is_filenames():
    recs = [f"hmi.sharp_cea_720s[11117][2024.05.01_00:{m:02d}:00_TAI]" for m in (0, 24)]
    out = export_urls(_FakeExportClient(recs), ["q"], "a@b.c", "Br")
    assert out["filename"].is_unique and len(out) == 2
    with pytest.raises(ValueError, match="Duplicate"):
        export_urls(_FakeExportClient([recs[0], recs[0]]), ["q"], "a@b.c", "Br")
