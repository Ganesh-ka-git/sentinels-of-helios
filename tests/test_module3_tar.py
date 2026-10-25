"""Unit tests for the Module 3 url-tar method: member mapping, extraction, cleanup, resume. Mocked network."""
import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_module3 import SEG, SERIES, _frames, _write_fits  # noqa: E402
from flarecast.data.jsoc import local_filename  # noqa: E402
from flarecast.preprocess.frames import Manifest, frame_path  # noqa: E402
from flarecast.preprocess.pipeline import (PreprocessSettings, Progress, TarMappingError, extract_tar,  # noqa: E402
                                           map_tar_members, pending_frames, process_frames_tar)

# Member names observed in a real JSOC url-tar as-is export (2025-06, HARP 13273, one record).
META = ["./JSOC_20261024_001114.drmsrun", "./JSOC_20261024_001114.qsub", "./index.txt",
        "./JSOC_20261024_001114.keywords.txt", "./index.html", "./index.json"]


def _fname(harp, t):
    return local_filename(f"{SERIES}[{harp}][{t.strftime('%Y.%m.%d_%H:%M:%S')}_TAI]", SEG)


def _make_tar(path: Path, harp, times, extra_members=(), skip=()):
    """Write a tar shaped like JSOC's (members './<name>'), with synthetic FITS data for each time."""
    with tarfile.open(path, "w") as tf:
        for m in META:
            ti = tarfile.TarInfo(m)
            ti.size = 3
            tf.addfile(ti, io.BytesIO(b"abc"))
        for t in times:
            if t in skip:
                continue
            fp = path.parent / ("_src_" + _fname(harp, t))
            _write_fits(fp)
            tf.add(fp, arcname="./" + _fname(harp, t))
            fp.unlink()
        for name in extra_members:
            ti = tarfile.TarInfo(name)
            ti.size = 3
            tf.addfile(ti, io.BytesIO(b"abc"))


# ---------------- member name -> (harpnum, time) ----------------
def test_map_tar_members_maps_fits_and_ignores_metadata():
    ts = list(pd.date_range("2025-06-01", periods=3, freq="24min"))
    names = {_fname(7, t) for t in ts}
    mapping = map_tar_members(META + ["./" + n for n in sorted(names)], names)
    assert set(mapping) == names and all(mapping[n] == "./" + n for n in names)
    # the (harpnum, time) encoded in the member name is exactly the requested one
    parsed = {}
    for n in mapping:
        m = re.match(r"hmi\.sharp_cea_720s\.(\d+)\.(\d{8}_\d{6})_TAI\.Br\.fits$", n)
        parsed[n] = (int(m[1]), pd.to_datetime(m[2], format="%Y%m%d_%H%M%S"))
    assert sorted(parsed.values()) == [(7, t) for t in ts]


def test_map_tar_members_unexpected_or_duplicate_fits_stop_with_names():
    ts = list(pd.date_range("2025-06-01", periods=2, freq="24min"))
    names = {_fname(7, t) for t in ts}
    stray = "./" + _fname(8, ts[0])
    with pytest.raises(TarMappingError) as e:
        map_tar_members(META + ["./" + n for n in names] + [stray], names)
    assert stray in str(e.value) and "all members" in str(e.value)
    n0 = sorted(names)[0]
    with pytest.raises(TarMappingError, match="Duplicate"):
        map_tar_members(["./" + n0, "sub/" + n0], names)


# ---------------- extraction ----------------
def test_extract_tar_writes_only_requested_flat_files_no_traversal(tmp_path):
    ts = list(pd.date_range("2025-06-01", periods=2, freq="24min"))
    names = {_fname(7, t) for t in ts}
    tar = tmp_path / "c.tar"
    _make_tar(tar, 7, ts)
    dest = tmp_path / "out"
    dest.mkdir()
    assert set(extract_tar(tar, dest, names)) == names
    assert {p.name for p in dest.iterdir()} == names  # metadata members not extracted
    evil = tmp_path / "evil.tar"
    _make_tar(evil, 7, ts, extra_members=["../../escape.fits"])
    with pytest.raises(TarMappingError):
        extract_tar(evil, dest, names)
    assert not (tmp_path / "escape.fits").exists() and not (tmp_path.parent / "escape.fits").exists()


# ---------------- pipeline ----------------
def _tar_mocks(tmp_path, calls=None, skip=(), extra=(), fail_export=False):
    rx = re.compile(r"^[\w.]+\[(?P<h>\d+)\]\[(?P<t>[^\]]+)\]\{\w+\}$")

    def export_fn(rs):
        if fail_export:
            raise ConnectionError("jsoc down")
        m = rx.match(rs)
        times = [pd.to_datetime(x, format="%Y.%m.%d_%H:%M:%S_TAI") for x in m["t"].split(",")]
        src = tmp_path / "_served" / f"{m['h']}_{times[0]:%Y%m%d%H%M}_{len(times)}.tar"
        src.parent.mkdir(exist_ok=True)
        _make_tar(src, int(m["h"]), times, extra_members=extra, skip=[t for t in times if t in skip])
        if calls is not None:
            calls.append((int(m["h"]), len(times)))
        return str(src)

    def download_fn(url, dest):
        dest.write_bytes(Path(url).read_bytes())
        return {"status": "ok", "error": None}
    return export_fn, download_fn


def _run_tar(frames, tmp_path, mocks, chunk=300, workers=2):
    st = PreprocessSettings(SERIES, SEG, workers=2, chunk_records=chunk)
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(tmp_path / "manifest.csv")
    try:
        return process_frames_tar(frames, frames_dir=tmp_path / "frames", raw_dir=tmp_path / "raw",
                                  manifest=manifest, tar_export_fn=mocks[0], tar_download_fn=mocks[1], st=st,
                                  progress=Progress(len(frames), every=2, printer=lambda s: None),
                                  chunk_workers=workers)
    finally:
        manifest.close()


def test_tar_pipeline_same_outputs_manifest_and_cleanup(tmp_path):
    frames = pd.concat([_frames(5, harp=11), _frames(3, harp=12)], ignore_index=True)
    calls = []
    counts, stats = _run_tar(frames, tmp_path, _tar_mocks(tmp_path, calls), chunk=3)
    assert counts == {"ok": 8, "failed": 0, "missing": 0}
    assert sorted(calls) == [(11, 2), (11, 3), (12, 3)]  # <= chunk records per tar, one HARP per tar
    assert stats["n_chunks"] == 3 and stats["n_files"] == 8 and stats["tar_bytes"] > 0
    for h, t in zip(frames["harpnum"], frames["time"]):
        arr = np.load(frame_path(tmp_path / "frames", h, t))  # same path layout as url-quick
        assert arr.shape == (128, 128) and arr.dtype == np.float16 and np.abs(arr).max() <= 1
    assert list((tmp_path / "raw").iterdir()) == []  # tars, raw FITS and chunk dirs all removed
    m = pd.read_csv(tmp_path / "manifest.csv")
    assert list(m.columns) == ["harpnum", "time", "status", "raw_h", "raw_w", "nan_fraction", "error"]
    assert (m["status"] == "ok").all() and len(m) == 8 and (m["raw_h"] == 60).all()


def test_tar_pipeline_is_resume_safe(tmp_path):
    frames = _frames(5)
    _run_tar(frames.head(2), tmp_path, _tar_mocks(tmp_path))
    pending, skipped = pending_frames(frames, tmp_path / "frames")
    assert skipped == 2 and len(pending) == 3
    calls = []
    counts, _ = _run_tar(pending, tmp_path, _tar_mocks(tmp_path, calls))
    assert counts["ok"] == 3 and calls == [(11117, 3)]
    assert pending_frames(frames, tmp_path / "frames")[0].empty


def test_tar_pipeline_missing_member_and_stale_chunk_dir(tmp_path):
    frames = _frames(3)
    skip = [frames["time"].iloc[1]]
    stale = tmp_path / "raw" / f"11117_{frames['time'].iloc[0]:%Y%m%d_%H%M}_3"
    stale.mkdir(parents=True)
    (stale / "junk.fits").write_bytes(b"x")
    counts, _ = _run_tar(frames, tmp_path, _tar_mocks(tmp_path, skip=skip))
    assert counts == {"ok": 2, "failed": 0, "missing": 1}
    assert list((tmp_path / "raw").iterdir()) == []
    assert not frame_path(tmp_path / "frames", 11117, skip[0]).exists()  # retried on the next run


def test_tar_pipeline_unmappable_member_stops_run_and_cleans_up(tmp_path):
    frames = _frames(3)
    stray = "./" + _fname(99999, frames["time"].iloc[0])
    with pytest.raises(TarMappingError, match="all members"):
        _run_tar(frames, tmp_path, _tar_mocks(tmp_path, extra=[stray]))
    assert list((tmp_path / "raw").iterdir()) == []


def test_tar_pipeline_corrupt_tar_and_export_failures_are_recorded(tmp_path):
    def bad_download(url, dest):
        dest.write_bytes(b"this is not a tar")
        return {"status": "ok", "error": None}
    exp, _ = _tar_mocks(tmp_path)
    counts, _ = _run_tar(_frames(2), tmp_path, (exp, bad_download))
    assert counts["failed"] == 2 and "extract" in pd.read_csv(tmp_path / "manifest.csv")["error"].iloc[0]
    assert list((tmp_path / "raw").iterdir()) == []

    frames = pd.concat([_frames(2, harp=h) for h in (1, 2, 3, 4)], ignore_index=True)
    with pytest.raises(RuntimeError, match="consecutive JSOC export failures"):
        _run_tar(frames, tmp_path / "b", _tar_mocks(tmp_path, fail_export=True), workers=1)
    m = pd.read_csv(tmp_path / "b" / "manifest.csv")
    assert (m["status"] == "failed").all() and "jsoc down" in m["error"].iloc[0]
