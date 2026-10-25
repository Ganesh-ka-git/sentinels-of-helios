"""Unit tests for Module 3 (preprocessing + download pipeline), synthetic arrays, mocked network."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.data.jsoc import local_filename  # noqa: E402
from flarecast.preprocess.frames import (Manifest, area_resize, chunk_frames, frame_path, pad_to_square,  # noqa: E402
                                         preprocess_br, read_br)
from flarecast.preprocess.pipeline import PreprocessSettings, Progress, pending_frames, process_frames  # noqa: E402

SERIES, SEG = "hmi.sharp_cea_720s", "Br"


# ---------------- preprocessing ----------------
def test_output_shape_dtype_and_range():
    raw = np.zeros((377, 745), dtype=np.float32)
    raw[:, :300], raw[:, 300:] = 5000.0, -5000.0  # far beyond +-1500 G
    out, _ = preprocess_br(raw)
    assert out.shape == (128, 128) and out.dtype == np.float16
    assert out.min() >= -1.0 and out.max() <= 1.0
    assert out.max() == 1.0 and out.min() == -1.0  # clipping saturates at exactly +-1


def test_nan_becomes_zero_and_fraction_reported():
    raw = np.full((200, 400), 750.0, dtype=np.float32)
    raw[:100, :] = np.nan
    out, nan_frac = preprocess_br(raw)
    assert nan_frac == pytest.approx(0.5)
    assert np.isfinite(out.astype(np.float32)).all()
    o = out.astype(np.float32)  # 200x400 -> padded 400x400 -> 128x128: pad rows 0-32, NaN rows 32-64, valid 64-96
    assert o[:64].max() == 0.0 and o[96:].max() == 0.0
    assert o[66:94].min() == pytest.approx(0.5, abs=1e-3)  # 750/1500 on the valid half


def test_clip_and_scale():
    out, _ = preprocess_br(np.full((50, 50), 99999.0, dtype=np.float32), size=16)
    assert np.all(out == 1.0)
    out, _ = preprocess_br(np.full((50, 50), -99999.0, dtype=np.float32), size=16)
    assert np.all(out == -1.0)
    out, _ = preprocess_br(np.full((50, 50), 300.0, dtype=np.float32), size=16)
    assert np.allclose(out.astype(np.float32), 0.2, atol=1e-3)


def test_padding_preserves_aspect_and_centres():
    img = np.ones((20, 60), dtype=np.float32)
    sq = pad_to_square(img)
    assert sq.shape == (60, 60)
    assert sq[20:40].all() and not sq[:20].any() and not sq[40:].any()
    out, _ = preprocess_br(img * 1500.0, size=60)  # 1:1 resize -> padded layout survives exactly
    assert out[20:40].min() == 1.0 and out[:20].max() == 0.0 and out[40:].max() == 0.0
    # also for the final 128 output: a 1:3 image fills the middle third of the rows
    out, _ = preprocess_br(np.full((100, 300), 1500.0, dtype=np.float32))
    filled = np.where(out.astype(np.float32).max(axis=1) > 0.99)[0]
    assert 40 <= filled.min() <= 44 and 83 <= filled.max() <= 87


def test_area_resize_is_exact_mean_for_integer_factor_and_conserves_mean():
    rng = np.random.default_rng(1)
    img = rng.random((64, 64)).astype(np.float32)
    out = area_resize(img, 16)
    assert np.allclose(out, img.reshape(16, 4, 16, 4).mean(axis=(1, 3)), atol=1e-6)
    img = rng.random((100, 100)).astype(np.float32)  # fractional factor
    assert area_resize(img, 128 // 3).mean() == pytest.approx(img.mean(), rel=1e-5)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        preprocess_br(np.zeros((3, 10, 10), dtype=np.float32))


def test_frame_path_and_chunking():
    p = frame_path(Path("f"), 11117, pd.Timestamp("2024-05-01 06:24:00"))
    assert p == Path("f/11117/20240501_0624.npy")
    df = pd.DataFrame({"harpnum": [1] * 7 + [2] * 2, "time": pd.date_range("2024-05-01", periods=9, freq="12min")})
    chunks = chunk_frames(df, 3)
    assert [len(c) for c in chunks] == [3, 3, 1, 2]
    assert all(c["harpnum"].nunique() == 1 for c in chunks)


# ---------------- pipeline with mocked network ----------------
def _write_fits(path: Path, shape=(60, 120), value=300.0, nan=True):
    from astropy.io import fits
    data = np.full(shape, value, dtype=np.float32)
    if nan:
        data[:5, :5] = np.nan
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data)]).writeto(path, overwrite=True)


def _mock_export(calls):
    rx = re.compile(r"^(?P<s>[\w.]+)\[(?P<h>\d+)\]\[(?P<t>[^\]]+)\]\{(?P<g>\w+)\}$")

    def export_fn(record_sets):
        rows = []
        for rs in record_sets:
            m = rx.match(rs)
            for t in m["t"].split(","):
                rec = f"{m['s']}[{m['h']}][{t}]"
                rows.append({"record": rec, "filename": local_filename(rec, m["g"]), "url": "http://x/" + t})
        calls.append(record_sets)
        return pd.DataFrame(rows)
    return export_fn


def _mock_download(fail_names=()):
    def download_fn(url, dest):
        if any(n in dest.name for n in fail_names):
            return {"status": "failed", "error": "boom"}
        _write_fits(dest)
        return {"status": "ok", "error": None}
    return download_fn


def _run(frames, tmp_path, export_calls=None, download_fn=None, export_fn=None, chunk=300):
    st = PreprocessSettings(SERIES, SEG, workers=3, chunk_records=chunk)
    manifest = Manifest(tmp_path / "manifest.csv")
    try:
        counts = process_frames(frames, frames_dir=tmp_path / "frames", raw_dir=tmp_path / "raw", manifest=manifest,
                                export_fn=export_fn or _mock_export(export_calls if export_calls is not None else []),
                                download_fn=download_fn or _mock_download(), st=st, progress=Progress(len(frames),
                                every=2, printer=lambda s: None))
    finally:
        manifest.close()
    return counts


def _frames(n=5, harp=11117):
    return pd.DataFrame({"harpnum": harp, "time": pd.date_range("2024-05-01 00:00", periods=n, freq="24min")})


def test_end_to_end_writes_npy_manifest_and_deletes_raw(tmp_path):
    counts = _run(_frames(5), tmp_path)
    assert counts == {"ok": 5, "failed": 0, "missing": 0}
    for t in _frames(5)["time"]:
        arr = np.load(frame_path(tmp_path / "frames", 11117, t))
        assert arr.shape == (128, 128) and arr.dtype == np.float16 and np.abs(arr).max() <= 1
    assert list((tmp_path / "raw").glob("*")) == []  # raw FITS removed
    m = pd.read_csv(tmp_path / "manifest.csv")
    assert list(m.columns) == ["harpnum", "time", "status", "raw_h", "raw_w", "nan_fraction", "error"]
    assert (m["status"] == "ok").all() and (m["raw_h"] == 60).all() and (m["raw_w"] == 120).all()
    assert np.allclose(m["nan_fraction"], 25 / 7200, atol=1e-5)


def test_resume_skips_existing_files(tmp_path):
    frames = _frames(5)
    _run(frames.head(2), tmp_path)
    pending, skipped = pending_frames(frames, tmp_path / "frames")
    assert skipped == 2 and len(pending) == 3
    calls = []
    counts = _run(pending, tmp_path, export_calls=calls)
    assert counts["ok"] == 3
    requested = ",".join(rs for call in calls for rs in call)
    assert "00:00:00" not in requested and "00:24:00" not in requested  # existing frames never re-requested
    pending, skipped = pending_frames(frames, tmp_path / "frames")
    assert skipped == 5 and pending.empty


def test_failed_download_is_logged_in_manifest_and_leaves_no_npy(tmp_path):
    frames = _frames(3)
    counts = _run(frames, tmp_path, download_fn=_mock_download(fail_names=["000000_TAI"]))
    assert counts == {"ok": 2, "failed": 1, "missing": 0}
    m = pd.read_csv(tmp_path / "manifest.csv")
    bad = m[m["status"] == "failed"]
    assert len(bad) == 1 and "boom" in bad["error"].iloc[0]
    assert not frame_path(tmp_path / "frames", 11117, frames["time"].iloc[0]).exists()
    assert pending_frames(frames, tmp_path / "frames")[1] == 2  # the failed frame is retried next run


def test_corrupt_fits_is_failed_not_dropped(tmp_path):
    def bad_download(url, dest):
        dest.write_bytes(b"not a fits file")
        return {"status": "ok", "error": None}
    counts = _run(_frames(2), tmp_path, download_fn=bad_download)
    assert counts["failed"] == 2
    assert list((tmp_path / "raw").glob("*")) == []


def test_record_not_returned_by_jsoc_is_missing(tmp_path):
    inner = _mock_export([])

    def export_fn(rs):
        df = inner(rs)
        return df.iloc[1:]  # JSOC "forgets" the first record
    counts = _run(_frames(3), tmp_path, export_fn=export_fn)
    assert counts == {"ok": 2, "failed": 0, "missing": 1}
    assert (pd.read_csv(tmp_path / "manifest.csv")["status"] == "missing").sum() == 1


def test_export_chunking_limits_records_per_request(tmp_path):
    calls = []
    _run(_frames(7), tmp_path, export_calls=calls, chunk=3)
    sizes = [rs.count("_TAI") for call in calls for rs in call]
    assert sizes == [3, 3, 1]


def test_repeated_export_failures_abort_loudly(tmp_path):
    def export_fn(rs):
        raise ConnectionError("jsoc down")
    frames = pd.concat([_frames(2, harp=h) for h in (1, 2, 3, 4)], ignore_index=True)
    with pytest.raises(RuntimeError, match="consecutive JSOC export failures"):
        _run(frames, tmp_path, export_fn=export_fn)
    m = pd.read_csv(tmp_path / "manifest.csv")
    assert (m["status"] == "failed").all() and "jsoc down" in m["error"].iloc[0]


def test_read_br_returns_first_2d_hdu(tmp_path):
    p = tmp_path / "x.fits"
    _write_fits(p, shape=(10, 20), nan=False)
    arr = read_br(p)
    assert arr.shape == (10, 20) and arr.dtype == np.float32


def test_progress_reports_every_n(tmp_path):
    t = iter([0.0, 10.0, 30.0])
    msgs = []
    pr = Progress(total=1000, every=500, clock=lambda: next(t), printer=msgs.append)
    pr.update(500)
    pr.update(500)
    assert len(msgs) == 2
    assert "[500/1000] 50.00 files/s" in msgs[0] and "[1000/1000] 25.00 files/s" in msgs[1]
