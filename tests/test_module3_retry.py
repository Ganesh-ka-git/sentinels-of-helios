"""Tests for the url_quick export retry (mocked drms client, fake sleep, no network)."""
import logging
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_module3 import SEG, SERIES, _frames, _mock_download  # noqa: E402
from flarecast.data.jsoc import export_urls, short_record_set  # noqa: E402
from flarecast.preprocess.frames import Manifest  # noqa: E402
from flarecast.preprocess.pipeline import PreprocessSettings, Progress, process_frames, retry_export  # noqa: E402

PENDING = ("User x@y.z has 1 pending export requests (JSOC_20261024_001150); please wait until at least one "
           "request has completed before submitting a new one. [status=7]")


class FakeRequest:
    def __init__(self, record_set):
        import re
        m = re.search(r"\[(\d+)\]\[([^\]]+)\]", record_set)
        recs = [f"{record_set.split('[')[0]}[{m[1]}][{t}]" for t in m[2].split(",")]
        self.urls = pd.DataFrame({"record": recs, "url": [f"http://x/{i}" for i in range(len(recs))]})

    def wait(self):
        return True


class FakeClient:
    """drms.Client stand-in whose first ``errors`` export() calls raise, then succeed."""

    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def export(self, record_set, method, protocol, email):
        self.calls += 1
        assert (method, protocol) == ("url_quick", "as-is")
        if self.errors:
            raise self.errors.pop(0)
        return FakeRequest(record_set)


RS = [f"{SERIES}[11117][2024.05.01_00:00:00_TAI,2024.05.01_00:24:00_TAI,2024.05.01_00:48:00_TAI]{{{SEG}}}"]


def _export(client):
    return lambda rs: export_urls(client, rs, "x@y.z", SEG)


def test_success_first_try_never_sleeps():
    sleeps = []
    out = retry_export(_export(FakeClient([])), sleep=sleeps.append)(RS)
    assert len(out) == 3 and sleeps == []


def test_retries_timeout_urlerror_and_pending_export_with_backoff():
    import urllib.error
    client = FakeClient([TimeoutError("timed out"), urllib.error.URLError("WinError 10060"),
                         RuntimeError(PENDING)])
    sleeps = []
    out = retry_export(_export(client), sleep=sleeps.append)(RS)
    assert len(out) == 3 and client.calls == 4
    assert sleeps == [30, 60, 120]


def test_gives_up_after_ten_minutes_total_and_raises_last_error():
    client = FakeClient([RuntimeError(f"fail {i}") for i in range(50)])
    sleeps = []
    with pytest.raises(RuntimeError, match="fail 5"):
        retry_export(_export(client), sleep=sleeps.append)(RS)
    assert sleeps == [30, 60, 120, 240, 150] and sum(sleeps) == 600
    assert client.calls == 6


def test_retry_logs_every_failure(caplog):
    client = FakeClient([RuntimeError(PENDING)])
    with caplog.at_level(logging.WARNING):
        retry_export(_export(client), sleep=lambda s: None)(RS)
    assert any("pending export requests" in r.message and "30 s" in r.message for r in caplog.records)


def test_stop_event_aborts_wait_and_raises():
    stop = threading.Event()
    client = FakeClient([RuntimeError("down") for _ in range(10)])

    def sleep_and_interrupt(s):
        stop.set()  # Ctrl+C arrives during the wait
    with pytest.raises(RuntimeError, match="down"):
        retry_export(_export(client), sleep=sleep_and_interrupt, stop=stop)(RS)
    assert client.calls == 1


def test_short_log_line_format(caplog):
    assert short_record_set(RS[0]) == "HARP 11117: 3 records, 2024.05.01_00:00:00_TAI -> 2024.05.01_00:48:00_TAI"
    with caplog.at_level(logging.INFO):
        export_urls(FakeClient([]), RS, "x@y.z", SEG)
    msgs = [r.getMessage() for r in caplog.records if "Export" in r.getMessage()]
    assert any("HARP 11117: 3 records, 2024.05.01_00:00:00_TAI -> 2024.05.01_00:48:00_TAI" in m for m in msgs)
    assert all(len(m) < 200 for m in msgs)


def test_pipeline_recovers_from_transient_export_errors_instead_of_failing_chunk(tmp_path):
    client = FakeClient([TimeoutError("timed out"), RuntimeError(PENDING)])
    sleeps = []
    export_fn = retry_export(_export(client), sleep=sleeps.append)
    manifest = Manifest(tmp_path / "manifest.csv")
    frames = _frames(3)
    try:
        counts = process_frames(frames, frames_dir=tmp_path / "frames", raw_dir=tmp_path / "raw", manifest=manifest,
                                export_fn=export_fn, download_fn=_mock_download(),
                                st=PreprocessSettings(SERIES, SEG, workers=2),
                                progress=Progress(3, every=2, printer=lambda s: None))
    finally:
        manifest.close()
    assert counts == {"ok": 3, "failed": 0, "missing": 0} and sleeps == [30, 60]


def test_pipeline_records_failed_chunk_only_after_retries_exhausted(tmp_path):
    client = FakeClient([RuntimeError(PENDING) for _ in range(50)])
    export_fn = retry_export(_export(client), sleep=lambda s: None)
    manifest = Manifest(tmp_path / "manifest.csv")
    try:
        counts = process_frames(_frames(3), frames_dir=tmp_path / "frames", raw_dir=tmp_path / "raw",
                                manifest=manifest, export_fn=export_fn, download_fn=_mock_download(),
                                st=PreprocessSettings(SERIES, SEG, workers=2),
                                progress=Progress(3, every=2, printer=lambda s: None))
    finally:
        manifest.close()
    assert counts == {"ok": 0, "failed": 3, "missing": 0}
    assert "pending export requests" in pd.read_csv(tmp_path / "manifest.csv")["error"].iloc[0]
