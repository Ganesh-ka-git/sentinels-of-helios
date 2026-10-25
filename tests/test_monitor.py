"""Tests for the read-only download monitor helpers (synthetic manifest files)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.preprocess.monitor import (ManifestTail, RateWindow, Tracker, error_kind, eta_seconds,  # noqa: E402
                                          fmt_duration, frame_key)

HEADER = "harpnum,time,status,raw_h,raw_w,nan_fraction,error\n"


def row(h, t, status, err=""):
    return f'{h},{t},{status},10,20,0.0,"{err}"\n'


def test_tail_reads_incrementally_and_ignores_partial_line(tmp_path):
    p = tmp_path / "m.csv"
    tail = ManifestTail(p)
    assert tail.poll() == []                                   # file does not exist yet
    p.write_text(HEADER + row(1, "2024-01-01 00:00:00", "ok"))
    assert [r["harpnum"] for r in tail.poll()] == ["1"]
    with open(p, "a") as f:
        f.write(row(2, "2024-01-01 00:24:00", "failed", "export: boom, with comma"))
        f.write("3,2024-01-01 00:48:00,ok,10")                 # half-written line, no newline yet
    got = tail.poll()
    assert len(got) == 1 and got[0]["error"] == "export: boom, with comma"
    with open(p, "a") as f:
        f.write(",20,0.0,\n")                                  # line completed
    assert [r["harpnum"] for r in tail.poll()] == ["3"] and tail.poll() == []


def test_tracker_counts_done_failed_and_month_progress():
    needed = {frame_key(1, "2024-01-01 00:00:00"): "2024-01", frame_key(1, "2024-01-01 00:24:00"): "2024-01",
              frame_key(2, "2024-02-01 00:00:00"): "2024-02"}
    tr = Tracker(needed)
    tr.update([{"harpnum": "1", "time": "2024-01-01 00:00:00", "status": "failed", "error": "export: HTTP Error 414"}])
    tr.mark_baseline()
    assert tr.done == 0 and tr.outstanding_failed() == 1
    tr.update([{"harpnum": "1", "time": "2024-01-01 00:00:00", "status": "ok"},          # retry succeeded
               {"harpnum": "1", "time": "2024-01-01 00:24:00", "status": "failed", "error": "download: ReadTimeout"},
               {"harpnum": "2", "time": "2024-02-01 00:00:00", "status": "ok"},
               {"harpnum": "9", "time": "2030-01-01 00:00:00", "status": "ok"}])         # not in the needed list
    assert tr.done == 2 and tr.outstanding_failed() == 1
    assert dict(tr.done_by_month()) == {"2024-01": 1, "2024-02": 1}
    assert dict(tr.total_by_month) == {"2024-01": 2, "2024-02": 1}
    assert tr.new_rows["ok"] == 3 and tr.new_errors["download: read timeout"] == 1
    assert tr.current_month is None   # last row belongs to a frame outside the list


def test_rate_window_and_eta():
    rw = RateWindow()
    assert rw.rate(60) == (None, 0.0)
    for i in range(0, 700, 10):                      # 2 frames/s for 690 s
        rw.add(float(i), 2 * i)
    r, span = rw.rate(600)
    assert abs(r - 2.0) < 1e-9 and 590 <= span <= 600
    assert eta_seconds(7200, 2.0) == 3600 and eta_seconds(10, None) is None and eta_seconds(10, 0) is None
    assert fmt_duration(3600 * 5 + 120) == "5h02m" and fmt_duration(95) == "1m35s" and fmt_duration(None) == "n/a"


def test_error_kinds():
    assert error_kind("export: DrmsExportError: User x has 1 pending export requests") == "export: pending export request"
    assert error_kind("export: HTTPError: HTTP Error 414: Failed") == "export: HTTP 414"
    assert error_kind("export: URLError: <urlopen error [WinError 10060]") == "export: connection timed out"
    assert error_kind("download: ConnectTimeout(MaxRetryError(") == "download: connect timeout"
    assert error_kind("download: ConnectionError(ProtocolError(") == "download: connection reset"
