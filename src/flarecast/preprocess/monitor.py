"""Read-only helpers for watching the Module 3 download: incremental manifest reader, per-month tracker, rate/ETA."""
from __future__ import annotations

import csv
import io
from collections import Counter, deque
from pathlib import Path


class ManifestTail:
    """Reads new rows appended to manifest.csv since the last poll. Never writes; tolerates a half-written last line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._offset = 0
        self._header: list[str] | None = None

    def poll(self) -> list[dict]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self._offset:  # file was replaced/truncated: start again
            self._offset, self._header = 0, None
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        end = data.rfind(b"\n") + 1  # only complete lines
        if end == 0:
            return []
        self._offset += end
        rows = list(csv.reader(io.StringIO(data[:end].decode("utf-8", errors="replace"))))
        out = []
        for r in rows:
            if not r:
                continue
            if self._header is None:
                self._header = r
                continue
            out.append(dict(zip(self._header, r)))
        return out


def frame_key(harpnum, time_str: str) -> str:
    return f"{int(harpnum)}|{time_str}"


class Tracker:
    """Per-frame state from manifest rows. ``needed`` maps frame_key -> month for every frame to be downloaded."""

    def __init__(self, needed: dict[str, str]):
        self.needed = needed
        self.total_by_month = Counter(needed.values())
        self.ok: set[str] = set()
        self.failed: dict[str, str] = {}    # latest error of frames that are not (yet) ok
        self.missing: set[str] = set()
        self.rows_seen = 0
        self.new_rows = Counter()           # status counts of rows added after ``mark_baseline``
        self.new_errors = Counter()         # short error kinds of failed rows added after baseline
        self.last_key: str | None = None
        self._baseline = False

    def mark_baseline(self) -> None:
        self._baseline = True

    def update(self, rows: list[dict]) -> None:
        for r in rows:
            k = frame_key(r["harpnum"], r["time"])
            st = r["status"]
            self.rows_seen += 1
            self.last_key = k
            if st == "ok":
                self.ok.add(k)
                self.failed.pop(k, None)
                self.missing.discard(k)
            elif st == "failed":
                if k not in self.ok:
                    self.failed[k] = r.get("error", "")
            elif st == "missing":
                if k not in self.ok:
                    self.missing.add(k)
            if self._baseline:
                self.new_rows[st] += 1
                if st == "failed":
                    self.new_errors[error_kind(r.get("error", ""))] += 1

    @property
    def done(self) -> int:
        return sum(1 for k in self.ok if k in self.needed)

    def done_by_month(self) -> Counter:
        c: Counter = Counter()
        for k in self.ok:
            m = self.needed.get(k)
            if m:
                c[m] += 1
        return c

    def outstanding_failed(self) -> int:
        return sum(1 for k in self.failed if k in self.needed)

    @property
    def current_month(self) -> str | None:
        return self.needed.get(self.last_key) if self.last_key else None


def error_kind(err: str) -> str:
    """Short label for a manifest error string."""
    e = err or ""
    for needle, label in (("pending export", "export: pending export request"), ("414", "export: HTTP 414"),
                          ("10060", "export: connection timed out"), ("export:", "export: other"),
                          ("ConnectTimeout", "download: connect timeout"), ("ReadTimeout", "download: read timeout"),
                          ("ConnectionError", "download: connection reset"), ("download:", "download: other"),
                          ("extract:", "extract"), ("Error", "preprocess")):
        if needle in e:
            return label
    return "other"


class RateWindow:
    """Remembers (time, done) samples; rate over the last ``window_s`` seconds (or all history if shorter)."""

    def __init__(self, keep_s: float = 3 * 3600):
        self.keep_s = keep_s
        self.samples: deque[tuple[float, int]] = deque()

    def add(self, t: float, done: int) -> None:
        self.samples.append((t, done))
        while self.samples and t - self.samples[0][0] > self.keep_s:
            self.samples.popleft()

    def rate(self, window_s: float) -> tuple[float | None, float]:
        """(frames/s or None, seconds of history actually used)."""
        if len(self.samples) < 2:
            return None, 0.0
        t1, d1 = self.samples[-1]
        t0, d0 = next(((t, d) for t, d in self.samples if t1 - t <= window_s), self.samples[0])
        span = t1 - t0
        return ((d1 - d0) / span if span > 0 else None), span


def fmt_duration(sec: float | None) -> str:
    if sec is None or sec != sec or sec == float("inf"):
        return "n/a"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def eta_seconds(remaining: int, rate: float | None) -> float | None:
    return remaining / rate if rate and rate > 0 else None
