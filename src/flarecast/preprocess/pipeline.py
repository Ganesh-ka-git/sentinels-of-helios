"""Download -> preprocess -> delete-raw pipeline for the frames listed in a frames CSV (decisions.md §6).

Network access is injected (``export_fn``, ``download_fn``) so the pipeline can be tested with mocks.
"""
from __future__ import annotations

import logging
import shutil
import tarfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..data.jsoc import _tai_str, local_filename
from .frames import Manifest, chunk_frames, frame_path, preprocess_br, read_br, save_frame

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_EXPORT_FAILURES = 3

ExportFn = Callable[[list[str]], pd.DataFrame]      # record sets -> DataFrame[record, filename, url]
DownloadFn = Callable[[str, Path], dict]            # url, dest -> {"status": "ok"|"skipped"|"failed", "error": ...}


@dataclass(frozen=True)
class PreprocessSettings:
    series: str
    segment: str
    size: int = 128
    clip_gauss: float = 1500.0
    dtype: str = "float16"
    workers: int = 8
    chunk_records: int = 300


def retry_export(export_fn: ExportFn, *, first_delay_s: float = 30.0, max_total_wait_s: float = 600.0,
                 sleep: Callable[[float], None] | None = None, stop: threading.Event | None = None) -> ExportFn:
    """Wrap ``export_fn`` so that any export error is retried with doubling waits (30, 60, 120, 240, ...).

    The waits are capped so their sum never exceeds ``max_total_wait_s`` (default 30+60+120+240+150 = 600 s); after
    that the last exception is raised unchanged and the caller records the chunk as failed. Every retry is logged as
    a warning. Setting ``stop`` aborts a wait at once and raises the last error (used for Ctrl+C). The wrapped
    function is only ever called from the single exporter thread, so exports are submitted one at a time.
    """
    wait = sleep or (stop.wait if stop is not None else time.sleep)

    def wrapped(record_sets: list[str]) -> pd.DataFrame:
        waited, delay, attempt = 0.0, first_delay_s, 1
        while True:
            try:
                return export_fn(record_sets)
            except Exception as e:  # noqa: BLE001 - any JSOC/network error is retried, then re-raised
                remaining = max_total_wait_s - waited
                if remaining <= 0 or (stop is not None and stop.is_set()):
                    log.error("Export failed after %d attempts (%.0f s of waiting): %r", attempt, waited, e)
                    raise
                pause = min(delay, remaining)
                log.warning("Export attempt %d failed (%r); waiting %.0f s before retrying", attempt, e, pause)
                wait(pause)
                if stop is not None and stop.is_set():
                    raise
                waited, delay, attempt = waited + pause, delay * 2, attempt + 1
    return wrapped


class Progress:
    """Counts finished frames; every ``every`` files prints done/total, files/s over the last ``every`` and the ETA."""

    def __init__(self, total: int, every: int = 500, clock: Callable[[], float] = time.monotonic,
                 printer: Callable[[str], None] = print):
        self.total, self.every, self._clock, self._print = total, every, clock, printer
        self.done = 0
        self.t_start = clock()
        self._stamps: deque[float] = deque([self.t_start])

    def update(self, n: int = 1) -> None:
        for _ in range(n):
            self.done += 1
            if self.done % self.every == 0:
                now = self._clock()
                self._stamps.append(now)
                if len(self._stamps) > 2:
                    self._stamps.popleft()
                rate = self.every / max(self._stamps[-1] - self._stamps[0], 1e-9)
                eta_min = (self.total - self.done) / rate / 60 if rate > 0 else float("inf")
                self._print(f"[{self.done}/{self.total}] {rate:.2f} files/s (last {self.every}), ETA {eta_min:.0f} min")

    def overall_rate(self) -> float:
        return self.done / max(self._clock() - self.t_start, 1e-9)


def pending_frames(frames: pd.DataFrame, frames_dir: Path) -> tuple[pd.DataFrame, int]:
    """Drop frames whose .npy already exists. Returns (pending frames, number skipped)."""
    exists = [frame_path(frames_dir, h, t).exists() for h, t in zip(frames["harpnum"], frames["time"])]
    mask = pd.Series(exists, index=frames.index)
    return frames[~mask], int(mask.sum())


def _process_file(harp: int, t: pd.Timestamp, url: str, raw_path: Path, out_path: Path,
                  download_fn: DownloadFn, st: PreprocessSettings) -> dict:
    """Worker: download one raw FITS, preprocess, save .npy, delete the raw file. Never raises."""
    row = {"harpnum": harp, "time": t, "status": "failed", "raw_h": None, "raw_w": None,
           "nan_fraction": None, "error": None}
    try:
        res = download_fn(url, raw_path)
        if res["status"] == "failed":
            row["error"] = f"download: {res['error']}"
            return row
        raw = read_br(raw_path)
        row["raw_h"], row["raw_w"] = int(raw.shape[0]), int(raw.shape[1])
        arr, nan_frac = preprocess_br(raw, st.size, st.clip_gauss, st.dtype)
        save_frame(out_path, arr)
        row.update(status="ok", nan_fraction=nan_frac)
    except Exception as e:  # recorded in the manifest and logged, not swallowed
        row["error"] = f"{type(e).__name__}: {e}"
        log.error("Frame HARP %d %s failed: %s", harp, t, row["error"])
    finally:
        try:
            raw_path.unlink(missing_ok=True)
            raw_path.with_name(raw_path.name + ".part").unlink(missing_ok=True)
        except OSError as e:
            log.warning("Could not delete raw file %s: %s", raw_path, e)
    return row


def process_frames(frames: pd.DataFrame, *, frames_dir: Path, raw_dir: Path, manifest: Manifest,
                   export_fn: ExportFn, download_fn: DownloadFn, st: PreprocessSettings,
                   progress: Progress) -> dict:
    """Process pending ``frames`` (columns harpnum, time) in chunks of <= st.chunk_records records per HARP.

    Export of chunk i+1 overlaps with download/preprocessing of chunk i. Every frame ends up in the manifest as
    ok / failed / missing. On KeyboardInterrupt, running frames are finished and recorded, queued ones are dropped
    (they are redone on the next run), and the interrupt is re-raised.
    """
    counts = {"ok": 0, "failed": 0, "missing": 0}
    chunks = chunk_frames(frames, st.chunk_records)
    if not chunks:
        return counts
    raw_dir.mkdir(parents=True, exist_ok=True)

    def record_set(chunk: pd.DataFrame) -> str:
        harp = int(chunk["harpnum"].iloc[0])
        times = ",".join(_tai_str(pd.Timestamp(t)) for t in chunk["time"])
        return f"{st.series}[{harp}][{times}]{{{st.segment}}}"

    def note(row: dict) -> None:
        manifest.add(**row)
        counts[row["status"]] += 1
        progress.update()

    exporter = ThreadPoolExecutor(max_workers=1)  # drms client is only ever touched from this one thread
    pool = ThreadPoolExecutor(max_workers=st.workers)
    export_futs = [exporter.submit(export_fn, [record_set(chunks[0])])]
    consec_fail = 0
    futs: list[Future] = []
    handled: set[Future] = set()
    try:
        for i, chunk in enumerate(chunks):
            if i + 1 < len(chunks):
                export_futs.append(exporter.submit(export_fn, [record_set(chunks[i + 1])]))
            harp = int(chunk["harpnum"].iloc[0])
            try:
                urls = export_futs[i].result()
                consec_fail = 0
            except Exception as e:
                consec_fail += 1
                err = f"export: {type(e).__name__}: {e}"
                log.error("Export failed for HARP %d (%d frames): %s", harp, len(chunk), err)
                for t in chunk["time"]:
                    note({"harpnum": harp, "time": t, "status": "failed", "error": err})
                if consec_fail >= MAX_CONSECUTIVE_EXPORT_FAILURES:
                    raise RuntimeError(f"{consec_fail} consecutive JSOC export failures; last: {err}") from e
                continue

            url_by_name = dict(zip(urls["filename"], urls["url"]))
            futs = []
            for t in chunk["time"]:
                t = pd.Timestamp(t)
                name = local_filename(f"{st.series}[{harp}][{_tai_str(t)}]", st.segment)
                if name not in url_by_name:
                    note({"harpnum": harp, "time": t, "status": "missing",
                          "error": "record not returned by JSOC export"})
                    continue
                futs.append(pool.submit(_process_file, harp, t, url_by_name[name], raw_dir / name,
                                        frame_path(frames_dir, harp, t), download_fn, st))
            for fut in as_completed(futs):
                note(fut.result())
                handled.add(fut)
    except KeyboardInterrupt:
        for f in futs:
            f.cancel()
        pool.shutdown(wait=True)  # lets in-flight frames finish so their raw files are cleaned up
        for f in futs:
            if f not in handled and f.done() and not f.cancelled():
                note(f.result())
        raise
    finally:
        exporter.shutdown(wait=False, cancel_futures=True)
        pool.shutdown(wait=True)
    return counts


# ---------------------------------------------------------------------------------------------------------------
# url-tar method: one staged .tar per (HARP, chunk) instead of one URL per file
# ---------------------------------------------------------------------------------------------------------------
TarExportFn = Callable[[str], str]   # record set -> absolute URL of the staged .tar (raises on failure)


class TarMappingError(ValueError):
    """A tar member cannot be mapped unambiguously to a requested (harpnum, time). Never swallowed."""


def map_tar_members(member_names: list[str], expected_names: set[str]) -> dict[str, str]:
    """Map tar member names to the expected local file names (HARP + TAI time are embedded in the member name).

    Only ``*.fits`` members are considered (the tar also holds index.json/index.txt/.drmsrun/.qsub/keywords.txt).
    Returns {local file name: member name}. Raises TarMappingError for duplicates or FITS members that were not
    requested; the message lists every member name so the mapping can be inspected.
    """
    out: dict[str, str] = {}
    for n in member_names:
        base = n.replace("\\", "/").rsplit("/", 1)[-1]
        if not base.lower().endswith(".fits"):
            continue
        if base not in expected_names:
            raise TarMappingError(f"Unexpected FITS member {n!r}; expected e.g. {sorted(expected_names)[:3]}; "
                                  f"all members: {member_names}")
        if base in out:
            raise TarMappingError(f"Duplicate FITS member {base!r}; all members: {member_names}")
        out[base] = n
    return out


def extract_tar(tar_path: Path, dest_dir: Path, expected_names: set[str]) -> list[str]:
    """Extract the requested FITS members of ``tar_path`` into ``dest_dir`` (flat, by base name; no path traversal).

    Returns the local file names extracted.
    """
    with tarfile.open(tar_path) as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        mapping = map_tar_members(list(members), expected_names)
        for fname, member_name in mapping.items():
            src = tf.extractfile(members[member_name])
            with open(dest_dir / fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return sorted(mapping)


def _preprocess_local(harp: int, t: pd.Timestamp, raw_path: Path, out_path: Path, st: PreprocessSettings) -> dict:
    """Preprocess an already-downloaded raw FITS (same function and settings as _process_file). Never raises."""
    row = {"harpnum": harp, "time": t, "status": "failed", "raw_h": None, "raw_w": None,
           "nan_fraction": None, "error": None}
    try:
        raw = read_br(raw_path)
        row["raw_h"], row["raw_w"] = int(raw.shape[0]), int(raw.shape[1])
        arr, nan_frac = preprocess_br(raw, st.size, st.clip_gauss, st.dtype)
        save_frame(out_path, arr)
        row.update(status="ok", nan_fraction=nan_frac)
    except Exception as e:  # recorded in the manifest and logged, not swallowed
        row["error"] = f"{type(e).__name__}: {e}"
        log.error("Frame HARP %d %s failed: %s", harp, t, row["error"])
    return row


def _tar_chunk_job(chunk: pd.DataFrame, chunk_dir: Path, frames_dir: Path, st: PreprocessSettings,
                   tar_export_fn: TarExportFn, tar_download_fn: DownloadFn) -> dict:
    """Stage, download, extract and preprocess one chunk. Returns {"rows", "export_failed", "stats"}.

    The chunk directory (tar + raw FITS) is always removed afterwards. TarMappingError propagates (stop the run).
    """
    harp = int(chunk["harpnum"].iloc[0])
    times = [pd.Timestamp(t) for t in chunk["time"]]
    names = {local_filename(f"{st.series}[{harp}][{_tai_str(t)}]", st.segment): t for t in times}
    record_set = f"{st.series}[{harp}][{','.join(_tai_str(t) for t in times)}]{{{st.segment}}}"
    stats = {"staging_s": 0.0, "download_s": 0.0, "tar_bytes": 0, "n_files": 0}

    def failed(err: str, export_failed: bool = False) -> dict:
        log.error("HARP %d chunk of %d frames failed: %s", harp, len(times), err)
        rows = [{"harpnum": harp, "time": t, "status": "failed", "error": err} for t in times]
        return {"rows": rows, "export_failed": export_failed, "stats": stats}

    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)  # stale leftovers from an interrupted run
    try:
        tic = time.monotonic()
        try:
            url = tar_export_fn(record_set)
        except Exception as e:
            return failed(f"export: {type(e).__name__}: {e}", export_failed=True)
        finally:
            stats["staging_s"] = time.monotonic() - tic

        chunk_dir.mkdir(parents=True)
        tar_path = chunk_dir / "chunk.tar"
        tic = time.monotonic()
        res = tar_download_fn(url, tar_path)
        stats["download_s"] = time.monotonic() - tic
        if res["status"] == "failed":
            return failed(f"download: {res['error']}")
        stats["tar_bytes"] = tar_path.stat().st_size

        try:
            extracted = set(extract_tar(tar_path, chunk_dir, set(names)))
        except TarMappingError:
            raise
        except Exception as e:
            return failed(f"extract: {type(e).__name__}: {e}")
        tar_path.unlink()

        rows = []
        for fname, t in names.items():
            if fname not in extracted:
                rows.append({"harpnum": harp, "time": t, "status": "missing",
                             "error": "record not in JSOC tar export"})
                continue
            rows.append(_preprocess_local(harp, t, chunk_dir / fname, frame_path(frames_dir, harp, t), st))
            (chunk_dir / fname).unlink(missing_ok=True)
        stats["n_files"] = len(extracted)
        return {"rows": rows, "export_failed": False, "stats": stats}
    finally:
        if chunk_dir.exists():
            try:
                shutil.rmtree(chunk_dir)
            except OSError as e:
                log.warning("Could not delete %s: %s", chunk_dir, e)


def process_frames_tar(frames: pd.DataFrame, *, frames_dir: Path, raw_dir: Path, manifest: Manifest,
                       tar_export_fn: TarExportFn, tar_download_fn: DownloadFn, st: PreprocessSettings,
                       progress: Progress, chunk_workers: int = 1) -> tuple[dict, dict]:
    """url-tar counterpart of :func:`process_frames`: one tar per (HARP, <= st.chunk_records frames).

    Same per-frame preprocessing, .npy paths and manifest rows. ``chunk_workers`` chunks run concurrently, each in
    ``raw_dir/<chunk id>/``. Returns (counts, stats) where stats sums staging/download seconds and tar bytes over
    all chunks. KeyboardInterrupt finishes running chunks, records them, and re-raises.
    """
    counts = {"ok": 0, "failed": 0, "missing": 0}
    stats = {"n_chunks": 0, "staging_s": 0.0, "download_s": 0.0, "tar_bytes": 0, "n_files": 0}
    chunks = chunk_frames(frames, st.chunk_records)
    if not chunks:
        return counts, stats
    if chunk_workers < 1:
        raise ValueError("chunk_workers must be >= 1")
    raw_dir.mkdir(parents=True, exist_ok=True)
    consec_fail = 0

    def note(res: dict) -> None:
        nonlocal consec_fail
        for row in res["rows"]:
            manifest.add(**row)
            counts[row["status"]] += 1
            progress.update()
        stats["n_chunks"] += 1
        for k in ("staging_s", "download_s", "tar_bytes", "n_files"):
            stats[k] += res["stats"][k]
        consec_fail = consec_fail + 1 if res["export_failed"] else 0
        if consec_fail >= MAX_CONSECUTIVE_EXPORT_FAILURES:
            raise RuntimeError(f"{consec_fail} consecutive JSOC export failures; last: {res['rows'][0]['error']}")

    def chunk_dir(c: pd.DataFrame) -> Path:
        return raw_dir / f"{int(c['harpnum'].iloc[0])}_{pd.Timestamp(c['time'].iloc[0]):%Y%m%d_%H%M}_{len(c)}"

    pool = ThreadPoolExecutor(max_workers=chunk_workers)
    futs = [pool.submit(_tar_chunk_job, c, chunk_dir(c), frames_dir, st, tar_export_fn, tar_download_fn)
            for c in chunks]
    handled: set[Future] = set()
    try:
        for fut in as_completed(futs):
            note(fut.result())
            handled.add(fut)
    except BaseException as e:
        for f in futs:
            f.cancel()
        pool.shutdown(wait=True)
        if isinstance(e, KeyboardInterrupt):
            for f in futs:
                if f not in handled and f.done() and not f.cancelled() and f.exception() is None:
                    note(f.result())
        raise
    finally:
        pool.shutdown(wait=True)
    return counts, stats
