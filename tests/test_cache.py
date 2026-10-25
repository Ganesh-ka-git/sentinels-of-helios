"""Tests for the packed float16 frame cache (Change Log 14): identical inputs with and without the cache."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from flarecast.dataset import FlareDataset, make_loaders  # noqa: E402
from flarecast.dataset.cache import build_cache, unique_frames, verify_cache  # noqa: E402
from flarecast.dataset.flare_dataset import frame_filename  # noqa: E402
from test_module4 import T0, make_row, write_frames  # noqa: E402


@pytest.fixture
def world(tmp_path):
    rows = [make_row(1, T0, 0, "train", 0), make_row(1, T0 + pd.Timedelta(hours=6), 1, "train", 1),
            make_row(2, T0, 2, "train", 1), make_row(3, T0, 0, "val"), make_row(4, T0, 1, "test")]
    samples = pd.DataFrame(rows)
    fdir = tmp_path / "frames"
    for r in rows:
        d = fdir / str(r["harpnum"])
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(r["harpnum"])
        for fid in r["frame_ids"]:   # distinct random content per frame, so a wrong row lookup is detected
            np.save(d / frame_filename(fid), rng.standard_normal((128, 128)).astype(np.float16))
    cdir = tmp_path / "cache"
    build_cache(samples, fdir, cdir, workers=2)
    return samples, fdir, cdir


def _same(a, b):
    if isinstance(a, (tuple, list)):
        return all(torch.equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


@pytest.mark.parametrize("mode", ["static", "single", "dual"])
def test_cached_and_uncached_datasets_identical(world, mode):
    samples, fdir, cdir = world
    plain = FlareDataset(samples, fdir, mode, use_cache=False)
    cached = FlareDataset(samples, fdir, mode, use_cache=True, cache_dir=cdir)
    assert len(plain) == len(cached) == len(samples)
    for i in range(len(plain)):
        (xa, ya, ma), (xb, yb, mb) = plain[i], cached[i]
        assert _same(xa, xb) and ya == yb and ma == mb


def test_cache_default_dir_is_sibling_of_frames(world):
    samples, fdir, cdir = world   # cdir == fdir.parent / "cache"
    ds = FlareDataset(samples, fdir, "static")   # use_cache=True by default, default cache dir
    assert _same(ds[0][0], FlareDataset(samples, fdir, "static", use_cache=False)[0][0])


def test_unique_frames_dedup_and_index(world):
    samples, fdir, cdir = world
    idx = unique_frames(samples)
    assert not idx.duplicated(["harpnum", "time"]).any()
    assert len(idx) < sum(len(i) for i in samples["frame_ids"])   # HARP 1 samples 6 h apart share frames
    assert pd.read_csv(cdir / "frames_index.csv").shape[0] == len(idx)
    assert verify_cache(fdir, cdir, n=1000) == len(idx)


def test_verify_detects_corruption(world):
    samples, fdir, cdir = world
    arr = np.load(cdir / "frames_f16.npy", mmap_mode="r+")
    arr[0] = arr[0] + np.float16(1)
    arr.flush()
    del arr
    with pytest.raises(AssertionError):
        verify_cache(fdir, cdir, n=1000)


def test_missing_cache_and_missing_frame_fail_loudly(world, tmp_path):
    samples, fdir, cdir = world
    with pytest.raises(FileNotFoundError):
        FlareDataset(samples, fdir, "static", use_cache=True, cache_dir=tmp_path / "nope")
    extra = samples.iloc[[0]].copy()
    extra["harpnum"] = 999
    with pytest.raises(KeyError):
        FlareDataset(extra, fdir, "static", use_cache=True, cache_dir=cdir)
    (fdir / "1" / frame_filename(samples["frame_ids"][0][0])).unlink()
    with pytest.raises(FileNotFoundError):
        build_cache(samples, fdir, tmp_path / "cache2")
    assert not (tmp_path / "cache2" / "frames_f16.npy").exists()


def test_pickle_does_not_carry_the_memmap(world):
    samples, fdir, cdir = world
    ds = FlareDataset(samples, fdir, "dual", use_cache=True, cache_dir=cdir)
    ds[0]                                        # opens the memmap in this process
    assert ds._cache_arr is not None
    clone = pickle.loads(pickle.dumps(ds))       # what spawn does for each worker
    assert clone._cache_arr is None and _same(clone[1][0], ds[1][0])


def test_loader_with_cache_and_workers(world):
    samples, fdir, cdir = world
    a = next(iter(make_loaders("dual", 3, 0, samples=samples, frames_dir=fdir, use_cache=False)["val"]))
    b = next(iter(make_loaders("dual", 3, 2, samples=samples, frames_dir=fdir, cache_dir=cdir)["val"]))
    assert _same(a[0], b[0]) and torch.equal(a[1], b[1])
