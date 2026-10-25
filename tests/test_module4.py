"""Unit tests for Module 4 (dataset, availability filter, loaders, class weights). Synthetic frames, no network."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.dataset import (COARSE, FINE, FlareDataset, availability_report, class_weights,  # noqa: E402
                               filter_available, load_samples, make_loaders)
from flarecast.dataset.flare_dataset import frame_filename  # noqa: E402
import functools  # noqa: E402

# these tests exercise the per-frame .npy path; the cache is tested in test_cache.py
FlareDataset = functools.partial(FlareDataset, use_cache=False)  # noqa: F811
make_loaders = functools.partial(make_loaders, use_cache=False)  # noqa: F811

T0 = pd.Timestamp("2024-03-01 12:00")


def ids_for(t, step_min=24):
    return [(t - pd.Timedelta(minutes=step_min * k)).strftime("%Y%m%dT%H%M") for k in range(30, -1, -1)]


def make_row(harp, t, label, split, persistence=0):
    return {"harpnum": harp, "t": t, "label": label, "persistence_label": persistence, "split": split,
            "included": True, "frame_ids": ids_for(t), "frames": ";".join(ids_for(t))}


def write_frames(frames_dir: Path, row, value_by_index=True):
    """Frame k of the sample is a constant image with value k/64 (exact in float16) so indices can be checked."""
    d = frames_dir / str(row["harpnum"])
    d.mkdir(parents=True, exist_ok=True)
    for k, fid in enumerate(row["frame_ids"]):
        np.save(d / frame_filename(fid), np.full((128, 128), k / 64 if value_by_index else 0.0, dtype=np.float16))


@pytest.fixture
def world(tmp_path):
    rows = [make_row(1, T0, 0, "train", 0), make_row(2, T0, 1, "train", 2), make_row(3, T0, 2, "train", 1),
            make_row(4, T0, 0, "val"), make_row(5, T0, 1, "test")]
    samples = pd.DataFrame(rows)
    for r in rows:
        write_frames(tmp_path / "frames", r)
    return samples, tmp_path / "frames"


# ---------------- index selection ----------------
def test_coarse_and_fine_indices_are_exact():
    idx = np.arange(31)
    assert idx[COARSE].tolist() == list(range(0, 31, 2)) and len(idx[COARSE]) == 16
    assert idx[FINE].tolist() == list(range(15, 31)) and len(idx[FINE]) == 16


def test_frame_filename_mapping():
    assert frame_filename("20231231T1200") == "20231231_1200.npy"
    with pytest.raises(ValueError):
        frame_filename("2023-12-31")


def test_dual_selects_exact_frames_and_t_is_last_in_every_mode(world):
    samples, fdir = world
    row = samples.iloc[[0]]
    k = lambda x: (x[:, 0, 0, 0] * 64).round().long()  # noqa: E731  -> frame index per image (constant images)
    (c, f), _, _ = FlareDataset(row, fdir, "dual")[0]
    assert k(c).tolist() == list(range(0, 31, 2))      # coarse = frames[0::2]
    assert k(f).tolist() == list(range(15, 31))        # fine = frames[15:]
    x, _, _ = FlareDataset(row, fdir, "static")[0]
    s, _, _ = FlareDataset(row, fdir, "single")[0]
    assert k(x[None]).item() == 30 and k(s).tolist() == list(range(31))  # t = index 30 (newest)
    assert k(c)[-1] == 30 and k(f)[-1] == 30 and k(s)[-1] == 30


def test_input_uses_only_files_up_to_t(world):
    samples, fdir = world
    t = pd.Timestamp(samples["t"].iloc[0])
    future = fdir / "1" / (t + pd.Timedelta(minutes=24)).strftime("%Y%m%d_%H%M.npy")
    np.save(future, np.full((128, 128), 0.99, dtype=np.float16))   # a frame after t must never be read
    for mode in ("static", "single", "dual"):
        x, _, _ = FlareDataset(samples.iloc[[0]], fdir, mode)[0]
        flat = torch.cat([v.flatten() for v in x]) if isinstance(x, tuple) else x.flatten()
        assert not torch.isclose(flat, torch.tensor(0.99, dtype=torch.float32), atol=1e-3).any()


def test_substituted_frames_keep_positional_selection(world):
    """With a substituted frame the 31 ids are not on the nominal grid; selection stays positional (spec)."""
    samples, fdir = world
    r = make_row(9, T0, 0, "train")
    r["frame_ids"][20] = r["frame_ids"][19]  # frame 20 replaced by the earlier record
    write_frames(fdir, r)
    (c, f), _, _ = FlareDataset(pd.DataFrame([r]), fdir, "dual")[0]
    assert c.shape == f.shape == (16, 1, 128, 128)


# ---------------- availability ----------------
def test_missing_frames_are_filtered_out(world):
    samples, fdir = world
    (fdir / "2" / frame_filename(samples["frame_ids"].iloc[1][7])).unlink()      # one frame of HARP 2 gone
    import shutil
    shutil.rmtree(fdir / "3")                                                  # HARP 3 not downloaded at all
    av = filter_available(samples, fdir)
    assert sorted(av["harpnum"]) == [1, 4, 5]
    lines = []
    table = availability_report(samples, av, print_fn=lines.append)
    assert table.set_index("split").loc["train", ["available", "total"]].tolist() == [1, 3]
    assert "train" in lines[0] and "1/3" in lines[0].replace(" ", "") and "0/0/0" not in lines[0]


def test_availability_with_empty_frames_dir(world, tmp_path):
    samples, _ = world
    av = filter_available(samples, tmp_path / "nonexistent")
    assert av.empty
    availability_report(samples, av, print_fn=lambda s: None)  # must not raise at any download stage


def test_load_samples_filters_included_and_checks_frames(tmp_path):
    r = make_row(1, T0, 1, "train")
    csv = pd.DataFrame([{**{k: r[k] for k in ("harpnum", "t", "label", "persistence_label", "split", "frames")},
                         "included": True},
                        {"harpnum": 2, "t": T0, "label": 0, "persistence_label": 0, "split": "val",
                         "included": False, "frames": None}])
    p = tmp_path / "s.csv"
    csv.to_csv(p, index=False)
    df = load_samples(p)
    assert len(df) == 1 and len(df["frame_ids"].iloc[0]) == 31 and df["frame_ids"].iloc[0][-1] == "20240301T1200"
    bad = csv.copy()
    bad.loc[0, "frames"] = ";".join(r["frame_ids"][:-1])
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="31 frames"):
        load_samples(p)


def test_no_harp_in_two_splits_on_real_index():
    p = Path(__file__).resolve().parents[1] / "outputs" / "index" / "samples_final.csv"
    if not p.exists():
        pytest.skip("samples_final.csv not present")
    df = load_samples(p)
    assert df.groupby("harpnum")["split"].nunique().max() == 1
    assert set(df["split"]) <= {"train", "val", "test"}


# ---------------- shapes / dtypes ----------------
def test_shapes_dtypes_and_meta(world):
    samples, fdir = world
    row = samples.iloc[[1]]
    x, y, meta = FlareDataset(row, fdir, "static")[0]
    assert x.shape == (1, 128, 128) and x.dtype == torch.float32
    assert y.dtype == torch.long and y.item() == 1
    assert meta == {"harpnum": 2, "t": "2024-03-01 12:00", "persistence_label": 2}
    x, _, _ = FlareDataset(row, fdir, "single")[0]
    assert x.shape == (31, 1, 128, 128) and x.dtype == torch.float32
    (c, f), _, _ = FlareDataset(row, fdir, "dual")[0]
    assert c.shape == f.shape == (16, 1, 128, 128) and c.dtype == f.dtype == torch.float32


def test_wrong_stored_dtype_or_shape_fails_loudly(world):
    samples, fdir = world
    bad = fdir / "1" / frame_filename(samples["frame_ids"].iloc[0][30])
    np.save(bad, np.zeros((64, 64), dtype=np.float16))
    with pytest.raises(ValueError, match="expected"):
        FlareDataset(samples.iloc[[0]], fdir, "static")[0]
    os_missing = fdir / "1" / frame_filename(samples["frame_ids"].iloc[0][30])
    os_missing.unlink()
    with pytest.raises(FileNotFoundError):
        FlareDataset(samples.iloc[[0]], fdir, "static")[0]


def test_invalid_mode_rejected(world):
    samples, fdir = world
    with pytest.raises(ValueError):
        FlareDataset(samples, fdir, "triple")


# ---------------- loaders ----------------
@pytest.mark.parametrize("mode,shape", [("static", (2, 1, 128, 128)), ("single", (2, 31, 1, 128, 128))])
def test_make_loaders_batches_and_shuffle_policy(world, mode, shape):
    samples, fdir = world
    loaders = make_loaders(mode, 2, 0, samples=samples, frames_dir=fdir)
    assert set(loaders) == {"train", "val", "test"}
    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert isinstance(loaders["val"].sampler, SequentialSampler) and isinstance(loaders["test"].sampler, SequentialSampler)
    x, y, meta = next(iter(loaders["train"]))
    assert tuple(x.shape) == shape and y.shape == (2,) and y.dtype == torch.long
    assert meta["harpnum"].shape == (2,) and len(meta["t"]) == 2 and meta["persistence_label"].shape == (2,)


def test_dual_batch_and_worker_flags(world):
    samples, fdir = world
    loaders = make_loaders("dual", 3, 1, samples=samples, frames_dir=fdir)
    assert loaders["train"].persistent_workers is True and loaders["train"].num_workers == 1
    assert loaders["train"].pin_memory == torch.cuda.is_available()
    assert make_loaders("dual", 3, 0, samples=samples, frames_dir=fdir)["train"].persistent_workers is False
    c, f = next(iter(make_loaders("dual", 3, 0, samples=samples, frames_dir=fdir)["train"]))[0]
    assert tuple(c.shape) == tuple(f.shape) == (3, 16, 1, 128, 128)


def test_loaders_skip_empty_split_but_train_is_required(world):
    samples, fdir = world
    assert "test" not in make_loaders("static", 2, 0, samples=samples[samples["split"] != "test"], frames_dir=fdir)
    with pytest.raises(ValueError, match="training"):
        make_loaders("static", 2, 0, samples=samples[samples["split"] != "train"], frames_dir=fdir)


# ---------------- class weights ----------------
def test_class_weights_formula_and_train_only():
    df = pd.DataFrame({"split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
                       "label": [0] * 6 + [1] * 3 + [2] * 1 + [2] * 5 + [0] * 5})
    w = class_weights(df)
    assert w.dtype == torch.float32
    assert w.tolist() == pytest.approx([10 / (3 * 6), 10 / (3 * 3), 10 / (3 * 1)])
    other = df.copy()
    other.loc[other["split"] != "train", "label"] = 1   # val/test labels must not matter
    assert class_weights(other).tolist() == pytest.approx(w.tolist())


def test_class_weights_missing_class_or_no_train_raises():
    df = pd.DataFrame({"split": ["train"] * 4 + ["val"], "label": [0, 0, 1, 1, 2]})
    with pytest.raises(ValueError, match="no training samples"):
        class_weights(df)
    with pytest.raises(ValueError, match="No training"):
        class_weights(df[df["split"] == "val"])
