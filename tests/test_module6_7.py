"""Unit tests for Modules 6 and 7 (training runner pieces, metrics, bootstrap, selection). Synthetic data, CPU only."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flarecast.evaluate import metrics as M  # noqa: E402
from flarecast.evaluate.bootstrap import paired_bootstrap, resample_counts  # noqa: E402
from flarecast.evaluate.report import select_best  # noqa: E402
from flarecast.train.runner import batch_loss, loss_weights, run_name, train_run  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_module4 import make_row, write_frames  # noqa: E402


# ---------------------------------------------------------------- metrics on hand-computed matrices
# rows = true (<M, M, X), cols = predicted
CM = np.array([[80, 10, 2],
               [6, 3, 1],
               [1, 1, 2]])
# >=M view: TP = 3+1+1+2 = 7, FN = 6+1 = 7, FP = 10+2 = 12, TN = 80


def test_binary_counts_hand_computed():
    assert tuple(M.binary_counts(CM)) == (7, 7, 12, 80)


def test_tss_hss_hand_computed():
    tp, fn, fp, tn = 7, 7, 12, 80
    assert M.tss_ge_m(CM) == pytest.approx(7 / 14 - 12 / 92)
    expected_hss = 2 * (7 * 80 - 12 * 7) / ((14) * (7 + 80) + (19) * (12 + 80))
    assert M.hss_ge_m(CM) == pytest.approx(expected_hss)
    assert M.tss(tp, fn, fp, tn) == pytest.approx(M.tss_ge_m(CM))


def test_perfect_and_constant_predictions():
    y = np.array([0, 0, 1, 2, 0, 1])
    assert M.tss_ge_m(M.confusion_matrix(y, y)) == 1.0
    assert M.hss_ge_m(M.confusion_matrix(y, y)) == 1.0
    assert M.tss_ge_m(M.confusion_matrix(y, np.zeros_like(y))) == 0.0


def test_undefined_is_nan_not_zero():
    cm = M.confusion_matrix(np.array([0, 0, 0]), np.array([0, 1, 0]))   # no positives
    assert np.isnan(M.tss_ge_m(cm))


def test_per_class_precision_recall_f1():
    p = M.per_class_prf(CM)
    assert p["precision"][1] == pytest.approx(3 / 14)
    assert p["recall"][1] == pytest.approx(3 / 10)
    assert p["f1"][1] == pytest.approx(2 * 3 / (14 + 10))
    assert p["recall"][2] == pytest.approx(2 / 4)
    assert M.macro_f1(CM) == pytest.approx(np.nanmean(p["f1"]))


def test_one_vs_rest_tss():
    # class M vs rest: TP=3, FN=7, FP=11, TN=80+10+2... computed from the matrix
    tp, fn = 3, 7
    fp = 10 + 1
    tn = CM.sum() - tp - fn - fp
    assert M.one_vs_rest_tss(CM)[1] == pytest.approx(tp / (tp + fn) - fp / (fp + tn))


def test_row_normalise_and_probs_to_pred():
    rn = M.row_normalise(CM)
    assert np.allclose(rn.sum(axis=1), 1)
    assert M.probs_to_pred(np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])).tolist() == [1, 0]
    with pytest.raises(ValueError):
        M.confusion_matrix([0, 3], [0, 1])


# ---------------------------------------------------------------- bootstrap
def test_resample_counts_whole_clusters():
    c = resample_counts(7, 50, np.random.default_rng(0))
    assert c.shape == (50, 7) and (c.sum(axis=1) == 7).all() and c.dtype.kind == "i"


def _toy():
    # 6 HARPs with different numbers of samples; model B is perfect, model A predicts all zeros
    harp = np.repeat([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 20])
    rng = np.random.default_rng(1)
    y = rng.choice([0, 1, 2], size=len(harp), p=[0.5, 0.35, 0.15])
    return harp, y, {"E0": np.zeros_like(y), "A": np.zeros_like(y), "B": y.copy()}


def test_bootstrap_resamples_whole_harps(monkeypatch):
    """The resampled sample count is sum_h count_h * size_h (whole HARPs), not a row-level draw of N rows."""
    import flarecast.evaluate.bootstrap as B
    harp, y, preds = _toy()
    seen = []
    orig = B.resample_counts

    def spy(n, nb, rng):
        c = orig(n, nb, rng)
        seen.append(c)
        return c

    monkeypatch.setattr(B, "resample_counts", spy)
    paired_bootstrap(harp, y, preds, [("A", "B")], ["tss_ge_m"], n_boot=200, seed=3)
    counts, sizes = seen[0], np.bincount(harp)[1:]
    totals = counts @ sizes
    assert (counts.sum(axis=1) == 6).all()
    assert totals.std() > 0                      # total N varies, as only cluster draws are fixed (6)


def test_bootstrap_ci_and_status():
    harp, y, preds = _toy()
    rows = paired_bootstrap(harp, y, preds, [("A", "B"), ("B", "A")], ["tss_ge_m", "recall_X"], n_boot=500, seed=0)
    r = {(x["comparison"], x["metric"]): x for x in rows}
    ab = r[("B - A", "tss_ge_m")]
    assert ab["estimate"] == pytest.approx(1.0) and ab["status"] == "supported" and ab["ci_lo"] > 0
    assert r[("A - B", "tss_ge_m")]["status"] == "reverse"
    again = paired_bootstrap(harp, y, preds, [("A", "B")], ["tss_ge_m"], n_boot=500, seed=0)[0]
    assert again["ci_lo"] == ab["ci_lo"] and again["ci_hi"] == ab["ci_hi"]      # fixed seed -> reproducible


def test_x_metrics_descriptive_when_fewer_than_10_x():
    harp, y, preds = _toy()
    y = np.where(y == 2, 1, y)
    y[:3] = 2                                                                    # 3 X samples only
    preds = {"A": np.zeros_like(y), "B": y.copy()}
    row = paired_bootstrap(harp, y, preds, [("A", "B")], ["recall_X"], n_boot=100, seed=0)[0]
    assert row["status"] == "descriptive only"


# ---------------------------------------------------------------- selection uses validation only
def test_selection_uses_val_only():
    runs = pd.DataFrame({"model": ["E1", "E1", "E2", "E2"], "lr": [1e-3, 3e-4, 1e-3, 3e-4],
                         "val_tss_ge_m": [0.30, 0.40, 0.50, 0.20], "test_tss_ge_m": [0.9, 0.0, 0.0, 0.9],
                         "run_dir": list("abcd")})
    sel = select_best(runs).set_index("model")
    assert sel.loc["E1", "lr"] == 3e-4 and sel.loc["E2", "lr"] == 1e-3        # test column would pick the others


# ---------------------------------------------------------------- class weights / loss
def _samples():
    rows = [make_row(i, pd.Timestamp("2024-03-01 12:00"), lab, sp) for i, (lab, sp) in
            enumerate([(0, "train"), (0, "train"), (1, "train"), (2, "train"), (0, "val"), (1, "test")])]
    return pd.DataFrame(rows)


def test_class_weights_train_only_and_e5_only():
    s = _samples()
    w = loss_weights("E5", s)
    assert w.tolist() == pytest.approx([4 / (3 * 2), 4 / (3 * 1), 4 / (3 * 1)])
    s2 = s.copy()
    s2.loc[s2["split"] != "train", "label"] = 2                                # change val/test labels
    assert torch.equal(loss_weights("E5", s2), w)
    assert loss_weights("E4", s) is None and loss_weights("E1", s) is None


def test_batch_loss_matches_weighted_cross_entropy():
    logits, y = torch.randn(6, 3), torch.tensor([0, 1, 2, 0, 0, 1])
    w = torch.tensor([0.5, 2.0, 5.0])
    mean, s, den = batch_loss(logits, y, w)
    assert mean.item() == pytest.approx(torch.nn.functional.cross_entropy(logits, y, weight=w).item(), rel=1e-6)
    assert (s / den).item() == pytest.approx(mean.item())


# ---------------------------------------------------------------- resume + end-to-end CPU run
def _load_run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "scripts" / "06_run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_all_skips_done(tmp_path):
    mod = _load_run_all()
    (tmp_path / run_name("E1", 1e-3, 0)).mkdir()
    (tmp_path / run_name("E1", 1e-3, 0) / "DONE").write_text("x")
    (tmp_path / run_name("E1", 3e-4, 0)).mkdir()                                # partial: no DONE
    done, pending = mod.pending_runs(["E1", "E2"], [1e-3, 3e-4], [0], tmp_path)
    assert done == [("E1", 1e-3, 0)]
    assert ("E1", 1e-3, 0) not in pending and ("E1", 3e-4, 0) in pending and len(pending) == 3


def test_train_run_end_to_end_cpu_and_skip(tmp_path):
    t0 = pd.Timestamp("2024-03-01 12:00")
    rows = []
    for h, (lab, sp) in enumerate([(0, "train"), (1, "train"), (0, "train"), (2, "train"), (0, "val"), (1, "val"),
                                   (0, "test"), (1, "test")], start=1):
        rows.append(make_row(h, t0, lab, sp))
    samples = pd.DataFrame(rows)
    for r in rows:
        write_frames(tmp_path / "frames", r)
    cfg = {"paths": {"outputs_dir": tmp_path / "out", "frames_dir": tmp_path / "frames"},
           "loader": {"batch_size": 4, "num_workers": 0, "use_cache": False}}
    d = train_run("E1", 1e-3, 0, cfg=cfg, max_epochs=2, tag="t", samples=samples, frames_dir=tmp_path / "frames")
    assert d.name == "E1_lr0.001_s0_t"
    for f in ("history.csv", "best.pt", "cost.json", "run.json", "DONE", "preds_train.csv", "preds_val.csv",
              "preds_test.csv"):
        assert (d / f).exists(), f
    p = pd.read_csv(d / "preds_test.csv")
    assert list(p.columns) == ["harpnum", "t", "label", "persistence_label", "p0", "p1", "p2"]
    assert np.allclose(p[["p0", "p1", "p2"]].sum(axis=1), 1, atol=1e-5)
    mtime = (d / "DONE").stat().st_mtime_ns
    train_run("E1", 1e-3, 0, cfg=cfg, max_epochs=2, tag="t", samples=samples, frames_dir=tmp_path / "frames")
    assert (d / "DONE").stat().st_mtime_ns == mtime                              # DONE -> skipped, not retrained
