"""Tests for Change Logs 15-16: tuned thresholds (validation only), macro TSS, AUC bootstrap, focal loss, E6-E8, queue."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flarecast.dataset import focal_alpha  # noqa: E402
from flarecast.evaluate import report as R  # noqa: E402
from flarecast.evaluate.bootstrap import multi_seed_bootstrap, multi_seed_score_bootstrap  # noqa: E402
from flarecast.evaluate.metrics import confusion_matrix  # noqa: E402
from flarecast.evaluate.thresholds import (GRID, auc_scores, macro_tss, tune_run, tune_thresholds,  # noqa: E402
                                           two_stage_predict)
from flarecast.models import build_model  # noqa: E402
from flarecast.train.runner import MODE_OF, batch_loss, loss_gamma, loss_weights  # noqa: E402


# ---------------------------------------------------------------- two-stage decision
def test_two_stage_toy_example():
    #            p0    p1    p2
    p = np.array([[0.90, 0.08, 0.02],     # s = 0.10 < tau1 -> 0
                  [0.50, 0.40, 0.10],     # s = 0.50 >= tau1, p2/s = 0.2 < tau2 -> M
                  [0.40, 0.10, 0.50],     # s = 0.60, p2/s = 0.833 >= tau2 -> X
                  [0.10, 0.45, 0.45],     # s = 0.90, p2/s = 0.5 >= tau2 -> X (boundary of >=)
                  [1.00, 0.00, 0.00]])    # s = 0 -> 0 (no division by zero)
    pred = two_stage_predict(p, tau1=0.5, tau2=0.5)
    assert pred.tolist() == [0, 1, 2, 2, 0]
    assert two_stage_predict(p, 0.5, 0.99).tolist() == [0, 1, 1, 1, 0]
    assert two_stage_predict(p, 0.95, 0.5).tolist() == [0, 0, 0, 0, 0]


def test_tau1_maximises_tss_and_ties_go_higher():
    # perfectly separable scores: every tau1 in (0.30, 0.70] gives TSS 1 -> the HIGHEST tied grid value (0.70) wins
    y = np.array([0, 0, 0, 1, 1, 2])                # one X sample so that TSS_X is defined
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    p2 = np.array([0, 0, 0, 0, 0, 0.9])
    p = np.stack([1 - s, s - p2, p2], axis=1)
    r = tune_thresholds(y, p, "tss")
    assert r["tau1"] == pytest.approx(0.70) and r["val_tss_ge_m"] == pytest.approx(1.0)


def test_tau2_ties_go_higher_and_grid():
    assert len(GRID) == 99 and GRID[0] == 0.01 and GRID[-1] == 0.99
    # the X sample has p2 = 0 -> never predicted X for any tau2 -> TSS_X is 0 everywhere -> tie -> the highest tau2
    y = np.array([0, 0, 1, 1, 2])
    p = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.1, 0.9, 0.0], [0.1, 0.9, 0.0]])
    r = tune_thresholds(y, p)
    assert r["tau2"] == pytest.approx(0.99) and r["val_tss_X"] == pytest.approx(0.0)


def test_tau2_chosen_for_x_separation():
    # M: p2/(p1+p2) = 0.111, X: 0.889 -> any tau2 in (0.111, 0.889] separates them; ties -> highest grid value 0.88
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0],
                  [0.1, 0.8, 0.1], [0.1, 0.8, 0.1],
                  [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
    r = tune_thresholds(y, p)
    assert r["tau2"] == pytest.approx(0.88) and r["val_tss_X"] == pytest.approx(1.0)


def test_hss_objective_is_optimal_for_hss():
    from flarecast.evaluate.metrics import hss_ge_m, tss_ge_m
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.15).astype(int)
    y[:10] = 2                                       # some X so that TSS_X is defined (all probability in p1)
    s = np.clip(0.35 * (y > 0) + rng.normal(0.3, 0.2, 400), 0.01, 0.99)
    p = np.stack([1 - s, s, np.zeros_like(s)], axis=1)
    t, h = tune_thresholds(y, p, "tss"), tune_thresholds(y, p, "hss")

    def at(tau, f):
        return f(confusion_matrix(y, two_stage_predict(p, tau, 0.99)))
    assert at(h["tau1"], hss_ge_m) >= at(t["tau1"], hss_ge_m) - 1e-12
    assert at(t["tau1"], tss_ge_m) >= at(h["tau1"], tss_ge_m) - 1e-12
    assert h["val_hss_ge_m"] == pytest.approx(at(h["tau1"], hss_ge_m))


# ---------------------------------------------------------------- thresholds never use test data
def _write_preds(d: Path, split: str, seed: int, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(60, int), np.ones(25, int), np.full(15, 2)]
    logits = rng.normal(size=(100, 3)) + np.eye(3)[y] * (1.5 + shift)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    pd.DataFrame({"harpnum": np.arange(100) // 4, "t": np.arange(100), "label": y, "persistence_label": 0,
                  "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2]}).to_csv(d / f"preds_{split}.csv", index=False)


def test_tuning_reads_validation_file_only(tmp_path):
    _write_preds(tmp_path, "val", 1)
    a = tune_run(tmp_path)                           # no test file exists at all -> would raise if it were read
    _write_preds(tmp_path, "test", 2)
    (tmp_path / "preds_test.csv").write_text("garbage,not,a,csv\n", encoding="utf-8")
    (tmp_path / "preds_train.csv").write_text("garbage\n", encoding="utf-8")
    assert tune_run(tmp_path) == a                   # unchanged however the other splits look


def test_thresholds_are_fixed_per_run_and_applied_to_test(tmp_path):
    _write_preds(tmp_path, "val", 1)
    _write_preds(tmp_path, "test", 2, shift=-1.0)
    thr = tune_run(tmp_path)
    test = pd.read_csv(tmp_path / "preds_test.csv")
    pred = R.predict_variant(test, "tuned_tss", thr)
    assert np.array_equal(pred, two_stage_predict(test[R.PROB_COLS].to_numpy(), thr["tau1"], thr["tau2"]))
    assert tune_run(tmp_path) == thr                 # re-tuning after seeing test changes nothing


# ---------------------------------------------------------------- macro TSS / AUC
def test_macro_tss_is_mean_of_one_vs_rest():
    cm = np.array([[70, 5, 1], [4, 15, 2], [1, 2, 5]])
    from flarecast.evaluate.metrics import one_vs_rest_tss
    assert macro_tss(cm) == pytest.approx(one_vs_rest_tss(cm).mean())
    # hand check of class M: TP 15, FN 6, FP 7, TN 70+1+1+5 = 77 -> 15/21 - 7/84
    assert one_vs_rest_tss(cm)[1] == pytest.approx(15 / 21 - 7 / 84)
    perfect = np.diag([10, 5, 3])
    assert macro_tss(perfect) == pytest.approx(1.0)
    assert math.isnan(macro_tss(np.diag([10, 5, 0])))      # X class empty -> undefined, never silently 0


def test_auc_scores_hand_example():
    y = np.array([0, 0, 1, 1, 2])
    p1 = np.array([0.1, 0.4, 0.35, 0.8, 0.0])
    p2 = np.array([0.0, 0.0, 0.0, 0.0, 0.9])
    p = np.stack([1 - p1 - p2, p1, p2], axis=1)            # s = p1 + p2 = [.1, .4, .35, .8, .9]
    a = auc_scores(y, p)
    assert a["roc_auc"] == pytest.approx(1 - 1 / 6)        # positives {.35,.8,.9} vs negatives {.1,.4}: 5 of 6 pairs
    assert 0 < a["pr_auc"] <= 1
    assert math.isnan(auc_scores(np.zeros(5, int), p)["roc_auc"])


def test_score_bootstrap_signs_and_weights():
    rng = np.random.default_rng(0)
    n = 300
    harp = np.arange(n) // 5
    y = (rng.random(n) < 0.2).astype(int)
    good = y * 0.6 + rng.random(n) * 0.5
    bad = rng.random(n)
    scores = {s: {"A": bad, "B": good} for s in (0, 1)}
    rows = multi_seed_score_bootstrap(harp, y, scores, [("A", "B")], "roc_auc", n_boot=200, seed=0)
    assert rows[0]["status"] == "supported" and rows[0]["estimate"] > 0.2
    rev = multi_seed_score_bootstrap(harp, y, scores, [("B", "A")], "roc_auc", n_boot=200, seed=0)
    assert rev[0]["status"] == "reverse"
    # sign disagreement across seeds is never "supported"
    mixed = {0: {"A": bad, "B": good}, 1: {"A": good, "B": bad}}
    assert multi_seed_score_bootstrap(harp, y, mixed, [("A", "B")], "roc_auc", n_boot=200)[0]["status"] != "supported"


def test_pooled_bootstrap_other_metrics():
    harp = np.arange(40) // 4
    y = np.r_[np.zeros(30, int), np.ones(10, int)]
    a = np.zeros(40, int)
    a[30:34] = 1
    b = y.copy()
    rows = multi_seed_bootstrap(harp, y, {0: {"A": a, "B": b}}, [("A", "B")], n_boot=100, metric="recall_ge_m")
    assert rows[0]["metric"] == "recall_ge_m" and rows[0]["estimate"] == pytest.approx(0.6)


# ---------------------------------------------------------------- focal loss
def test_focal_loss_hand_computed():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y = torch.tensor([0, 2])
    alpha = torch.tensor([0.5, 1.0, 2.0])
    p0 = math.exp(2) / (math.exp(2) + 2)                   # softmax prob of the true class of sample 1
    p2 = 1 / (math.exp(1) + 2)                             # ... and of sample 2 (class 2)
    expect = (0.5 * (1 - p0) ** 2 * -math.log(p0) + 2.0 * (1 - p2) ** 2 * -math.log(p2)) / 2
    mean, s, den = batch_loss(logits, y, alpha, gamma=2.0)
    assert mean.item() == pytest.approx(expect, rel=1e-6)
    assert den.item() == 2 and (s / den).item() == pytest.approx(mean.item())


def test_focal_gamma0_no_alpha_is_cross_entropy_and_downweights_easy():
    logits, y = torch.randn(8, 3), torch.randint(0, 3, (8,))
    foc, _, _ = batch_loss(logits, y, None, gamma=0.0)
    ce, _, _ = batch_loss(logits, y, None)
    assert foc.item() == pytest.approx(ce.item(), rel=1e-6)
    assert batch_loss(logits, y, None, gamma=2.0)[0].item() < ce.item()


def test_focal_loss_backward_finite():
    logits = torch.randn(4, 3, requires_grad=True)
    batch_loss(logits, torch.tensor([0, 1, 2, 0]), torch.tensor([0.5, 1.0, 1.5]), 2.0)[0].backward()
    assert torch.isfinite(logits.grad).all()


def _train_samples(counts):
    rows = [{"split": "train", "label": c} for c, n in enumerate(counts) for _ in range(n)]
    rows += [{"split": "val", "label": 2}] * 5
    return pd.DataFrame(rows)


def test_focal_alpha_inverse_sqrt_frequency_mean_one_train_only():
    s = _train_samples([2727, 237, 16])
    a = focal_alpha(s)
    assert a.mean().item() == pytest.approx(1.0, rel=1e-6)
    raw = 1 / np.sqrt(np.array([2727, 237, 16]) / 2980)
    assert a.numpy() == pytest.approx(raw / raw.mean(), rel=1e-5)
    assert a[0] < a[1] < a[2]
    s2 = s.copy()
    s2.loc[s2["split"] != "train", "label"] = 0              # val/test labels must not matter
    assert torch.equal(focal_alpha(s2), a)


def test_models_e6_e7_e8():
    assert MODE_OF["E6"] == "dual" and MODE_OF["E7"] == "single" and MODE_OF["E8"] == "static"
    for new, old in (("E6", "E4"), ("E7", "E2"), ("E8", "E1")):
        assert type(build_model(new)) is type(build_model(old))
    s = _train_samples([50, 10, 4])
    for m in ("E6", "E7", "E8"):
        assert loss_gamma(m) == 2.0
        assert torch.equal(loss_weights(m, s), focal_alpha(s))
    for m in ("E1", "E2", "E3", "E4", "E5"):
        assert loss_gamma(m) is None


# ---------------------------------------------------------------- queue (06c)
def _queue():
    spec = importlib.util.spec_from_file_location("queue_focal", ROOT / "scripts" / "06c_queue_focal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_queue_stage_jobs_order_and_resume(tmp_path):
    q = _queue()
    done, pend = q.stage1_jobs(tmp_path)
    assert [(m, lr, s) for m, lr, s in pend] == [(m, lr, 0) for m in ("E6", "E7", "E8") for lr in (1e-3, 3e-4)]
    chosen = {m: 3e-4 for m in q.ALL_MODELS}
    d2, p2 = q.stage2_jobs(chosen, [1, 2], tmp_path)
    assert len(p2) == 16 and p2[0] == ("E1", 3e-4, 1) and p2[8] == ("E1", 3e-4, 2) and not d2
    (tmp_path / "E1_lr0.0003_s1").mkdir()
    (tmp_path / "E1_lr0.0003_s1" / "DONE").write_text("x")
    assert len(q.stage2_jobs(chosen, [1, 2], tmp_path)[1]) == 15          # DONE runs are skipped
