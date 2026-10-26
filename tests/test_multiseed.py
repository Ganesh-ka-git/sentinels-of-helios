"""Tests for the multi-seed evaluation (06b_queue_seeds.py, report seed tables, multi_seed_bootstrap). Synthetic, CPU only."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flarecast.evaluate import report as R  # noqa: E402
from flarecast.evaluate.bootstrap import multi_seed_bootstrap  # noqa: E402
from flarecast.train.runner import run_name  # noqa: E402

N = 100          # samples per split; the first 20 are class 1 (M), 4 samples per HARP
N_POS = 20


def _labels() -> np.ndarray:
    y = np.zeros(N, dtype=int)
    y[:N_POS] = 1
    return y


def _preds_df(hits: int) -> pd.DataFrame:
    """Predictions detecting the first ``hits`` positives, no false alarms (TSS>=M = hits / 20)."""
    y = _labels()
    pred = np.zeros(N, dtype=int)
    pred[:hits] = 1
    p = np.eye(3)[pred]
    return pd.DataFrame({"harpnum": np.arange(N) // 4, "t": np.arange(N), "label": y, "persistence_label": 0,
                         "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2]})


def _write_run(runs_dir: Path, model: str, lr: float, seed: int, val_hits: int, test_hits: int) -> Path:
    d = runs_dir / run_name(model, lr, seed)
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"model": model, "lr": lr, "seed": seed, "tag": None}), encoding="utf-8")
    (d / "cost.json").write_text(json.dumps({"best_epoch": 3, "epochs_run": 5}), encoding="utf-8")
    _preds_df(val_hits).to_csv(d / "preds_val.csv", index=False)
    _preds_df(test_hits).to_csv(d / "preds_test.csv", index=False)
    (d / "DONE").write_text("x")
    return d


def _load_queue():
    spec = importlib.util.spec_from_file_location("queue_seeds", ROOT / "scripts" / "06b_queue_seeds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- mean / sd over seeds
def test_seed_mean_sd_hand_computed(tmp_path):
    for s, hits in enumerate([10, 12, 14]):                  # TSS>=M = 0.5, 0.6, 0.7
        _write_run(tmp_path, "E1", 1e-3, s, val_hits=hits, test_hits=hits)
    sel0 = R.select_best(R.discover_runs(tmp_path, None, 0))
    sel = R.select_seed_runs(tmp_path, None, sel0, [0, 1, 2])
    long, agg = R.seed_metric_table(sel)
    assert len(long) == 6                                    # 3 seeds x (val, test)
    row = agg[(agg["model"] == "E1") & (agg["split"] == "test")].iloc[0]
    assert row["n_seeds"] == 3
    assert row["tss_ge_m_mean"] == pytest.approx(0.6)
    assert row["tss_ge_m_sd"] == pytest.approx(0.1)          # ddof=1 of (0.5, 0.6, 0.7)
    assert row["recall_M_mean"] == pytest.approx(0.6)        # no X in these data, M recall = TSS here
    assert "0.600 ± 0.100" in R.seed_md_table(agg)["tss_ge_m"].tolist()


# ---------------------------------------------------------------- selection on validation only
def test_selection_val_only_and_lr_fixed_across_seeds(tmp_path):
    # seed 0: lr 1e-3 has the better VALIDATION score but a terrible test score; lr 3e-4 the reverse
    _write_run(tmp_path, "E1", 1e-3, 0, val_hits=16, test_hits=2)
    _write_run(tmp_path, "E1", 3e-4, 0, val_hits=8, test_hits=20)
    # seed 1: the other lr looks better on val, but the lr must stay the one chosen on seed 0
    _write_run(tmp_path, "E1", 1e-3, 1, val_hits=4, test_hits=4)
    _write_run(tmp_path, "E1", 3e-4, 1, val_hits=20, test_hits=20)
    sel0 = R.select_best(R.discover_runs(tmp_path, None, 0))
    assert sel0["lr"].tolist() == [1e-3]
    sel = R.select_seed_runs(tmp_path, None, sel0, [0, 1])
    assert sel[1]["lr"].tolist() == [1e-3]
    assert Path(sel[1].iloc[0]["run_dir"]).name == run_name("E1", 1e-3, 1)
    with pytest.raises(SystemExit, match="Missing finished run"):
        R.select_seed_runs(tmp_path, None, sel0, [0, 1, 2])    # seed 2 was never run


def test_queue_waits_for_seed0_and_chooses_val_best_lr(tmp_path):
    q = _load_queue()
    assert len(q.seed0_pending(tmp_path)) == 10               # nothing finished yet
    best = {"E1": 1e-3, "E2": 3e-4, "E3": 1e-3, "E4": 3e-4, "E5": 1e-3}
    for m in q.MODELS:
        for lr in q.LRS:
            good = lr == best[m]
            # test is deliberately anti-correlated with validation: it must not influence the choice
            _write_run(tmp_path, m, lr, 0, val_hits=16 if good else 6, test_hits=2 if good else 18)
    assert q.seed0_pending(tmp_path) == []
    chosen = q.choose_lrs(tmp_path)
    assert chosen == best
    # one seed-1 run already DONE -> skipped; 10 jobs in total, only for the chosen lr
    _write_run(tmp_path, "E1", 1e-3, 1, 10, 10)
    done, pending = q.seed_jobs(chosen, [1, 2], tmp_path)
    assert done == [("E1", 1e-3, 1)]
    assert len(pending) == 9 and all(lr == best[m] and s in (1, 2) for m, lr, s in pending)


# ---------------------------------------------------------------- sign-consistency verdict
def _bootstrap_inputs(b_hits_per_seed: list[int]):
    """30 HARPs x 4 samples, positives = first 20 samples (HARPs 0-4). a predicts nothing; b detects ``hits``."""
    n, y = 120, np.zeros(120, dtype=int)
    y[:20] = 1
    harp = np.arange(n) // 4
    a = np.zeros(n, dtype=int)
    preds = {}
    for s, hits in enumerate(b_hits_per_seed):
        b = np.zeros(n, dtype=int)
        pos = np.flatnonzero(y == 1)
        # detect whole HARPs first so that resampling does not make every draw identical
        b[pos[:hits]] = 1
        preds[s] = {"a": a, "b": b}
    return harp, y, preds


def test_verdict_supported_when_pooled_ci_excludes_0_and_all_seed_signs_positive():
    harp, y, preds = _bootstrap_inputs([20, 20, 16])
    row = multi_seed_bootstrap(harp, y, preds, [("a", "b")], n_boot=500, seed=0)[0]
    assert row["comparison"] == "b - a" and row["n_seeds"] == 3
    assert row["est_s0"] == pytest.approx(1.0) and row["est_s2"] == pytest.approx(0.8)
    assert row["estimate"] == pytest.approx((1.0 + 1.0 + 0.8) / 3)
    assert row["ci_lo"] > 0 and row["sign_consistent"] and row["status"] == "supported"


def test_verdict_not_supported_if_one_seed_has_no_improvement():
    # seed 2: b == a (difference exactly 0). Pooled CI still excludes 0, but the sign is not positive in all seeds
    harp, y, preds = _bootstrap_inputs([20, 20, 0])
    row = multi_seed_bootstrap(harp, y, preds, [("a", "b")], n_boot=500, seed=0)[0]
    assert row["ci_lo"] > 0                                   # pooled analysis alone would say "supported"
    assert not row["sign_consistent"]
    assert row["status"] == "inconsistent across seeds"


def test_verdict_reverse_and_not_supported():
    harp, y, preds = _bootstrap_inputs([20, 20, 20])
    rev = multi_seed_bootstrap(harp, y, preds, [("b", "a")], n_boot=500, seed=0)[0]
    assert rev["status"] == "reverse" and rev["estimate"] == pytest.approx(-1.0)
    harp, y, preds = _bootstrap_inputs([0, 0, 0])
    nul = multi_seed_bootstrap(harp, y, preds, [("a", "b")], n_boot=200, seed=0)[0]
    assert nul["status"] == "not supported" and not nul["sign_consistent"]


def test_pooled_ci_uses_same_resample_for_all_seeds():
    # identical seeds -> pooled differences equal each per-seed difference, so the pooled CI equals each seed's CI
    harp, y, preds = _bootstrap_inputs([12, 12, 12])
    row = multi_seed_bootstrap(harp, y, preds, [("a", "b")], n_boot=300, seed=1)[0]
    assert row["ci_s0"] == row["ci_s1"] == row["ci_s2"] == f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"


def test_x_flag_table_counts(tmp_path):
    """3 test X samples: argmax-M for two, argmax-X for one; checks X flagged as >=M and the number of X predictions."""
    d = tmp_path / run_name("E1", 1e-3, 0)
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"model": "E1", "lr": 1e-3, "seed": 0, "tag": None}), encoding="utf-8")
    probs = [[.1, .8, .1], [.1, .7, .2], [.1, .3, .6], [.9, .05, .05], [.2, .6, .2], [.9, .05, .05]]
    df = pd.DataFrame(probs, columns=["p0", "p1", "p2"])
    df["label"] = [2, 2, 2, 0, 1, 0]
    df["harpnum"], df["t"], df["persistence_label"] = np.arange(6), np.arange(6), 0
    df.to_csv(d / "preds_test.csv", index=False)
    sel = pd.DataFrame([{"model": "E1", "lr": 1e-3, "seed": 0, "run_dir": str(d)}])
    thr = pd.DataFrame([{"model": "E1", "lr": 1e-3, "seed": 0, "tau1": 0.5, "tau2": 0.5, "tau1_hss": 0.5, "tau2_hss": 0.5}])
    t = R.x_flag_table({0: sel}, thr).set_index("variant")
    a = t.loc["argmax"]
    assert (a["n_X_true"], a["X_flagged_ge_m"], a["X_predicted_X"], a["n_pred_X"]) == (3, 3, 1, 1)
    # tuned: s = .9, .9, .9, .1, .8, .1 -> >=M for 0,1,2,4; X if p2/s >= .5 -> only sample 2 (.667)
    k = t.loc["tuned_tss"]
    assert (k["X_flagged_ge_m"], k["X_predicted_X"], k["n_pred_X"], k["n_pred_ge_m"]) == (3, 1, 1, 4)
