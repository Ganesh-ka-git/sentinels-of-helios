"""Post-hoc two-stage decision thresholds, tuned on VALIDATION only (decisions.md Change Log 15).

Stage 1: s = P(>=M) = p1 + p2; predicted >=M if s >= tau1.  Stage 2: among predicted >=M, predicted X if
p2 / (p1 + p2) >= tau2, otherwise M.  Samples below tau1 are class 0. tau1 maximises validation TSS>=M (or HSS>=M for the
sensitivity variant); tau2 then maximises validation one-vs-rest TSS_X of the final 3-class prediction at that tau1.
Grid 0.01 ... 0.99, step 0.01; ties -> the HIGHER threshold. Nothing in this module reads a test file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import binary_counts, confusion_matrix, hss, one_vs_rest_tss, tss

PROB_COLS = ["p0", "p1", "p2"]
GRID = np.round(np.arange(1, 100) / 100.0, 2)   # 0.01 ... 0.99


def _probs(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected [N,3] probabilities, got {p.shape}")
    return p


def score_ge_m(p) -> np.ndarray:
    """P(>=M) = p1 + p2 for [N,3] probabilities."""
    p = _probs(p)
    return p[:, 1] + p[:, 2]


def two_stage_predict(p, tau1: float, tau2: float) -> np.ndarray:
    """[N,3] probabilities -> predicted classes {0,1,2} with the two thresholds."""
    p = _probs(p)
    s = p[:, 1] + p[:, 2]
    pos = s >= tau1
    frac_x = np.divide(p[:, 2], s, out=np.zeros_like(s), where=s > 0)
    pred = np.zeros(len(p), dtype=int)
    pred[pos] = 1
    pred[pos & (frac_x >= tau2)] = 2
    return pred


def _argmax_last(values: np.ndarray) -> int:
    """Index of the maximum ignoring NaN; ties -> the last (= highest threshold). Raises if all NaN."""
    if not np.isfinite(values).any():
        raise ValueError("metric undefined for every threshold (a class is empty in the tuning split?)")
    v = np.where(np.isfinite(values), values, -np.inf)
    return int(len(v) - 1 - np.argmax(v[::-1]))


def tune_thresholds(y, p, objective: str = "tss") -> dict[str, float]:
    """(tau1, tau2) from validation labels ``y`` and probabilities ``p`` only.

    ``objective``: "tss" (main) or "hss" (sensitivity) for tau1. Returns the thresholds and the validation values
    they reach.
    """
    if objective not in ("tss", "hss"):
        raise ValueError(f"objective must be 'tss' or 'hss', got {objective!r}")
    y, p = np.asarray(y, dtype=int), _probs(p)
    if len(y) != len(p):
        raise ValueError("labels and probabilities differ in length")
    s, ypos = score_ge_m(p), y >= 1
    f = tss if objective == "tss" else hss
    obj = np.empty(len(GRID))
    for i, t in enumerate(GRID):
        pred_pos = s >= t
        tp, fn = int((pred_pos & ypos).sum()), int((~pred_pos & ypos).sum())
        fp, tn = int((pred_pos & ~ypos).sum()), int((~pred_pos & ~ypos).sum())
        obj[i] = f(tp, fn, fp, tn)
    i1 = _argmax_last(obj)
    tau1 = float(GRID[i1])
    tx = np.array([one_vs_rest_tss(confusion_matrix(y, two_stage_predict(p, tau1, t2)))[2] for t2 in GRID])
    i2 = _argmax_last(tx)
    return {"tau1": tau1, "tau2": float(GRID[i2]), f"val_{objective}_ge_m": float(obj[i1]),
            "val_tss_X": float(tx[i2])}


def threshold_curves(y, p) -> dict[str, np.ndarray]:
    """Validation curves for the threshold figure: TSS>=M and HSS>=M vs tau1 (all grid values), and TSS_X vs tau2 at the
    TSS-optimal tau1."""
    y, p = np.asarray(y, dtype=int), _probs(p)
    s, ypos = score_ge_m(p), y >= 1
    tss_c, hss_c = np.empty(len(GRID)), np.empty(len(GRID))
    for i, t in enumerate(GRID):
        pr = s >= t
        cnt = (int((pr & ypos).sum()), int((~pr & ypos).sum()), int((pr & ~ypos).sum()), int((~pr & ~ypos).sum()))
        tss_c[i], hss_c[i] = tss(*cnt), hss(*cnt)
    tau1 = float(GRID[_argmax_last(tss_c)])
    tx = np.array([one_vs_rest_tss(confusion_matrix(y, two_stage_predict(p, tau1, t2)))[2] for t2 in GRID])
    return {"grid": GRID, "tss_ge_m": tss_c, "hss_ge_m": hss_c, "tss_X": tx, "tau1": tau1}


def macro_tss(cm: np.ndarray) -> float:
    """Mean of the three one-vs-rest TSS values (NaN if any is undefined)."""
    return float(np.mean(one_vs_rest_tss(cm)))


def auc_scores(y, p) -> dict[str, float]:
    """Threshold-free ROC-AUC and PR-AUC (average precision) of P(>=M); NaN if a class is empty."""
    ypos = (np.asarray(y) >= 1).astype(int)
    if ypos.min() == ypos.max():
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    s = score_ge_m(p)
    return {"roc_auc": float(roc_auc_score(ypos, s)), "pr_auc": float(average_precision_score(ypos, s))}


def tune_run(run_dir: str | Path) -> dict[str, float]:
    """Tune from ``preds_val.csv`` of a run (the ONLY file read); returns tau1/tau2 (TSS) and tau1_hss/tau2_hss."""
    val = pd.read_csv(Path(run_dir) / "preds_val.csv")
    main = tune_thresholds(val["label"], val[PROB_COLS].to_numpy(), "tss")
    sens = tune_thresholds(val["label"], val[PROB_COLS].to_numpy(), "hss")
    return {"tau1": main["tau1"], "tau2": main["tau2"], "val_tss_ge_m_at_tau": main["val_tss_ge_m"],
            "val_tss_X_at_tau": main["val_tss_X"], "tau1_hss": sens["tau1"], "tau2_hss": sens["tau2"],
            "val_hss_ge_m_at_tau_hss": sens["val_hss_ge_m"]}


__all__ = ["GRID", "tune_thresholds", "two_stage_predict", "tune_run", "macro_tss", "auc_scores", "score_ge_m",
           "threshold_curves", "binary_counts"]
