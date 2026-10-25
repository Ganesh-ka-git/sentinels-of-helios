"""Classification metrics on 3-class (0 = <M, 1 = M, 2 = X) predictions (decisions.md §11).

The ">=M view" collapses classes 1 and 2 into "positive": a sample is predicted positive if the predicted class is in
{1, 2} and is truly positive if its label is in {1, 2}. Undefined ratios (empty denominator) are NaN, never 0.
"""
from __future__ import annotations

import numpy as np

K = 3
CLASS_NAMES = ("<M", "M", "X")


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def confusion_matrix(y_true, y_pred, k: int = K) -> np.ndarray:
    """k x k integer matrix, rows = true class, columns = predicted class."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"y_true/y_pred must be 1-D and equal length, got {y_true.shape} and {y_pred.shape}")
    if len(y_true) and (y_true.min() < 0 or y_true.max() >= k or y_pred.min() < 0 or y_pred.max() >= k):
        raise ValueError(f"labels must be in 0..{k - 1}")
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def binary_counts(cm: np.ndarray) -> tuple[float, float, float, float]:
    """(TP, FN, FP, TN) of the >=M view from a 3x3 confusion matrix (works on float/weighted matrices too)."""
    return cm[1:, 1:].sum(), cm[1:, 0].sum(), cm[0, 1:].sum(), cm[0, 0]


def tss(tp, fn, fp, tn) -> float:
    """True skill statistic TP/(TP+FN) - FP/(FP+TN); NaN if either class is empty."""
    return _ratio(tp, tp + fn) - _ratio(fp, fp + tn)


def hss(tp, fn, fp, tn) -> float:
    """Heidke skill score 2(TP*TN - FP*FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN)); NaN if the denominator is 0."""
    return _ratio(2.0 * (tp * tn - fp * fn), (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def tss_ge_m(cm: np.ndarray) -> float:
    return tss(*binary_counts(cm))


def hss_ge_m(cm: np.ndarray) -> float:
    return hss(*binary_counts(cm))


def per_class_prf(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class precision, recall, F1 (arrays of length K). Precision is NaN if the class is never predicted,
    recall NaN if it never occurs, F1 = 2TP/(2TP+FP+FN) is NaN only if the class neither occurs nor is predicted."""
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(axis=0).astype(float), cm.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred > 0, tp / pred, np.nan)
        recall = np.where(true > 0, tp / true, np.nan)
        f1 = np.where(pred + true > 0, 2 * tp / (pred + true), np.nan)
    return {"precision": precision, "recall": recall, "f1": f1}


def one_vs_rest_tss(cm: np.ndarray) -> np.ndarray:
    """TSS of each class against the rest, array of length K."""
    out = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        out.append(tss(tp, fn, fp, cm.sum() - tp - fn - fp))
    return np.asarray(out)


def macro_f1(cm: np.ndarray) -> float:
    """Mean F1 over the classes where it is defined."""
    f1 = per_class_prf(cm)["f1"]
    return float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")


def row_normalise(cm: np.ndarray) -> np.ndarray:
    """Rows sum to 1 (NaN rows for classes with no true samples)."""
    cm = cm.astype(float)
    s = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, cm / s, np.nan)


def summarise(y_true, y_pred) -> dict[str, float]:
    """Flat dict of every scalar metric used in the tables."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = binary_counts(cm)
    prf, ovr = per_class_prf(cm), one_vs_rest_tss(cm)
    out = {"n": int(cm.sum()), "tss_ge_m": tss(tp, fn, fp, tn), "hss_ge_m": hss(tp, fn, fp, tn),
           "precision_ge_m": _ratio(tp, tp + fp), "recall_ge_m": _ratio(tp, tp + fn), "macro_f1": macro_f1(cm)}
    for c, name in enumerate(CLASS_NAMES):
        out[f"precision_{name}"], out[f"recall_{name}"] = prf["precision"][c], prf["recall"][c]
        out[f"f1_{name}"], out[f"tss_ovr_{name}"] = prf["f1"][c], ovr[c]
        out[f"support_{name}"] = int(cm[c].sum())
    return out


def probs_to_pred(p: np.ndarray) -> np.ndarray:
    """argmax over the 3 class probabilities [N,3] -> [N]."""
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[1] != K:
        raise ValueError(f"expected [N,{K}] probabilities, got {p.shape}")
    return p.argmax(axis=1)
