"""Paired cluster bootstrap over HARPs (decisions.md §3 decision rule, §11).

Whole HARPs are resampled with replacement (a HARP's samples are strongly correlated), the *same* resample is applied
to every model, and the metric difference is computed per resample. Per-HARP confusion matrices are precomputed, so a
resample is one matrix product.
"""
from __future__ import annotations

import numpy as np

from .metrics import K, confusion_matrix, tss_ge_m

X_MIN_SUPPORT = 10  # decisions.md §11: fewer than 10 test X samples -> X metrics are descriptive only


def _recall(c: int):
    def f(cm: np.ndarray) -> float:
        n = cm[c].sum()
        return float(cm[c, c] / n) if n > 0 else float("nan")
    return f


METRICS = {"tss_ge_m": tss_ge_m, "recall_M": _recall(1), "recall_X": _recall(2)}


def resample_counts(n_clusters: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """[n_boot, n_clusters] integer multiplicities: how often each HARP is drawn (rows sum to n_clusters)."""
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    return np.stack([np.bincount(r, minlength=n_clusters) for r in draws])


def per_harp_confusion(harp, y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """(sorted unique HARPs [H], per-HARP confusion matrices [H,K*K])."""
    harp = np.asarray(harp)
    ids = np.unique(harp)
    cms = np.stack([confusion_matrix(np.asarray(y_true)[harp == h], np.asarray(y_pred)[harp == h]).ravel()
                    for h in ids])
    return ids, cms


def paired_bootstrap(harp, y_true, preds: dict[str, np.ndarray], comparisons: list[tuple[str, str]],
                     metrics: list[str], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> list[dict]:
    """For each (a, b) in ``comparisons`` and metric: difference metric(b) - metric(a).

    ``preds`` maps a model name (incl. "E0") to predicted classes aligned with ``harp``/``y_true``.
    Returns one dict per (comparison, metric): point estimate on the full test set, percentile CI, number of valid
    resamples (a metric is undefined, e.g. no X sample drawn, in the others) and a status:
      * "descriptive only"  -- X metric with fewer than 10 test X samples
      * "supported"          -- CI lower bound > 0 (b better than a)
      * "reverse"            -- CI upper bound < 0 (b worse than a)
      * "not supported"      -- CI contains 0
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    cms, ids = {}, None
    for name, p in preds.items():
        i, c = per_harp_confusion(harp, y_true, p)
        if ids is not None and not np.array_equal(ids, i):
            raise ValueError("HARP sets differ between models")
        ids, cms[name] = i, c
    counts = resample_counts(len(ids), n_boot, rng).astype(float)            # [B,H]
    boot_cm = {n: (counts @ c).reshape(n_boot, K, K) for n, c in cms.items()}  # [B,K,K]
    full_cm = {n: c.sum(axis=0).reshape(K, K) for n, c in cms.items()}
    n_x = int((y_true == 2).sum())
    rows = []
    for a, b in comparisons:
        for m in metrics:
            f = METRICS[m]
            full = f(full_cm[b].astype(float)) - f(full_cm[a].astype(float))
            d = np.array([f(boot_cm[b][i]) - f(boot_cm[a][i]) for i in range(n_boot)])
            ok = d[np.isfinite(d)]
            lo, hi = (np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(ok) else (np.nan, np.nan))
            if m == "recall_X" and n_x < X_MIN_SUPPORT:
                status = "descriptive only"
            elif not len(ok):
                status = "undefined"
            elif lo > 0:
                status = "supported"
            elif hi < 0:
                status = "reverse"
            else:
                status = "not supported"
            rows.append({"comparison": f"{b} - {a}", "metric": m, "estimate": full, "ci_lo": float(lo),
                         "ci_hi": float(hi), "n_valid": int(len(ok)), "n_boot": n_boot, "n_harps": len(ids),
                         "status": status})
    return rows
