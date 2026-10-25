"""Paired cluster bootstrap over HARPs (decisions.md §3 decision rule, §11).

Whole HARPs are resampled with replacement (a HARP's samples are strongly correlated), the *same* resample is applied
to every model, and the metric difference is computed per resample. Per-HARP confusion matrices are precomputed, so a
resample is one matrix product.
"""
from __future__ import annotations

import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import K, binary_counts, confusion_matrix, hss_ge_m, tss_ge_m

X_MIN_SUPPORT = 10  # decisions.md §11: fewer than 10 test X samples -> X metrics are descriptive only


def _recall(c: int):
    def f(cm: np.ndarray) -> float:
        n = cm[c].sum()
        return float(cm[c, c] / n) if n > 0 else float("nan")
    return f


def _precision_ge_m(cm: np.ndarray) -> float:
    tp, _, fp, _ = binary_counts(cm)
    return float(tp / (tp + fp)) if tp + fp > 0 else float("nan")


def _recall_ge_m(cm: np.ndarray) -> float:
    tp, fn, _, _ = binary_counts(cm)
    return float(tp / (tp + fn)) if tp + fn > 0 else float("nan")


METRICS = {"tss_ge_m": tss_ge_m, "recall_M": _recall(1), "recall_X": _recall(2), "hss_ge_m": hss_ge_m,
           "precision_ge_m": _precision_ge_m, "recall_ge_m": _recall_ge_m}


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


def _ci(d: np.ndarray, alpha: float) -> tuple[float, float]:
    ok = d[np.isfinite(d)]
    if not len(ok):
        return float("nan"), float("nan")
    lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _verdict(lo: float, hi: float, e: np.ndarray) -> str:
    """Pooled-CI decision rule (decisions.md §3): CI excludes 0 AND every per-seed estimate has the same sign."""
    all_pos, all_neg = bool(np.all(e > 0)), bool(np.all(e < 0))
    if not np.isfinite(lo):
        return "undefined"
    if lo > 0 and all_pos:
        return "supported"
    if hi < 0 and all_neg:
        return "reverse"
    if lo > 0 or hi < 0:
        return "inconsistent across seeds"
    return "not supported"


def multi_seed_bootstrap(harp, y_true, preds_by_seed: dict[int, dict[str, np.ndarray]],
                         comparisons: list[tuple[str, str]], n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05, metric: str = "tss_ge_m") -> list[dict]:
    """``metric`` (default TSS>=M) difference metric(b) - metric(a) over several training seeds (decisions.md §3).

    ``preds_by_seed[s][model]`` are predicted classes aligned with ``harp``/``y_true`` (the same test samples for every
    seed; "E0" may be identical across seeds). One HARP resample is drawn per bootstrap iteration and applied to every
    seed and model. Per seed: point estimate and percentile CI. Pooled: the per-resample difference is the mean over
    seeds, the point estimate the mean of the per-seed estimates, the CI the percentile CI of the pooled differences.
    Verdict: "supported" only if the pooled CI lower bound > 0 AND the per-seed point estimates are all > 0;
    "reverse" likewise for < 0; "inconsistent across seeds" if the pooled CI excludes 0 but the per-seed signs differ;
    otherwise "not supported".
    """
    y_true = np.asarray(y_true)
    seeds = sorted(preds_by_seed)
    if not seeds:
        raise ValueError("No seeds given")
    f = METRICS[metric]
    rng = np.random.default_rng(seed)
    cms, ids = {}, None
    for s in seeds:
        for name, p in preds_by_seed[s].items():
            i, c = per_harp_confusion(harp, y_true, p)
            if ids is not None and not np.array_equal(ids, i):
                raise ValueError("HARP sets differ between models/seeds")
            ids, cms[(s, name)] = i, c
    counts = resample_counts(len(ids), n_boot, rng).astype(float)
    boot = {k: (counts @ c).reshape(n_boot, K, K) for k, c in cms.items()}
    full = {k: c.sum(axis=0).reshape(K, K).astype(float) for k, c in cms.items()}
    rows = []
    for a, b in comparisons:
        est, d_seed = {}, {}
        for s in seeds:
            est[s] = f(full[(s, b)]) - f(full[(s, a)])
            d_seed[s] = np.array([f(boot[(s, b)][i]) - f(boot[(s, a)][i]) for i in range(n_boot)])
        pooled = np.mean([d_seed[s] for s in seeds], axis=0)
        lo, hi = _ci(pooled, alpha)
        e = np.array([est[s] for s in seeds])
        all_pos, all_neg = bool(np.all(e > 0)), bool(np.all(e < 0))
        status = _verdict(lo, hi, e)
        row = {"comparison": f"{b} - {a}", "metric": metric, "n_seeds": len(seeds),
               "estimate": float(e.mean()), "ci_lo": lo, "ci_hi": hi}
        for s in seeds:
            slo, shi = _ci(d_seed[s], alpha)
            row[f"est_s{s}"] = float(est[s])
            row[f"ci_s{s}"] = f"[{slo:.3f}, {shi:.3f}]"
        row.update({"sign_consistent": all_pos or all_neg, "n_boot": n_boot, "n_harps": len(ids), "status": status})
        rows.append(row)
    return rows


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


SCORE_STATS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}


def multi_seed_score_bootstrap(harp, y_true, scores_by_seed: dict[int, dict[str, np.ndarray]],
                               comparisons: list[tuple[str, str]], stat: str = "roc_auc", n_boot: int = 2000,
                               seed: int = 0, alpha: float = 0.05) -> list[dict]:
    """Threshold-free statistic (``roc_auc`` or ``pr_auc`` of the >=M score) difference stat(b) - stat(a).

    Same design and decision rule as :func:`multi_seed_bootstrap` (paired HARP resampling with the same ``seed`` and
    hence the same resamples; pooled mean over seeds; "supported" needs the pooled CI above 0 AND every seed's estimate
    positive). A resample is applied as sample weights (HARP multiplicity), which equals duplicating the HARP's rows.
    ``scores_by_seed[s][model]`` is a score aligned with ``harp``/``y_true`` ("E0" may be its 0/1 persistence call).
    """
    f = SCORE_STATS[stat]
    harp, y = np.asarray(harp), (np.asarray(y_true) >= 1).astype(int)
    seeds = sorted(scores_by_seed)
    if not seeds:
        raise ValueError("No seeds given")
    ids = np.unique(harp)
    idx = np.searchsorted(ids, harp)
    counts = resample_counts(len(ids), n_boot, np.random.default_rng(seed))    # [B,H]
    w = counts[:, idx].astype(float)                                          # [B,N] per-sample multiplicity
    need = sorted({m for a, b in comparisons for m in (a, b)})

    def boot_stat(score: np.ndarray) -> np.ndarray:
        out = np.full(n_boot, np.nan)
        for i in range(n_boot):
            yw = y[w[i] > 0]
            if yw.min() == yw.max():
                continue
            out[i] = f(y, score, sample_weight=w[i])
        return out

    full = {(s, m): float(f(y, scores_by_seed[s][m])) for s in seeds for m in need}
    boot = {(s, m): boot_stat(np.asarray(scores_by_seed[s][m], dtype=float)) for s in seeds for m in need}
    rows = []
    for a, b in comparisons:
        est = np.array([full[(s, b)] - full[(s, a)] for s in seeds])
        d_seed = {s: boot[(s, b)] - boot[(s, a)] for s in seeds}
        lo, hi = _ci(np.mean([d_seed[s] for s in seeds], axis=0), alpha)
        row = {"comparison": f"{b} - {a}", "metric": stat, "n_seeds": len(seeds), "estimate": float(est.mean()),
               "ci_lo": lo, "ci_hi": hi}
        for s, e in zip(seeds, est):
            slo, shi = _ci(d_seed[s], alpha)
            row[f"est_s{s}"], row[f"ci_s{s}"] = float(e), f"[{slo:.3f}, {shi:.3f}]"
        row.update({"sign_consistent": bool(np.all(est > 0) or np.all(est < 0)), "n_boot": n_boot,
                    "n_harps": len(ids), "status": _verdict(lo, hi, est)})
        rows.append(row)
    return rows
