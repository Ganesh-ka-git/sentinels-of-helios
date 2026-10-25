"""Module 7 — run discovery, model selection (validation only), tables and figures."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import CLASS_NAMES, confusion_matrix, probs_to_pred, row_normalise, summarise  # noqa: E402
from .thresholds import (GRID, auc_scores, macro_tss, score_ge_m, threshold_curves, tune_run,  # noqa: E402
                         two_stage_predict)

log = logging.getLogger(__name__)

MODELS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
COLORS = {"E0": "#666666", "E1": "#0072B2", "E2": "#E69F00", "E3": "#009E73", "E4": "#CC79A7", "E5": "#D55E00",
          "E6": "#56B4E9", "E7": "#332288", "E8": "#999933"}
FOCAL = ("E6", "E7", "E8")
TABLE_COLS = ["tss_ge_m", "hss_ge_m", "macro_f1", "recall_M", "recall_X"]
PROB_COLS = ["p0", "p1", "p2"]


# ---------------------------------------------------------------- discovery / selection
def discover_runs(runs_dir: Path, tag: str | None = None, seed: int = 0) -> pd.DataFrame:
    """One row per finished (DONE) run with the given tag and seed. Reads run.json, never the directory name."""
    rows = []
    for d in sorted(Path(runs_dir).glob("*")):
        if not (d / "DONE").exists() or not (d / "run.json").exists():
            continue
        r = json.loads((d / "run.json").read_text(encoding="utf-8"))
        if r.get("tag") != tag or r["seed"] != seed:
            continue
        cost = json.loads((d / "cost.json").read_text(encoding="utf-8"))
        val = pd.read_csv(d / "preds_val.csv")
        m = summarise(val["label"], probs_to_pred(val[PROB_COLS].to_numpy()))
        rows.append({"model": r["model"], "lr": r["lr"], "seed": r["seed"], "run_dir": str(d),
                     "val_tss_ge_m": m["tss_ge_m"], "best_epoch": cost["best_epoch"],
                     "epochs_run": cost["epochs_run"]})
    return pd.DataFrame(rows, columns=["model", "lr", "seed", "run_dir", "val_tss_ge_m", "best_epoch", "epochs_run"])


def select_best(runs: pd.DataFrame) -> pd.DataFrame:
    """Per model the run with the highest VALIDATION TSS>=M (ties -> smaller lr). Only ``val_tss_ge_m`` is used."""
    if runs.empty:
        return runs
    ordered = runs.sort_values(["model", "val_tss_ge_m", "lr"], ascending=[True, False, True])
    return ordered.groupby("model", sort=True).head(1).reset_index(drop=True)


SEED_METRICS = ["tss_ge_m", "hss_ge_m", "recall_M", "recall_X"]


def select_seed_runs(runs_dir: Path, tag: str | None, selected0: pd.DataFrame, seeds: list[int]) -> dict[int, pd.DataFrame]:
    """Per seed the runs of the (model, lr) pairs chosen on seed 0 by validation TSS>=M (no re-selection per seed).

    Fails loudly if a selected (model, lr) has no finished run for a requested seed.
    """
    out = {}
    for s in seeds:
        runs = selected0 if s == 0 else discover_runs(runs_dir, tag, s)
        rows = []
        for r in selected0.itertuples():
            hit = runs[(runs["model"] == r.model) & np.isclose(runs["lr"], r.lr)]
            if hit.empty:
                raise SystemExit(f"Missing finished run: model={r.model} lr={r.lr:g} seed={s} tag={tag!r}")
            rows.append(hit.iloc[0])
        out[s] = pd.DataFrame(rows).reset_index(drop=True)
    return out


def seed_metric_table(sel_by_seed: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(long table: model, split, seed, metrics; aggregated: mean and sd (ddof=1) over seeds) for val and test."""
    rows = []
    for s, sel in sel_by_seed.items():
        for r in sel.itertuples():
            for sp in ("val", "test"):
                d = load_preds(r.run_dir, sp)
                m = summarise(d["label"], pred_of(d))
                rows.append({"model": r.model, "lr": r.lr, "split": sp, "seed": s, **{k: m[k] for k in SEED_METRICS}})
    long = pd.DataFrame(rows)
    g = long.groupby(["model", "split"], sort=True)
    agg = g[SEED_METRICS].agg(["mean", "std"])
    agg.columns = [f"{k}_{'sd' if f == 'std' else f}" for k, f in agg.columns]
    agg.insert(0, "n_seeds", g.size())
    agg.insert(0, "lr", g["lr"].first())
    return long, agg.reset_index()


def seed_md_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Compact 'mean ± sd' strings for the markdown summary."""
    out = agg[["model", "split", "lr", "n_seeds"]].copy()
    for k in SEED_METRICS:
        out[k] = [f"{m:.3f} ± {s:.3f}" if np.isfinite(m) and np.isfinite(s) else "n/a"
                  for m, s in zip(agg[f"{k}_mean"], agg[f"{k}_sd"])]
    return out


def load_preds(run_dir: str | Path, split: str) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / f"preds_{split}.csv")


def pred_of(df: pd.DataFrame) -> np.ndarray:
    return probs_to_pred(df[PROB_COLS].to_numpy())


def align_on_samples(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Sort every model's predictions by (harpnum, t) and check they cover exactly the same samples and labels."""
    out = {n: d.sort_values(["harpnum", "t"]).reset_index(drop=True) for n, d in dfs.items()}
    ref_name = next(iter(out))
    ref = out[ref_name]
    for n, d in out.items():
        if not (d[["harpnum", "t"]].equals(ref[["harpnum", "t"]]) and d["label"].equals(ref["label"])):
            raise ValueError(f"Prediction files of {n} and {ref_name} do not cover the same samples/labels")
    return out


# ---------------------------------------------------------------- tables
def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    def fmt(v, col=""):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "n/a"
            return f"{v:g}" if col == "lr" else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v, c) for v, c in zip(r, df.columns)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def split_table(selected: pd.DataFrame) -> pd.DataFrame:
    """Train / val / test metrics for E0 and every selected model (over/underfitting check)."""
    splits = ("train", "val", "test")
    rows = []
    for sp in splits:  # E0 uses the persistence label stored in the first selected model's prediction file
        d = load_preds(selected.iloc[0]["run_dir"], sp)
        rows.append({"model": "E0", "lr": np.nan, "split": sp, **{k: summarise(d["label"], d["persistence_label"])[k]
                                                               for k in TABLE_COLS}})
    for r in selected.itertuples():
        for sp in splits:
            d = load_preds(r.run_dir, sp)
            m = summarise(d["label"], pred_of(d))
            rows.append({"model": r.model, "lr": r.lr, "split": sp, **{k: m[k] for k in TABLE_COLS}})
    return pd.DataFrame(rows)


def test_tables(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """(main table E0-E5, per-class long table, confusion long table, aligned test predictions incl. 'E0')."""
    raw = {r.model: load_preds(r.run_dir, "test") for r in selected.itertuples()}
    aligned = align_on_samples(raw)
    first = next(iter(aligned.values()))
    y = first["label"].to_numpy()
    preds = {"E0": first["persistence_label"].to_numpy(), **{n: pred_of(d) for n, d in aligned.items()}}
    main, per_class, conf = [], [], []
    for n, p in preds.items():
        m = summarise(y, p)
        main.append({"model": n, **{k: m[k] for k in TABLE_COLS}, "precision_ge_m": m["precision_ge_m"],
                     "tss_ovr_X": m["tss_ovr_X"], "n": m["n"], "n_X": m["support_X"],
                     "X_note": "descriptive only" if m["support_X"] < 10 else ""})
        for c in CLASS_NAMES:
            per_class.append({"model": n, "class": c, "precision": m[f"precision_{c}"], "recall": m[f"recall_{c}"],
                              "f1": m[f"f1_{c}"], "tss_ovr": m[f"tss_ovr_{c}"], "support": m[f"support_{c}"]})
        cm = confusion_matrix(y, p)
        rn = row_normalise(cm)
        for i, t in enumerate(CLASS_NAMES):
            for j, q in enumerate(CLASS_NAMES):
                conf.append({"model": n, "split": "test", "true": t, "pred": q, "count": int(cm[i, j]),
                             "row_normalised": rn[i, j]})
    return pd.DataFrame(main), pd.DataFrame(per_class), pd.DataFrame(conf), {"E0": first, **aligned}


def val_confusion(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in selected.itertuples():
        d = load_preds(r.run_dir, "val")
        cm = confusion_matrix(d["label"], pred_of(d))
        rn = row_normalise(cm)
        for i, t in enumerate(CLASS_NAMES):
            for j, q in enumerate(CLASS_NAMES):
                rows.append({"model": r.model, "split": "val", "true": t, "pred": q, "count": int(cm[i, j]),
                             "row_normalised": rn[i, j]})
    return pd.DataFrame(rows)


def cost_table(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in selected.itertuples():
        c = json.loads((Path(r.run_dir) / "cost.json").read_text(encoding="utf-8"))
        lat = c.get("latency_batch1") or {}
        rows.append({"model": r.model, "lr": r.lr, "params": c["params"], "peak_mem_mb": c["peak_mem_allocated_mb"],
                     "s_per_epoch": c["s_per_epoch"], "total_train_s": c["total_train_s"],
                     "epochs_run": c["epochs_run"], "best_epoch": c["best_epoch"],
                     "latency_ms_median": lat.get("latency_ms_median")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    # not with_suffix(): the base name contains a dot (lr0.0003)
    fig.savefig(Path(f"{out_base}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(Path(f"{out_base}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_loss(run_dir: Path, out_dir: Path) -> None:
    h = pd.read_csv(run_dir / "history.csv")
    r = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(h["epoch"], h["train_loss"], color="#0072B2", lw=2, label="train loss")
    ax.plot(h["epoch"], h["val_loss"], color="#D55E00", lw=2, ls="--", label="validation loss")
    best = h.loc[h["is_best"].astype(bool), "epoch"].max()
    ax.axvline(best, color="#666666", lw=1, ls=":", label=f"best val TSS≥M (epoch {best})")
    _style(ax, "epoch", "focal loss (γ=2)" if r["model"] in FOCAL else
           "cross-entropy" + (" (class-weighted)" if r["model"] == "E5" else ""))
    ax.set_title(f"{r['model']}, lr={r['lr']:g}, seed {r['seed']}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir / f"loss_{r['model']}_lr{r['lr']:g}_s{r['seed']}")


def fig_val_tss(runs: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    sel = set(selected["run_dir"])
    styles = {lr: ls for lr, ls in zip(sorted(set(runs["lr"]), reverse=True), ["-", "--", ":", "-."])}
    for r in runs.sort_values(["model", "lr"]).itertuples():
        h = pd.read_csv(Path(r.run_dir) / "history.csv")
        ax.plot(h["epoch"], h["val_tss_ge_m"], color=COLORS[r.model], ls=styles[r.lr],
                lw=2.2 if r.run_dir in sel else 1.1, label=f"{r.model} lr={r.lr:g}" + (" ★" if r.run_dir in sel else ""))
    _style(ax, "epoch", "validation TSS (≥M)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("Validation TSS≥M per epoch (★ = selected, thick)", fontsize=10)
    _save(fig, out_dir / "val_tss_per_epoch")


def fig_confusion(model: str, conf_val: pd.DataFrame, conf_test: pd.DataFrame, out_dir: Path, suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (name, c) in zip(axes, (("validation", conf_val), ("test", conf_test))):
        c = c[c["model"] == model]
        cnt = c.pivot(index="true", columns="pred", values="count").loc[list(CLASS_NAMES), list(CLASS_NAMES)]
        rn = c.pivot(index="true", columns="pred", values="row_normalised").loc[list(CLASS_NAMES), list(CLASS_NAMES)]
        ax.imshow(rn.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                v = rn.iloc[i, j]
                ax.text(j, i, f"{cnt.iloc[i, j]}\n" + ("" if not np.isfinite(v) else f"({100 * v:.0f}%)"),
                        ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "black")
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{model} – {name}" + (f" ({suffix})" if suffix else ""), fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / (f"confusion_{model}" + (f"_{suffix}" if suffix else "")))


# ================================================================ post-hoc: tuned thresholds, AUC (Change Log 15)
VARIANTS = ["argmax", "tuned_tss", "tuned_hss"]
VARIANT_LABEL = {"argmax": "argmax (pre-registered)", "tuned_tss": "tuned τ (val TSS≥M)", "tuned_hss": "tuned τ (val HSS≥M)"}
EVAL_METRICS = ["tss_ge_m", "hss_ge_m", "precision_ge_m", "far", "recall_M", "recall_X", "tss_ovr_<M", "tss_ovr_M",
                "tss_ovr_X", "macro_tss", "roc_auc", "pr_auc"]
MAIN_EVAL = ["tss_ge_m", "hss_ge_m", "precision_ge_m", "far", "recall_M", "recall_X", "macro_tss", "roc_auc", "pr_auc"]


def run_thresholds(runs: pd.DataFrame) -> pd.DataFrame:
    """Thresholds of every run in ``runs`` (tuned on that run's preds_val.csv only; no test file is opened)."""
    rows = []
    for r in runs.itertuples():
        rows.append({"model": r.model, "lr": r.lr, "seed": r.seed, **tune_run(r.run_dir)})
    return pd.DataFrame(rows)


def _thr_of(thr: pd.DataFrame, model: str, lr: float, seed: int) -> dict:
    t = thr[(thr["model"] == model) & np.isclose(thr["lr"], lr) & (thr["seed"] == seed)]
    if len(t) != 1:
        raise ValueError(f"thresholds for {model} lr={lr:g} seed={seed}: {len(t)} rows (expected 1)")
    return t.iloc[0].to_dict()


def predict_variant(df: pd.DataFrame, variant: str, thr: dict | None) -> np.ndarray:
    """Predicted classes of one prediction file under argmax / tuned_tss / tuned_hss thresholds."""
    p = df[PROB_COLS].to_numpy()
    if variant == "argmax":
        return probs_to_pred(p)
    if thr is None:
        raise ValueError(f"variant {variant!r} needs thresholds")
    if variant == "tuned_tss":
        return two_stage_predict(p, thr["tau1"], thr["tau2"])
    if variant == "tuned_hss":
        return two_stage_predict(p, thr["tau1_hss"], thr["tau2_hss"])
    raise ValueError(f"unknown variant {variant!r}")


def metric_row(df: pd.DataFrame, pred: np.ndarray, probs: np.ndarray | None = None) -> dict[str, float]:
    """Metrics of ``pred`` + false-alarm ratio FP/(TP+FP), macro TSS and the threshold-free AUCs of ``probs``
    (default: the probability columns of ``df``)."""
    y = df["label"].to_numpy()
    m = summarise(y, pred)
    out = {k: m[k] for k in EVAL_METRICS if k in m}
    out["far"] = 1.0 - m["precision_ge_m"] if np.isfinite(m["precision_ge_m"]) else float("nan")
    out["macro_tss"] = macro_tss(confusion_matrix(y, pred))
    out.update(auc_scores(y, df[PROB_COLS].to_numpy() if probs is None else probs))
    return out


def eval_long(selected: pd.DataFrame, thr: pd.DataFrame, splits=("val", "test")) -> pd.DataFrame:
    """One row per (model, variant, split) for the selected runs of ONE seed, plus the E0 persistence rows.
    ``thr`` = :func:`run_thresholds` output (looked up by model, lr, seed)."""
    rows = []
    for sp in splits:
        d0 = load_preds(selected.iloc[0]["run_dir"], sp)
        pl = d0["persistence_label"].to_numpy()
        rows.append({"model": "E0", "lr": np.nan, "seed": int(selected.iloc[0]["seed"]), "variant": "persistence",
                     "split": sp, "tau1": np.nan, "tau2": np.nan, **metric_row(d0, pl, np.eye(3)[pl])})
    for r in selected.itertuples():
        t = _thr_of(thr, r.model, r.lr, r.seed)
        for sp in splits:
            d = load_preds(r.run_dir, sp)
            for v in VARIANTS:
                tau1, tau2 = {"argmax": (np.nan, np.nan), "tuned_tss": (t["tau1"], t["tau2"]),
                              "tuned_hss": (t["tau1_hss"], t["tau2_hss"])}[v]
                rows.append({"model": r.model, "lr": r.lr, "seed": r.seed, "variant": v, "split": sp,
                             "tau1": tau1, "tau2": tau2, **metric_row(d, predict_variant(d, v, t))})
    return pd.DataFrame(rows)


def eval_seed_summary(long: pd.DataFrame) -> pd.DataFrame:
    """mean and sd (ddof=1) over seeds per (model, variant, split) of every metric in ``long`` (all seeds stacked)."""
    g = long.groupby(["model", "variant", "split"], sort=False)
    agg = g[EVAL_METRICS].agg(["mean", "std"])
    agg.columns = [f"{k}_{'sd' if f == 'std' else f}" for k, f in agg.columns]
    agg.insert(0, "n_seeds", g["seed"].nunique())
    return agg.reset_index()


def eval_md_table(long: pd.DataFrame, split: str, variants: list[str], metrics: list[str] = MAIN_EVAL) -> pd.DataFrame:
    """Compact table for the markdown summary; when ``long`` stacks several seeds, cells are 'mean ± sd'."""
    d = long[(long["split"] == split) & long["variant"].isin(variants)]
    multi = d["seed"].nunique() > 1
    rows = []
    for (model, variant), g in d.groupby(["model", "variant"], sort=False):
        row = {"model": model, "variant": variant, "n_seeds": g["seed"].nunique()}
        if not multi:
            row["tau1"], row["tau2"] = g["tau1"].iloc[0], g["tau2"].iloc[0]
        for k in metrics:
            if multi:
                m, s = g[k].mean(), g[k].std(ddof=1)
                row[k] = f"{m:.3f} ± {s:.3f}" if np.isfinite(m) and np.isfinite(s) else "n/a"
            else:
                row[k] = g[k].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_variants(selected: pd.DataFrame, thr: pd.DataFrame, variants: list[str], splits=("val", "test")) -> pd.DataFrame:
    rows = []
    for r in selected.itertuples():
        t = _thr_of(thr, r.model, r.lr, r.seed)
        for sp in splits:
            d = load_preds(r.run_dir, sp)
            for v in variants:
                cm = confusion_matrix(d["label"], predict_variant(d, v, t))
                rn = row_normalise(cm)
                for i, a in enumerate(CLASS_NAMES):
                    for j, b in enumerate(CLASS_NAMES):
                        rows.append({"model": r.model, "variant": v, "split": sp, "true": a, "pred": b,
                                     "count": int(cm[i, j]), "row_normalised": rn[i, j]})
    return pd.DataFrame(rows)


def tuned_test_preds(sel_by_seed: dict[int, pd.DataFrame], thr: pd.DataFrame, variant: str):
    """(reference aligned test frame, {seed: {model: predicted classes}}, {seed: {model: >=M score}}) for the bootstrap.
    Thresholds come from validation only (``thr``). Raises if the seeds' test samples differ."""
    ref, preds, scores = None, {}, {}
    for s, sel in sel_by_seed.items():
        aligned = align_on_samples({r.model: load_preds(r.run_dir, "test") for r in sel.itertuples()})
        first = next(iter(aligned.values()))
        if ref is None:
            ref = first
        elif not (first[["harpnum", "t"]].equals(ref[["harpnum", "t"]]) and first["label"].equals(ref["label"])):
            raise ValueError(f"Test samples of seed {s} differ from the first seed")
        pl = first["persistence_label"].to_numpy()
        preds[s], scores[s] = {"E0": pl}, {"E0": (pl >= 1).astype(float)}
        for r in sel.itertuples():
            preds[s][r.model] = predict_variant(aligned[r.model], variant, _thr_of(thr, r.model, r.lr, s))
            scores[s][r.model] = score_ge_m(aligned[r.model][PROB_COLS].to_numpy())
    return ref, preds, scores


# ---------------------------------------------------------------- figures: ROC / PR / thresholds
def fig_roc_pr(selected: pd.DataFrame, aligned_test: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """ROC and precision-recall curves of P(>=M), all selected models on the test set; E0 as a single operating point."""
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
    first = aligned_test["E0"]
    y = (first["label"].to_numpy() >= 1).astype(int)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for m in selected["model"]:
        s = score_ge_m(aligned_test[m][PROB_COLS].to_numpy())
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        ls = "--" if m in FOCAL else "-"
        a1.plot(fpr, tpr, color=COLORS[m], ls=ls, lw=1.6, label=f"{m} (AUC {roc_auc_score(y, s):.3f})")
        a2.plot(rc, pr, color=COLORS[m], ls=ls, lw=1.6, label=m)
    pl = (first["persistence_label"].to_numpy() >= 1).astype(int)
    tp, fp = int(((pl == 1) & (y == 1)).sum()), int(((pl == 1) & (y == 0)).sum())
    a1.scatter([fp / max((y == 0).sum(), 1)], [tp / max(y.sum(), 1)], color=COLORS["E0"], marker="s", zorder=5,
               label="E0 persistence")
    a2.scatter([tp / max(y.sum(), 1)], [tp / max(tp + fp, 1)], color=COLORS["E0"], marker="s", zorder=5, label="E0")
    a1.plot([0, 1], [0, 1], color="#aaaaaa", lw=0.8, ls=":")
    a2.axhline(y.mean(), color="#aaaaaa", lw=0.8, ls=":")
    _style(a1, "false-positive rate", "true-positive rate")
    _style(a2, "recall (≥M)", "precision (≥M)")
    a1.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    a2.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    a1.set_title("ROC, test (≥M)", fontsize=10)
    a2.set_title("Precision–recall, test (≥M); dotted = prevalence", fontsize=10)
    a1.legend(frameon=False, fontsize=7, loc="lower right")
    a2.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / "roc_pr_test")


def fig_threshold_curves(selected: pd.DataFrame, thr: pd.DataFrame, out_dir: Path) -> None:
    """Validation TSS>=M and HSS>=M versus tau1, and HSS_X versus tau2 (at the TSS-optimal tau1); stars = chosen values."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for r in selected.itertuples():
        d = load_preds(r.run_dir, "val")
        c = threshold_curves(d["label"].to_numpy(), d[PROB_COLS].to_numpy())
        t = _thr_of(thr, r.model, r.lr, r.seed)
        ls = "--" if r.model in FOCAL else "-"
        col = COLORS[r.model]
        for ax, key, tau in zip(axes, ("tss_ge_m", "hss_ge_m", "hss_X"), (t["tau1"], t["tau1_hss"], t["tau2"])):
            ax.plot(c["grid"], c[key], color=col, ls=ls, lw=1.5, label=r.model)
            i = int(np.argmin(np.abs(c["grid"] - tau)))
            ax.scatter([tau], [c[key][i]], color=col, marker="*", s=70, zorder=5)
    for ax, xl, yl in zip(axes, ("τ1 (on P(≥M))", "τ1 (on P(≥M))", "τ2 (on P(X | ≥M), at the TSS-optimal τ1)"),
                          ("validation TSS≥M", "validation HSS≥M", "validation HSS_X")):
        _style(ax, xl, yl)
        ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Validation threshold curves (★ = chosen; selected runs, validation data only)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "threshold_curves_val")
