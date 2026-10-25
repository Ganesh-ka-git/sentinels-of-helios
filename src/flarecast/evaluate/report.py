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

log = logging.getLogger(__name__)

MODELS = ["E1", "E2", "E3", "E4", "E5"]
COLORS = {"E0": "#666666", "E1": "#0072B2", "E2": "#E69F00", "E3": "#009E73", "E4": "#CC79A7", "E5": "#D55E00"}
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
    _style(ax, "epoch", "cross-entropy" + (" (class-weighted)" if r["model"] == "E5" else ""))
    ax.set_title(f"{r['model']}, lr={r['lr']:g}, seed {r['seed']}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_dir / f"loss_{r['model']}_lr{r['lr']:g}_s{r['seed']}")


def fig_val_tss(runs: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    sel = set(selected["run_dir"])
    styles = {lr: ls for lr, ls in zip(sorted(runs["lr"], reverse=True), ["-", "--", ":", "-."])}
    for r in runs.sort_values(["model", "lr"]).itertuples():
        h = pd.read_csv(Path(r.run_dir) / "history.csv")
        ax.plot(h["epoch"], h["val_tss_ge_m"], color=COLORS[r.model], ls=styles[r.lr],
                lw=2.2 if r.run_dir in sel else 1.1, label=f"{r.model} lr={r.lr:g}" + (" ★" if r.run_dir in sel else ""))
    _style(ax, "epoch", "validation TSS (≥M)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("Validation TSS≥M per epoch (★ = selected, thick)", fontsize=10)
    _save(fig, out_dir / "val_tss_per_epoch")


def fig_confusion(model: str, conf_val: pd.DataFrame, conf_test: pd.DataFrame, out_dir: Path) -> None:
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
        ax.set_title(f"{model} – {name}", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / f"confusion_{model}")
