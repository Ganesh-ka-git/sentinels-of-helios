"""Grad-CAM, comparison-grid and attention figures (300 dpi PNG + PDF via evaluate.report._save)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..evaluate.report import COLORS, _save  # noqa: E402
from ..preprocess.frames import read_br  # noqa: E402

BR_LIM = 1000.0          # display range of the magnetograms [G]
OVERLAY_ALPHA = 0.45
CAM_MIN_SHOWN = 0.05     # CAM values below this are not drawn in the overlays
CAM_CMAP = "jet"


def read_raw(path: Path) -> np.ndarray:
    """Full-resolution Br [G] (NaN kept). The url_quick / as-is files carry no HARPNUM / T_REC / NOAA keywords; those come
    from a JSOC keyword query saved by the download stage (``case_keywords.csv``)."""
    return read_br(path)


def parse_trec(value: str) -> pd.Timestamp:
    """'2024.05.12_12:00:00.000_TAI' -> Timestamp (minute resolution is enough for 12-min records)."""
    core = value.split("_TAI")[0]
    fmt = "%Y.%m.%d_%H:%M:%S.%f" if "." in core.split("_", 1)[1] else "%Y.%m.%d_%H:%M:%S"
    return pd.to_datetime(core, format=fmt).floor("min")


def noaa_label(info: dict) -> str:
    """'13682' for one region; 'a,b,c' as listed in NOAA_ARS otherwise ('none' if empty / 0)."""
    def clean(v) -> str:
        v = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
        return "" if v in ("nan", "0", "None") else v
    return clean(info.get("noaa_ars")) or clean(info.get("noaa_ar")) or "none"


def case_title(case: pd.Series, info: dict) -> str:
    return (f"HARP {int(case['harpnum'])} · NOAA {noaa_label(info)} · {pd.Timestamp(case['t']):%Y-%m-%d %H:%M} "
            f"(T_REC, TAI)\ntrue class: {case['true_class']}")


def _show_magnetogram(ax, raw: np.ndarray):
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(raw), cmap=cmap, vmin=-BR_LIM, vmax=BR_LIM, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _overlay(ax, raw: np.ndarray, cam: np.ndarray):
    """Magnetogram + CAM (alpha 0.45); CAM values below CAM_MIN_SHOWN are transparent so the field stays readable."""
    _show_magnetogram(ax, raw)
    if float(cam.max()) <= 0:
        ax.text(0.5, 0.5, "no positive Grad-CAM inside the patch\n(no evidence, or only in the padding)", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#b00000", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    return ax.imshow(np.ma.masked_less(cam, CAM_MIN_SHOWN), cmap=CAM_CMAP, vmin=0, vmax=1, alpha=OVERLAY_ALPHA,
                     origin="lower", interpolation="bilinear")


def decision_text(p: float, tau1: float) -> str:
    return f"P(≥M)={p:.2f} → " + ("predicted ≥M" if p >= tau1 else "predicted <M") + f" (τ1={tau1:.2f})"


def fig_case(case: pd.Series, info: dict, raw: np.ndarray, cam: np.ndarray, model: str, p: float, tau1: float,
             out_dir: Path) -> Path:
    """(a) LEFT magnetogram, RIGHT magnetogram + Grad-CAM overlay (horizontal colorbars below, titles wrapped)."""
    h, w = raw.shape
    width = 9.0 if w / h > 2.0 else 7.4                      # very wide active-region patches need more width
    fig, axes = plt.subplots(1, 2, figsize=(width, width / 2 * h / w + 2.0), constrained_layout=True)
    im = _show_magnetogram(axes[0], raw)
    axes[0].set_title(f"HARP {int(case['harpnum'])} · NOAA {noaa_label(info)}\n{pd.Timestamp(case['t']):%Y-%m-%d %H:%M} "
                      f"(T_REC, TAI) · true class: {case['true_class']}", fontsize=8)
    fig.colorbar(im, ax=axes[0], orientation="horizontal", fraction=0.07, pad=0.04).set_label("B$_r$ [G]", fontsize=8)
    ov = _overlay(axes[1], raw, cam)
    axes[1].set_title(f"{model} Grad-CAM\n{decision_text(p, tau1)}", fontsize=8)
    fig.colorbar(ov, ax=axes[1], orientation="horizontal", fraction=0.07, pad=0.04).set_label("Grad-CAM (normalised)",
                                                                                              fontsize=8)
    base = out_dir / f"gradcam_{case['case']}_{model}"
    _save(fig, base)
    return base


def fig_grid(case: pd.Series, info: dict, raw: np.ndarray, rows: list[dict], out_dir: Path) -> Path:
    """(b) rows = models; columns = CAM alone | CAM overlaid on the magnetogram. ``rows``: model, cam, p, tau1."""
    h, w = raw.shape
    n = len(rows)
    cell_h = 2.4 * max(0.8, min(1.3, h / w))
    fig, axes = plt.subplots(n, 2, figsize=(6.2, cell_h * n + 0.8), constrained_layout=True)
    for i, r in enumerate(rows):
        a0, a1 = axes[i]
        a0.imshow(r["cam"], cmap=CAM_CMAP, vmin=0, vmax=1, origin="lower", interpolation="bilinear")
        if float(r["cam"].max()) <= 0:
            a0.text(0.5, 0.5, "CAM = 0 in patch", transform=a0.transAxes, ha="center", va="center", fontsize=8, color="white")
        _overlay(a1, raw, r["cam"])
        for a in (a0, a1):
            a.set_xticks([])
            a.set_yticks([])
        a0.set_ylabel(r["model"], rotation=0, labelpad=16, fontsize=10, va="center", color=COLORS.get(r["model"], "k"))
        a1.set_title(decision_text(r["p"], r["tau1"]), fontsize=7, pad=2)
    axes[0][0].set_title("Grad-CAM (normalised)", fontsize=8, pad=2)
    fig.suptitle(case_title(case, info).replace("\n", " · "), fontsize=8)
    base = out_dir / f"grid_{case['case']}"
    _save(fig, base)
    return base


def fig_attention(case: pd.Series, attn: dict[str, dict[str, np.ndarray]], out_dir: Path) -> Path:
    """(c) attention weights of E4, E5, E6; x = hours before t; coarse (48 min) and fine (24 min) branches."""
    steps = {"coarse": 0.8, "fine": 0.4}      # hours between frames
    titles = {"coarse": "coarse branch (12 h @ 48 min)", "fine": "fine branch (6 h @ 24 min)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    for ax, br in zip(axes, ("coarse", "fine")):
        hours = (15 - np.arange(16)) * steps[br]
        for m, a in attn.items():
            ax.plot(hours, a[br], marker="o", ms=3, lw=1.6, color=COLORS.get(m), label=m)
        ax.axhline(1 / 16, color="#999999", lw=0.8, ls=":", label="uniform")
        ax.invert_xaxis()
        ax.set_xlabel("hours before t")
        ax.set_ylabel("attention weight")
        ax.set_title(titles[br], fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Attention weights, case {case['case']} (HARP {int(case['harpnum'])}, {pd.Timestamp(case['t']):%Y-%m-%d %H:%M}, "
                 f"true class {case['true_class']})", fontsize=8)
    base = out_dir / f"attention_{case['case']}"
    _save(fig, base)
    return base
