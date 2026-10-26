"""Module 8 — Grad-CAM cases, full-resolution frames and figures for E1-E8 (decisions.md §11, Change Log 13).

Stages (run in this order; each is resumable):
  select    Test cases from E8 seed 0 at its validation-tuned tau1 (2 TP incl. one X, 2 missed, 2 false alarms, 1 TN)
            -> outputs/explain/cases.csv. Reads preds_val.csv (for tau1) and preds_test.csv of E8 only.
  download  Re-downloads exactly those frames at full resolution from JSOC (url_quick / as-is, ONE export at a time)
            -> data/figure_fits/ (kept). Needs jsoc.email in config/config.yaml.
  figures   Grad-CAM (CPU, fp32) of the seed-0 selected runs of E1-E8 on the cases; per-case figures, comparison grids,
            attention plots (E4, E5, E6) and figure_index.md -> outputs/figures/gradcam/, outputs/explain/gradcam_results.csv.

Run from the project root (PowerShell):
    conda run -n flare --no-capture-output python scripts/08_gradcam.py select
    conda run -n flare --no-capture-output python scripts/08_gradcam.py download
    conda run -n flare --no-capture-output python scripts/08_gradcam.py figures
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flarecast.config import load_config, require_email  # noqa: E402
from flarecast.data.jsoc import build_record_sets, download_file, export_urls, local_filename  # noqa: E402
from flarecast.dataset import load_samples  # noqa: E402
from flarecast.evaluate import report as R  # noqa: E402
from flarecast.evaluate.thresholds import tune_run  # noqa: E402
from flarecast.explain import figures as FG  # noqa: E402
from flarecast.explain.cases import CASE_SPEC, select_cases  # noqa: E402
from flarecast.explain.gradcam import cam_to_patch, explain_cases, load_model  # noqa: E402
from flarecast.preprocess.frames import preprocess_br  # noqa: E402

log = logging.getLogger("gradcam")
MODELS = [f"E{i}" for i in range(1, 9)]
ATTN_MODELS = ["E4", "E5", "E6"]
REF_MODEL = "E8"
CASE_FIG_MODEL = REF_MODEL          # model shown in the per-case figure (a); all models are in the grids (b)
MAX_STORED_DIFF = 2e-2              # recomputed vs stored 128 px frame (float16 storage + area resize)
MAX_P_DIFF = 0.05                   # fp32-CPU vs saved (fp16 AMP, GPU) P(>=M)


def paths(cfg: dict) -> dict[str, Path]:
    out = cfg["paths"]["outputs_dir"]
    return {"runs": out / "runs", "explain": out / "explain", "fig": out / "figures" / "gradcam",
            "fits": cfg["paths"]["data_dir"] / "figure_fits"}


def selected_runs(runs_dir: Path) -> pd.DataFrame:
    sel = R.select_best(R.discover_runs(runs_dir, None, 0)).set_index("model")
    missing = [m for m in MODELS if m not in sel.index]
    if missing:
        raise SystemExit(f"No finished seed-0 run for {missing}")
    return sel.loc[MODELS]


def stage_select(cfg: dict) -> int:
    p = paths(cfg)
    sel = selected_runs(p["runs"])
    ref = sel.loc[REF_MODEL]
    tau1 = float(tune_run(ref["run_dir"])["tau1"])            # tuned on that run's validation predictions only
    log.info("%s seed 0 lr=%g: validation-tuned tau1 = %.2f", REF_MODEL, ref["lr"], tau1)
    cases = select_cases(R.load_preds(ref["run_dir"], "test"), tau1)
    p["explain"].mkdir(parents=True, exist_ok=True)
    cases.to_csv(p["explain"] / "cases.csv", index=False)
    print(R.md_table(cases))
    print(f"\nWrote {p['explain'] / 'cases.csv'}  (spec: {CASE_SPEC})")
    return 0


def load_cases(p: dict) -> pd.DataFrame:
    f = p["explain"] / "cases.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found: run the 'select' stage first")
    return pd.read_csv(f, parse_dates=["t"], dtype={"case": str})


def fits_path(p: dict, cfg: dict, c) -> Path:
    rec = build_record_sets(pd.DataFrame({"harpnum": [c.harpnum], "time": [c.t]}), cfg["jsoc"]["series"],
                            cfg["jsoc"]["segment"])[0]
    return p["fits"] / local_filename(rec, cfg["jsoc"]["segment"]), rec


def stage_download(cfg: dict) -> int:
    import drms
    import requests

    p = paths(cfg)
    cases = load_cases(p)
    email = require_email(cfg)
    p["fits"].mkdir(parents=True, exist_ok=True)
    todo = []
    for c in cases.itertuples():
        dest, rec = fits_path(p, cfg, c)
        if dest.exists():
            log.info("%s already present: skipping", dest.name)
        else:
            todo.append((c, dest, rec))
    client = drms.Client(email=email)
    kw_file = p["explain"] / "case_keywords.csv"
    if not kw_file.exists():                                   # keyword query only (no export): HARPNUM, T_REC, NOAA_ARS
        rows = []
        for c in cases.itertuples():
            rs = f"{cfg['jsoc']['series']}[{c.harpnum}][{c.t:%Y.%m.%d_%H:%M:%S}_TAI]"
            kw = client.query(rs, key="HARPNUM, T_REC, NOAA_ARS, NOAA_AR", convert_numeric=False)
            if len(kw) != 1:
                raise RuntimeError(f"{rs}: expected 1 keyword record, got {len(kw)}")
            rows.append({"case": c.case, "harpnum": str(kw["HARPNUM"].iloc[0]).strip(), "t_rec": str(kw["T_REC"].iloc[0]),
                         "noaa_ars": str(kw["NOAA_ARS"].iloc[0]), "noaa_ar": str(kw["NOAA_AR"].iloc[0])})
        pd.DataFrame(rows).to_csv(kw_file, index=False)
        log.info("wrote %s", kw_file)
    if not todo:
        print(f"All {len(cases)} frames already in {p['fits']}")
        return 0
    dl, failed = cfg["download"], []
    session = requests.Session()
    for c, dest, rec in todo:                                  # one export at a time (JSOC: one pending export per user)
        urls = export_urls(client, [rec], email, cfg["jsoc"]["segment"])
        if len(urls) != 1:
            raise RuntimeError(f"{rec}: expected 1 file, export returned {len(urls)}")
        res = download_file(urls["url"].iloc[0], dest, session, dl["timeout_s"], dl["retries"])
        log.info("%s: %s (%d bytes)", dest.name, res["status"], res["bytes"])
        if res["status"] == "failed":
            failed.append((dest.name, res["error"]))
    if failed:
        print(f"!!!!! FAILED downloads (not retried): {failed}")
        return 1
    print(f"Downloaded {len(todo)} frame(s) to {p['fits']}")
    return 0


def stage_figures(cfg: dict) -> int:
    p = paths(cfg)
    cases = load_cases(p)
    sel = selected_runs(p["runs"])
    samples = load_samples(cfg["paths"]["samples_csv"])
    models = {m: load_model(m, Path(sel.loc[m, "run_dir"]) / "best.pt") for m in MODELS}
    tau1 = {m: float(tune_run(sel.loc[m, "run_dir"])["tau1"]) for m in MODELS}      # per model, validation only
    res = explain_cases(models, cases, samples, cfg["paths"]["frames_dir"], cfg["loader"].get("use_cache", True),
                        cfg["paths"].get("cache_dir"))
    p["fig"].mkdir(parents=True, exist_ok=True)
    kw_file = p["explain"] / "case_keywords.csv"
    if not kw_file.exists():
        raise SystemExit(f"{kw_file} not found: run the 'download' stage first")
    keywords = pd.read_csv(kw_file, dtype=str).set_index("case")
    saved ={m: R.load_preds(sel.loc[m, "run_dir"], "test") for m in MODELS}
    rows, index = [], []
    for c in cases.itertuples():
        cs = cases.set_index("case").loc[c.case].copy()
        cs["case"] = c.case
        raw = FG.read_raw(fits_path(p, cfg, c)[0])
        info = keywords.loc[c.case].to_dict()
        if int(info["harpnum"]) != c.harpnum or FG.parse_trec(str(info["t_rec"])) != c.t:
            raise ValueError(f"{c.case}: JSOC keywords {info} do not match the case")
        stored = _stored_frame(cfg, c)
        diff = float(np.abs(preprocess_br(raw)[0].astype(np.float32) - stored).max())
        log.info("%s: raw patch %s, max |recomputed - stored 128px frame| = %.4f", c.case, raw.shape, diff)
        if diff > MAX_STORED_DIFF:
            raise ValueError(f"{c.case}: re-downloaded frame differs from the stored training frame (max diff {diff:.4f})")
        grid_rows, attn = [], {}
        for m in MODELS:
            r = res[(m, c.case)]
            s = saved[m]
            hit = s[(s["harpnum"] == c.harpnum) & (pd.to_datetime(s["t"]) == c.t)]
            p_saved = float(hit["p1"].iloc[0] + hit["p2"].iloc[0])
            if abs(p_saved - r["p_ge_m"]) > MAX_P_DIFF:
                raise ValueError(f"{c.case} {m}: P(>=M) {r['p_ge_m']:.3f} differs from the saved {p_saved:.3f}")
            cam = cam_to_patch(r["cam"], raw.shape)
            grid_rows.append({"model": m, "cam": cam, "p": r["p_ge_m"], "tau1": tau1[m]})
            if m in ATTN_MODELS:
                attn[m] = r["attention"]
            rows.append({"case": c.case, "model": m, "p_ge_m": r["p_ge_m"], "p_ge_m_saved": p_saved, "tau1_model": tau1[m],
                         "decision": ">=M" if r["p_ge_m"] >= tau1[m] else "<M", "target_log_p": r["target"],
                         "cam_max_raw": float(r["cam"].max()), "cam_patch_max": float(cam.max()), "raw_h": raw.shape[0], "raw_w": raw.shape[1],
                         "noaa_ars": FG.noaa_label(info)})
        g = next(x for x in grid_rows if x["model"] == CASE_FIG_MODEL)
        fa = FG.fig_case(cs, info, raw, g["cam"], CASE_FIG_MODEL, g["p"], g["tau1"], p["fig"])
        fb = FG.fig_grid(cs, info, raw, grid_rows, p["fig"])
        fc = FG.fig_attention(cs, attn, p["fig"])
        index += [("a", c, fa), ("b", c, fb), ("c", c, fc)]
    pd.DataFrame(rows).to_csv(p["explain"] / "gradcam_results.csv", index=False)
    write_index(p, index, tau1)
    for _, _, base in index:
        print(f"{base}.png\n{base}.pdf")
    print(f"\nIndex: {p['fig'] / 'figure_index.md'}\nResults: {p['explain'] / 'gradcam_results.csv'}")
    return 0


def _stored_frame(cfg: dict, c) -> np.ndarray:
    """The 128 x 128 frame at t as the models saw it (read from the per-frame files, not the cache)."""
    f = cfg["paths"]["frames_dir"] / str(c.harpnum) / f"{c.t:%Y%m%d_%H%M}.npy"
    return np.load(f).astype(np.float32)


def write_index(p: dict, index: list, tau1: dict[str, float]) -> None:
    rel = lambda base: f"{Path(base).name}.png / .pdf"  # noqa: E731
    cap = {
        "a": ("Left: full-resolution HMI SHARP Br magnetogram (grey, ±1000 G) of {desc}. Right: the same magnetogram with the "
              f"{CASE_FIG_MODEL} Grad-CAM overlay (alpha 0.45; target log P(≥M), last convolutional block, frame at t; CAM "
              "normalised to its maximum; values below 0.05 are transparent). The title gives the model's P(≥M) and its decision "
              "at the validation-tuned τ1. 'No positive Grad-CAM inside the patch' means the CAM is zero within the patch after removing the padding (no "
              "positive evidence for ≥M, or all of it lies in the padded margin)."),
        "b": ("Grad-CAM of all models E1–E8 for {desc}. Left column: CAM alone; right column: overlay on the Br magnetogram. "
              "Same target (log P(≥M)) and layer for every model; each CAM is normalised to its own maximum, so intensities "
              "are not comparable across models; overlay values below 0.05 are transparent, and 'CAM = 0 in patch' means no positive CAM inside the patch (none at all, or only in the padding). "
              "Titles give each model's P(≥M) and decision at its own validation-tuned τ1."),
        "c": ("Additive-attention weights of the attention models E4, E5 and E6 for {desc}, for the coarse (12 h @ 48 min) "
              "and fine (6 h @ 24 min) branches; x = hours before t. The dotted line marks uniform attention (1/16)."),
    }
    lines = ["# Grad-CAM figure index (draft captions)", "",
             f"Cases are test samples chosen with {REF_MODEL} seed 0 at its validation-tuned τ1 = {tau1[REF_MODEL]:.2f} "
             "(2 true positives incl. one X flare, 2 missed events, 2 false alarms, 1 true negative; distinct HARPs; "
             "rules in `flarecast.explain.cases`). All models use the seed-0 runs with the lr selected on validation TSS≥M. "
             "T_REC is TAI. Grad-CAM was computed in fp32 on the CPU, so P(≥M) can differ slightly from the saved "
             "(fp16, GPU) predictions; the largest difference is in `outputs/explain/gradcam_results.csv`.", ""]
    for kind, c, base in index:
        desc = f"case {c.case} (HARP {c.harpnum}, {c.t:%Y-%m-%d %H:%M}, true class {c.true_class}; category {c.category})"
        lines += [f"## {Path(base).name} ({kind})", f"Files: {rel(base)}", "", cap[kind].format(desc=desc), ""]
    (p["fig"] / "figure_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "download", "figures"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    return {"select": stage_select, "download": stage_download, "figures": stage_figures}[a.stage](cfg)


if __name__ == "__main__":
    sys.exit(main())
