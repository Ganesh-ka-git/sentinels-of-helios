"""Module 7 — evaluation and reporting from finished runs (outputs/runs/*/DONE). No retraining: everything is computed
from the saved preds_<split>.csv files.

Selection per model = the lr with the best VALIDATION TSS>=M under argmax (test is never used for selection).
Pre-registered results use argmax. Post-hoc (decisions.md Change Log 15): two-stage thresholds tuned on VALIDATION only
and applied once to test, an HSS-optimal sensitivity variant, and threshold-free ROC-AUC / PR-AUC for >=M. Models E6-E8
(focal loss, Change Log 16) are evaluated the same way when their runs exist.

Writes CSV tables and results_summary.md to outputs/results/ and 300-dpi PNG + PDF figures to outputs/figures/ (a
sub-folder <tag>/ when --tag is given, e.g. smoke).

Run from the project root (PowerShell):
    conda run -n flare --no-capture-output python scripts/07_evaluate.py
    conda run -n flare --no-capture-output python scripts/07_evaluate.py --seeds 0 1 2
    conda run -n flare --no-capture-output python scripts/07_evaluate.py --tag smoke
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.evaluate import report as R  # noqa: E402
from flarecast.evaluate.bootstrap import (multi_seed_bootstrap, multi_seed_score_bootstrap,  # noqa: E402
                                          paired_bootstrap)
from flarecast.evaluate.metrics import probs_to_pred  # noqa: E402

log = logging.getLogger("evaluate")
N_BOOT, BOOT_SEED = 2000, 0
# (first, second): difference = second - first. Primary chain (H1-H3), then focal comparisons (H4, H5).
CHAIN = [("E1", "E2"), ("E2", "E3"), ("E3", "E4"), ("E4", "E5")]
FOCAL_PAIRS = [("E1", "E8"), ("E2", "E7"), ("E4", "E6"), ("E5", "E6")]      # E8vsE1, E7vsE2, E6vsE4 (H4), E6vsE5 (H5)
H5_METRICS = ["hss_ge_m", "precision_ge_m", "recall_ge_m"]


def comparisons(models: list[str]) -> list[tuple[str, str]]:
    """Available pairs of CHAIN + FOCAL_PAIRS, then E0 vs every model."""
    have = set(models)
    pairs = [(a, b) for a, b in CHAIN + FOCAL_PAIRS if a in have and b in have]
    skipped = [f"{a}vs{b}" for a, b in CHAIN + FOCAL_PAIRS if (a, b) not in pairs]
    if skipped:
        log.warning("Comparisons skipped (model missing): %s", skipped)
    return pairs + [("E0", m) for m in models]


def run_bootstrap(aligned: dict[str, pd.DataFrame], models: list[str]) -> pd.DataFrame:
    """Pre-registered single-seed argmax bootstrap (TSS>=M for every comparison; recall M/X for E4 vs E5)."""
    first = aligned["E0"]
    harp, y = first["harpnum"].to_numpy(), first["label"].to_numpy()
    preds = {"E0": first["persistence_label"].to_numpy(),
             **{m: probs_to_pred(aligned[m][R.PROB_COLS].to_numpy()) for m in models}}
    comps = comparisons(models)
    rows = paired_bootstrap(harp, y, preds, comps, ["tss_ge_m"], N_BOOT, BOOT_SEED)
    if ("E4", "E5") in comps:
        rows += paired_bootstrap(harp, y, preds, [("E4", "E5")], ["recall_M", "recall_X"], N_BOOT, BOOT_SEED)
    return pd.DataFrame(rows)


def run_pooled(sel_by_seed, thr: pd.DataFrame, models: list[str], variant: str, metrics: list[str],
               pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Pooled multi-seed paired HARP bootstrap (decisions.md §3) of class-prediction metrics under ``variant``."""
    ref, preds, _ = R.tuned_test_preds(sel_by_seed, thr, variant)
    pairs = comparisons(models) if pairs is None else pairs
    rows = []
    for m in metrics:
        rows += multi_seed_bootstrap(ref["harpnum"].to_numpy(), ref["label"].to_numpy(), preds, pairs, N_BOOT,
                                     BOOT_SEED, metric=m)
    out = pd.DataFrame(rows)
    out.insert(1, "variant", variant)
    return out


def run_auc(sel_by_seed, thr: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    ref, _, scores = R.tuned_test_preds(sel_by_seed, thr, "argmax")
    rows = []
    for stat in ("roc_auc", "pr_auc"):
        rows += multi_seed_score_bootstrap(ref["harpnum"].to_numpy(), ref["label"].to_numpy(), scores,
                                           comparisons(models), stat, N_BOOT, BOOT_SEED)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="evaluate only runs trained with this tag (default: untagged runs)")
    ap.add_argument("--seed", type=int, default=0, help="seed used for selection, tables, Grad-CAM and figures")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="e.g. 0 1 2: add multi-seed tables (mean ± sd) and pooled/sign-consistent bootstrap verdicts. "
                         "The lr per model is chosen on --seed by validation TSS>=M and reused for the other seeds")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if a.seeds is not None and a.seed not in a.seeds:
        raise SystemExit(f"--seeds {a.seeds} must include the selection seed --seed {a.seed}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    out = cfg["paths"]["outputs_dir"]
    res_dir = out / "results" / (a.tag or "")
    fig_dir = out / "figures" / (a.tag or "")
    res_dir.mkdir(parents=True, exist_ok=True)

    runs = R.discover_runs(out / "runs", a.tag, a.seed)
    if runs.empty:
        raise SystemExit(f"No finished runs (DONE) with tag={a.tag!r} seed={a.seed} in {out / 'runs'}")
    selected = R.select_best(runs)
    runs.assign(selected=runs["run_dir"].isin(selected["run_dir"])).drop(columns="run_dir").to_csv(
        res_dir / "selection.csv", index=False)
    models = list(selected["model"])
    seeds = sorted(a.seeds) if a.seeds is not None else [a.seed]
    log.info("Runs found: %d; selected (best val TSS>=M per model): %s", len(runs),
             {r.model: f"lr={r.lr:g}" for r in selected.itertuples()})

    # ---- pre-registered argmax tables (unchanged definitions)
    t_split = R.split_table(selected)
    t_main, t_cls, c_test, aligned = R.test_tables(selected)
    c_val = R.val_confusion(selected)
    t_cost = R.cost_table(selected)
    t_boot = run_bootstrap(aligned, models)
    tables = {"table_train_val_test": t_split, "table_test_main": t_main, "table_test_per_class": t_cls,
              "confusion": pd.concat([c_val, c_test]), "table_cost": t_cost, "bootstrap": t_boot}

    # ---- seeds, thresholds (validation only), post-hoc tables
    sel_by_seed = R.select_seed_runs(out / "runs", a.tag, selected, seeds) if a.seeds is not None else {a.seed: selected}
    all_runs = pd.concat([runs] + [R.discover_runs(out / "runs", a.tag, s) for s in seeds if s != a.seed],
                         ignore_index=True)
    thr = R.run_thresholds(all_runs)                      # reads preds_val.csv only
    long = pd.concat([R.eval_long(sel, thr) for sel in sel_by_seed.values()], ignore_index=True)
    long_seed = long[long["seed"] == a.seed]
    c_tuned = R.confusion_variants(selected, thr[thr["seed"] == a.seed], ["tuned_tss"])
    pooled_tuned = run_pooled(sel_by_seed, thr, models, "tuned_tss", ["tss_ge_m"])
    pooled_argmax = run_pooled(sel_by_seed, thr, models, "argmax", ["tss_ge_m"])
    auc_boot = run_auc(sel_by_seed, thr, models)
    h5_pairs = [p for p in FOCAL_PAIRS[-1:] if p[0] in models and p[1] in models]
    h5 = pd.concat([run_pooled(sel_by_seed, thr, models, v, H5_METRICS, h5_pairs) for v in ("tuned_tss", "argmax")],
                   ignore_index=True) if h5_pairs else pd.DataFrame()
    tables.update({"thresholds": thr, "table_eval_long": long, "confusion_tuned": c_tuned,
                   "bootstrap_tuned": pooled_tuned, "bootstrap_seeds": pooled_argmax, "bootstrap_auc": auc_boot,
                   "bootstrap_h5": h5})
    if len(seeds) > 1:
        tables["table_eval_summary"] = R.eval_seed_summary(long)
    for name, df in tables.items():
        df.to_csv(res_dir / f"{name}.csv", index=False)

    # ---- figures
    for r in runs.itertuples():
        R.fig_loss(Path(r.run_dir), fig_dir)
    R.fig_val_tss(runs, selected, fig_dir)
    for m in models:
        R.fig_confusion(m, c_val, c_test, fig_dir)
        R.fig_confusion(m, c_tuned[c_tuned["split"] == "val"], c_tuned[c_tuned["split"] == "test"], fig_dir, "tuned")
    R.fig_roc_pr(selected, aligned, fig_dir)
    R.fig_threshold_curves(selected, thr[thr["seed"] == a.seed], fig_dir)

    # ---- markdown summary
    sel = runs.assign(selected=runs["run_dir"].isin(selected["run_dir"])).drop(columns="run_dir")
    n_x = int(t_main["n_X"].iloc[0])
    x_note = " – X metrics descriptive only" if n_x < 10 else ""
    seeds_txt = f"seeds {seeds}, mean ± sd (ddof=1), lr per model fixed from seed {a.seed}" if len(seeds) > 1 else f"seed {a.seed}"
    multi_note = ("Pooled = mean over seeds of the per-resample difference; `supported` only if the pooled 95% CI excludes 0 "
                  "on the positive side AND the difference is positive in every seed (decisions.md §3); `reverse` "
                  "likewise negative; `inconsistent across seeds` = pooled CI excludes 0 but seed signs differ (no claim).")
    boot_cols = ["comparison", "metric", "variant", "estimate", "ci_lo", "ci_hi", "status"]
    seed_cols = [f"est_s{s}" for s in seeds]
    md = [f"# Results summary (tag: {a.tag or 'none'}, selection seed {a.seed})", "",
          "Model selection: per model, the lr with the best **validation** TSS≥M (argmax). The test set is never used "
          "for selection or threshold tuning.", "", "## Runs and selection", R.md_table(sel), "",
          "# A. Pre-registered (argmax)", "",
          "## Train / validation / test (selected runs; E0 = persistence)", R.md_table(t_split), "",
          f"## Test set, argmax (n = {int(t_main['n'].iloc[0])}; X samples: {n_x}{x_note})", R.md_table(t_main), "",
          "### Per class (test, argmax)", R.md_table(t_cls), "",
          "## Cost", R.md_table(t_cost, 2), "",
          f"## Paired cluster bootstrap on test, argmax, seed {a.seed} ({N_BOOT} resamples of HARPs, seed {BOOT_SEED}; "
          "difference = second − first; 95% percentile CI)", R.md_table(t_boot), "",
          "`supported` = CI excludes 0 with a positive difference; `reverse` = CI excludes 0, negative; X metrics are "
          "`descriptive only` when the test set has fewer than 10 X samples.", "",
          "# B. Post-hoc (Change Log 15): thresholds tuned on validation only, AUC", "",
          "τ1 on P(≥M)=p1+p2 maximises val TSS≥M; τ2 on p2/(p1+p2) maximises val HSS_X (one-vs-rest); grid 0.01–0.99, ties → higher. "
          "`tuned_hss` = τ1 maximising val HSS≥M (sensitivity). AUCs use P(≥M) and do not depend on thresholds; E0's "
          "AUC is that of its 0/1 call. `far` = FP/(TP+FP).", "",
          f"## Test, {seeds_txt}", R.md_table(R.eval_md_table(long if len(seeds) > 1 else long_seed, "test",
                                                               ["persistence", *R.VARIANTS])), "",
          f"## Validation (where the thresholds were fitted), {seeds_txt}",
          R.md_table(R.eval_md_table(long if len(seeds) > 1 else long_seed, "val", ["persistence", *R.VARIANTS])), "",
          f"## One-vs-rest TSS per class, test, seed {a.seed}",
          R.md_table(R.eval_md_table(long_seed, "test", ["persistence", *R.VARIANTS],
                                     ["tss_ovr_<M", "tss_ovr_M", "tss_ovr_X", "macro_tss"])), "",
          f"## Pooled paired bootstrap, tuned τ (main post-hoc), TSS≥M, seeds {seeds}",
          R.md_table(pooled_tuned[boot_cols + seed_cols + ["sign_consistent"]]), "", multi_note, "",
          f"## Pooled paired bootstrap, ROC-AUC and PR-AUC of P(≥M), seeds {seeds}",
          R.md_table(auc_boot[[c for c in boot_cols if c != "variant"] + seed_cols + ["sign_consistent"]]), "",
          f"## Secondary: pooled bootstrap, argmax, TSS≥M, seeds {seeds}",
          R.md_table(pooled_argmax[boot_cols + seed_cols + ["sign_consistent"]]), "",
          f"## H5: E6 vs E5 (HSS≥M, precision≥M, recall≥M), seeds {seeds}",
          R.md_table(h5[boot_cols + seed_cols + ["sign_consistent"]]) if len(h5) else "E5/E6 not both available.", "",
          "## Chosen thresholds per run", R.md_table(thr.sort_values(["model", "lr", "seed"])), ""]
    (res_dir / "results_summary.md").write_text("\n".join(md), encoding="utf-8")
    print((res_dir / "results_summary.md").read_text(encoding="utf-8"))
    print(f"\nTables: {res_dir}\nFigures: {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
