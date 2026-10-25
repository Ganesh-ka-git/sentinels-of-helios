"""Module 7 — evaluation and reporting from finished runs (outputs/runs/*/DONE).

Selection per model = the lr with the best VALIDATION TSS>=M (test is never used for selection). Writes CSV tables and
results_summary.md to outputs/results/ and 300-dpi PNG + PDF figures to outputs/figures/ (a sub-folder <tag>/ when
--tag is given, e.g. smoke).

Run from the project root (PowerShell):
    conda run -n flare --no-capture-output python scripts/07_evaluate.py
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
from flarecast.evaluate.bootstrap import paired_bootstrap  # noqa: E402
from flarecast.evaluate.metrics import probs_to_pred  # noqa: E402

log = logging.getLogger("evaluate")
N_BOOT, BOOT_SEED = 2000, 0
CHAIN = [("E1", "E2"), ("E2", "E3"), ("E3", "E4"), ("E4", "E5")]


def run_bootstrap(aligned: dict[str, pd.DataFrame], models: list[str]) -> pd.DataFrame:
    first = aligned["E0"]
    harp, y = first["harpnum"].to_numpy(), first["label"].to_numpy()
    preds = {"E0": first["persistence_label"].to_numpy(),
             **{m: probs_to_pred(aligned[m][R.PROB_COLS].to_numpy()) for m in models}}
    have = set(preds)
    chain = [(a, b) for a, b in CHAIN if a in have and b in have]
    vs_e0 = [("E0", m) for m in models]
    rows = []
    if chain or vs_e0:
        rows += paired_bootstrap(harp, y, preds, chain + vs_e0, ["tss_ge_m"], N_BOOT, BOOT_SEED)
    if ("E4", "E5") in chain:
        rows += paired_bootstrap(harp, y, preds, [("E4", "E5")], ["recall_M", "recall_X"], N_BOOT, BOOT_SEED)
    skipped = [f"{a}vs{b}" for a, b in CHAIN if (a, b) not in chain]
    if skipped:
        log.warning("Comparisons skipped (model missing): %s", skipped)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="evaluate only runs trained with this tag (default: untagged runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
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
    log.info("Runs found: %d; selected (best val TSS>=M per model): %s", len(runs),
             {r.model: f"lr={r.lr:g}" for r in selected.itertuples()})

    t_split = R.split_table(selected)
    t_main, t_cls, c_test, aligned = R.test_tables(selected)
    c_val = R.val_confusion(selected)
    t_cost = R.cost_table(selected)
    t_boot = run_bootstrap(aligned, models)
    for name, df in {"table_train_val_test": t_split, "table_test_main": t_main, "table_test_per_class": t_cls,
                     "confusion": pd.concat([c_val, c_test]), "table_cost": t_cost, "bootstrap": t_boot}.items():
        df.to_csv(res_dir / f"{name}.csv", index=False)

    for r in runs.itertuples():
        R.fig_loss(Path(r.run_dir), fig_dir)
    R.fig_val_tss(runs, selected, fig_dir)
    for m in models:
        R.fig_confusion(m, c_val, c_test, fig_dir)

    sel = runs.assign(selected=runs["run_dir"].isin(selected["run_dir"])).drop(columns="run_dir")
    n_x = int(t_main["n_X"].iloc[0])
    md = [f"# Results summary (tag: {a.tag or 'none'}, seed {a.seed})", "",
          "Model selection: per model, the lr with the best **validation** TSS≥M. The test set is never used for "
          "selection.", "", "## Runs and selection", R.md_table(sel), "",
          "## Train / validation / test (selected runs; E0 = persistence)", R.md_table(t_split), "",
          f"## Test set, E0–E5 (n = {int(t_main['n'].iloc[0])}; X samples: {n_x}"
          + (" – X metrics descriptive only)" if n_x < 10 else ")"), R.md_table(t_main), "",
          "### Per class (test)", R.md_table(t_cls), "",
          "## Cost", R.md_table(t_cost, 2), "",
          f"## Paired cluster bootstrap on test ({N_BOOT} resamples of HARPs, seed {BOOT_SEED}; difference = second − "
          "first; 95% percentile CI)", R.md_table(t_boot), "",
          "`supported` = CI excludes 0 with a positive difference; `reverse` = CI excludes 0, negative; X metrics are "
          "`descriptive only` when the test set has fewer than 10 X samples.", ""]
    (res_dir / "results_summary.md").write_text("\n".join(md), encoding="utf-8")
    print((res_dir / "results_summary.md").read_text(encoding="utf-8"))
    print(f"\nTables: {res_dir}\nFigures: {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
