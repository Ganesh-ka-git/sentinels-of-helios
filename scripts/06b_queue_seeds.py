"""Module 6b — queue seeds 1 and 2 for the lr chosen per model on seed 0 (decisions.md §3, §9).

Starts only once ALL seed-0 runs (E1..E5 x lr {1e-3, 3e-4}) have a DONE marker. Per model the lr with the best
VALIDATION TSS>=M on seed 0 is chosen (same selection as 07_evaluate.py: flarecast.evaluate.report.select_best), printed,
and then E1..E5 x chosen lr x seeds {1, 2} (10 runs) are run, one process each. Resumable: runs with a DONE marker are
skipped. A failed run is reported, NOT retried; the exit code is 1 if anything failed. This script only reads
outputs/runs/ (to select) and starts 06_train.py, which writes the new runs.

Run from the project root (PowerShell):
    conda run -n flare --no-capture-output python scripts/06b_queue_seeds.py            # exits if seed 0 is incomplete
    conda run -n flare --no-capture-output python scripts/06b_queue_seeds.py --wait     # poll until seed 0 is complete
    conda run -n flare --no-capture-output python scripts/06b_queue_seeds.py --dry-run  # select and print only
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.evaluate import report as R  # noqa: E402
from flarecast.train import run_name  # noqa: E402


def _load_run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "scripts" / "06_run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN_ALL = _load_run_all()
MODELS, LRS = RUN_ALL.MODELS, RUN_ALL.LRS
EXTRA_SEEDS = [1, 2]


def seed0_pending(runs_dir: Path) -> list[tuple[str, float, int]]:
    """Seed-0 runs of the base queue that are not DONE yet."""
    return RUN_ALL.pending_runs(MODELS, LRS, [0], runs_dir)[1]


def choose_lrs(runs_dir: Path) -> dict[str, float]:
    """model -> lr with the best validation TSS>=M among the finished seed-0 runs (test is never read)."""
    selected = R.select_best(R.discover_runs(runs_dir, None, 0))
    chosen = {r.model: float(r.lr) for r in selected.itertuples()}
    missing = [m for m in MODELS if m not in chosen]
    if missing:
        raise SystemExit(f"No finished seed-0 run for: {missing}")
    return chosen


def seed_jobs(chosen: dict[str, float], seeds: list[int], runs_dir: Path):
    """(done, pending) (model, lr, seed) lists for the chosen lr per model."""
    done, pending = [], []
    for s in seeds:
        for m in MODELS:
            d, p = RUN_ALL.pending_runs([m], [chosen[m]], [s], runs_dir)
            done += d
            pending += p
    return done, pending


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=EXTRA_SEEDS)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: config)")
    ap.add_argument("--wait", action="store_true", help="poll until all seed-0 runs are DONE")
    ap.add_argument("--poll-s", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true", help="select and print the jobs, start nothing")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if 0 in a.seeds:
        raise SystemExit("Seed 0 belongs to 06_run_all.py; this script queues the additional seeds only")
    cfg = load_config(a.config)
    runs_dir = cfg["paths"]["outputs_dir"] / "runs"

    while True:
        pending0 = seed0_pending(runs_dir)
        if not pending0:
            break
        names = [run_name(*p) for p in pending0]
        if not a.wait:
            raise SystemExit(f"{len(pending0)} seed-0 run(s) not DONE yet: {names}\n"
                             "Re-run when the seed-0 queue has finished (or use --wait).")
        print(f"{time.strftime('%H:%M:%S')} waiting for {len(pending0)} seed-0 run(s): {names}", flush=True)
        time.sleep(a.poll_s)

    chosen = choose_lrs(runs_dir)
    val = R.discover_runs(runs_dir, None, 0)
    print("Seed-0 validation TSS>=M per (model, lr):")
    print(R.md_table(val[["model", "lr", "val_tss_ge_m"]].sort_values(["model", "lr"]), 4))
    print("\nChosen lr per model (best validation TSS>=M, seed 0):")
    for m in MODELS:
        print(f"  {m}: lr = {chosen[m]:g}")
    done, pending = seed_jobs(chosen, a.seeds, runs_dir)
    print(f"\n{len(done)} run(s) already DONE, {len(pending)} pending", flush=True)
    if a.dry_run:
        print("Dry run:", [run_name(*p) for p in pending])
        return 0

    failed = []
    for i, (m, lr, s) in enumerate(pending, 1):
        print(f"\n===== [{i}/{len(pending)}] {run_name(m, lr, s)} =====", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "06_train.py"), "--model", m, "--lr", str(lr),
               "--seed", str(s), "--max-epochs", str(a.max_epochs)]
        if a.num_workers is not None:
            cmd += ["--num-workers", str(a.num_workers)]
        if a.config:
            cmd += ["--config", a.config]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!!!!! {run_name(m, lr, s)} FAILED (exit code {rc}); not retrying", flush=True)
            failed.append(run_name(m, lr, s))
    print(f"\nFinished. Failed runs: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
