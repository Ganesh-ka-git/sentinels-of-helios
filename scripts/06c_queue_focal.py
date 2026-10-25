"""Module 6c — queue the focal-loss models and the extra seeds (decisions.md Change Log 16, §9).

Stage 1: E6, E7, E8 x lr {1e-3, 3e-4} x seed 0 (6 runs).
Stage 2: seeds 1 and 2 for every model E1..E8 at the lr chosen per model on seed 0 (best VALIDATION TSS>=M under argmax,
         the same selection as 07_evaluate.py; the test set is never read). Needs all of E1..E8 finished on seed 0; a model
         whose seed-0 runs are missing (e.g. a failed stage-1 run) is reported and skipped, and the exit code is 1.
Resumable: runs with a DONE marker are skipped. Each run is its own process (06_train.py). A failed run is reported,
NOT retried. Supersedes 06b_queue_seeds.py (which covers E1..E5 only; its runs are skipped here if already DONE).

Run from the project root (PowerShell):
    conda run -n flare --no-capture-output python scripts/06c_queue_focal.py --dry-run
    conda run -n flare --no-capture-output python scripts/06c_queue_focal.py
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

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
LRS = RUN_ALL.LRS
FOCAL_MODELS = ["E6", "E7", "E8"]
ALL_MODELS = ["E1", "E2", "E3", "E4", "E5", *FOCAL_MODELS]
EXTRA_SEEDS = [1, 2]


def stage1_jobs(runs_dir: Path):
    """(done, pending) (model, lr, seed) for E6-E8 x lr x seed 0."""
    return RUN_ALL.pending_runs(FOCAL_MODELS, LRS, [0], runs_dir)


def choose_lrs(runs_dir: Path, models: list[str] = ALL_MODELS) -> tuple[dict[str, float], list[str]]:
    """({model: lr with the best seed-0 validation TSS>=M}, models with an incomplete seed-0 pair of runs).
    A model is only chosen if BOTH lr runs are DONE, so the choice never depends on a partial comparison."""
    runs = R.discover_runs(runs_dir, None, 0)
    chosen, missing = {}, []
    for m in models:
        sub = runs[runs["model"] == m]
        if len({round(float(x), 12) for x in sub["lr"]}) < len(LRS):
            missing.append(m)
            continue
        chosen[m] = float(R.select_best(sub).iloc[0]["lr"])
    return chosen, missing


def stage2_jobs(chosen: dict[str, float], seeds: list[int], runs_dir: Path):
    """(done, pending) (model, lr, seed): seed by seed, models in order, at the chosen lr."""
    done, pending = [], []
    for s in seeds:
        for m in ALL_MODELS:
            if m in chosen:
                d, p = RUN_ALL.pending_runs([m], [chosen[m]], [s], runs_dir)
                done += d
                pending += p
    return done, pending


def _run(jobs, a, label: str) -> list[str]:
    failed = []
    for i, (m, lr, s) in enumerate(jobs, 1):
        print(f"\n===== {label} [{i}/{len(jobs)}] {run_name(m, lr, s)} =====", flush=True)
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
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=EXTRA_SEEDS)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: config)")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs (stage 2 only if its lr choice is possible)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if 0 in a.seeds:
        raise SystemExit("Seed 0 is stage 1 (E6-E8) and the existing E1-E5 queue; this option is for extra seeds")
    cfg = load_config(a.config)
    runs_dir = cfg["paths"]["outputs_dir"] / "runs"

    done1, pend1 = stage1_jobs(runs_dir)
    print(f"Stage 1 (E6-E8 x lr x seed 0): {len(done1)} DONE, {len(pend1)} pending: {[run_name(*p) for p in pend1]}")
    failed = [] if a.dry_run else _run(pend1, a, "stage 1")

    chosen, missing = choose_lrs(runs_dir)
    print("\nChosen lr per model (best seed-0 validation TSS>=M, argmax):")
    for m in ALL_MODELS:
        print(f"  {m}: " + (f"lr = {chosen[m]:g}" if m in chosen else "NOT AVAILABLE (seed-0 runs incomplete)"))
    if missing:
        print(f"!!!!! seed-0 runs incomplete for {missing}: those models are skipped in stage 2", flush=True)
    done2, pend2 = stage2_jobs(chosen, a.seeds, runs_dir)
    print(f"\nStage 2 (seeds {a.seeds}): {len(done2)} DONE, {len(pend2)} pending: {[run_name(*p) for p in pend2]}",
          flush=True)
    if not a.dry_run:
        failed += _run(pend2, a, "stage 2")
    print(f"\nFinished. Failed runs: {failed or 'none'}; models skipped: {missing or 'none'}")
    return 1 if (failed or missing) and not a.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
