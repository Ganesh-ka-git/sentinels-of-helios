"""Module 6 — queue E1..E5 x lr {1e-3, 3e-4} x seed 0. Resumable: runs with a DONE marker are skipped.

Each run is a separate process (frees GPU memory and DataLoader workers). A failed run is reported, NOT retried; the
queue continues with the next run and the exit code is 1 if anything failed.

Run from the project root (PowerShell):
    conda run -n flare --no-capture-output python scripts/06_run_all.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.train import run_name  # noqa: E402

MODELS = ["E1", "E2", "E3", "E4", "E5"]
LRS = [1e-3, 3e-4]
SEEDS = [0]


def pending_runs(models, lrs, seeds, runs_dir: Path, tag: str | None = None):
    """(done, pending) lists of (model, lr, seed); a run is done iff its directory has a DONE marker."""
    done, pending = [], []
    for s in seeds:
        for m in models:
            for lr in lrs:
                (done if (runs_dir / run_name(m, lr, s, tag) / "DONE").exists() else pending).append((m, lr, s))
    return done, pending


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--lrs", nargs="+", type=float, default=LRS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: config)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    runs_dir = cfg["paths"]["outputs_dir"] / "runs"
    done, pending = pending_runs(a.models, a.lrs, a.seeds, runs_dir)
    print(f"{len(done)} run(s) already DONE, {len(pending)} pending", flush=True)
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
