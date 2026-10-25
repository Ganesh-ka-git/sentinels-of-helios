"""Module 6 — train one run (model, lr, seed). Output: outputs/runs/<model>_lr<lr>_s<seed>[_<tag>]/ with a DONE marker.

Run from the project root (PowerShell):
    conda run -n flare --no-capture-output python scripts/06_train.py --model E1 --lr 3e-4 --seed 0
Smoke test:
    conda run -n flare --no-capture-output python scripts/06_train.py --model E4 --max-epochs 2 --tag smoke
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.train import MODE_OF, train_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODE_OF))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--tag", default=None, help="suffix for the run directory (e.g. smoke)")
    ap.add_argument("--num-workers", type=int, default=None, help="default: loader.num_workers from the config")
    ap.add_argument("--force", action="store_true", help="retrain even if a DONE marker exists")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    run_dir = train_run(a.model, a.lr, a.seed, cfg=cfg, max_epochs=a.max_epochs, tag=a.tag,
                        num_workers=a.num_workers, force=a.force)
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
