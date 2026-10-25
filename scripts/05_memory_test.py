"""Module 5 — GPU memory / cost test with random tensors (no data needed).

For each of E1-E4, image size in {128, 64} and batch in {16, 8}: 3 training steps (AdamW, autocast float16,
GradScaler), peak GPU memory (torch.cuda.max_memory_allocated, reset per config), seconds/step, and OOM if it happens.
Also parameters per model and batch-1 inference latency (median of 200 runs after 20 warm-up, cuda synchronize).
Writes outputs/models/memory_test.json and prints a table.

Run from the project root (PowerShell):
    python scripts/05_memory_test.py
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.config import load_config  # noqa: E402
from flarecast.models import build_model, count_params  # noqa: E402

log = logging.getLogger("memory_test")
MODELS = ["E1", "E2", "E3", "E4"]


def make_inputs(name: str, batch: int, size: int, device: str) -> tuple[torch.Tensor, ...]:
    r = lambda *s: torch.randn(*s, device=device)  # noqa: E731
    if name == "E1":
        return (r(batch, 1, size, size),)
    if name == "E2":
        return (r(batch, 31, 1, size, size),)
    return (r(batch, 16, 1, size, size), r(batch, 16, 1, size, size))


def train_config(name: str, size: int, batch: int, steps: int, device: str) -> dict:
    """Run ``steps`` training steps; returns peak memory (MB) and s/step, or status OOM."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    res = {"model": name, "image_size": size, "batch": batch, "status": "ok", "peak_mem_mb": None,
           "s_per_step": None, "s_per_step_first": None}
    try:
        model = build_model(name).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        x = make_inputs(name, batch, size, device)
        y = torch.randint(0, 3, (batch,), device=device)
        times = []
        for _ in range(steps):
            torch.cuda.synchronize()
            tic = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(*x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - tic)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} produced a non-finite loss")
        res["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        res["s_per_step_first"] = round(times[0], 3)
        steady = times[1:] or times
        res["s_per_step"] = round(sum(steady) / len(steady), 3)  # excludes the first (cuDNN autotune) step
    except torch.cuda.OutOfMemoryError as e:  # the only exception that is expected and recorded
        res["status"] = "OOM"
        res["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        log.warning("OOM for %s size=%d batch=%d: %s", name, size, batch, str(e).splitlines()[0])
    finally:
        model = opt = scaler = x = y = loss = None  # noqa: F841  (release before the next config)
        gc.collect()
        torch.cuda.empty_cache()
    return res


@torch.no_grad()
def latency(name: str, size: int, device: str, warmup: int = 20, runs: int = 200) -> dict:
    """Batch-1 inference latency in ms (fp32, eval mode), median of ``runs`` after ``warmup`` runs."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        model = build_model(name).to(device).eval()
        x = make_inputs(name, 1, size, device)
        for _ in range(warmup):
            model(*x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(runs):
            tic = time.perf_counter()
            model(*x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - tic) * 1000)
        return {"model": name, "image_size": size, "latency_ms_median": round(statistics.median(ts), 2),
                "latency_ms_p95": round(sorted(ts)[int(0.95 * len(ts)) - 1], 2), "runs": runs, "warmup": warmup}
    finally:
        model = x = None  # noqa: F841
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 8])
    ap.add_argument("--out", default=None, help="default <outputs>/models/memory_test.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: this test measures GPU memory and must run on the GPU.")
    cfg = load_config(args.config)
    device = "cuda"
    free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    env = {"gpu": props.name, "total_mem_mb": round(total / 2**20), "free_mem_mb_at_start": round(free / 2**20),
           "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "steps": args.steps, "amp": "autocast float16 + GradScaler", "optimizer": "AdamW wd=1e-4"}
    print(f"GPU {env['gpu']}: {env['total_mem_mb']} MB total, {env['free_mem_mb_at_start']} MB free at start")

    params = {m: count_params(build_model(m)) for m in args.models}
    train = []
    for m in args.models:
        for size in args.sizes:
            for batch in args.batches:
                r = train_config(m, size, batch, args.steps, device)
                train.append(r)
                log.info("%s size=%d batch=%d -> %s peak %s MB, %s s/step", m, size, batch, r["status"],
                         r["peak_mem_mb"], r["s_per_step"])
    lat = [latency(m, s, device) for m in args.models for s in args.sizes]

    report = {"environment": env, "params": params, "training": train, "inference_batch1": lat}
    out = Path(args.out) if args.out else cfg["paths"]["outputs_dir"] / "models" / "memory_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lat_d = {(r["model"], r["image_size"]): r for r in lat}
    print("\nmodel  params     size batch  status  peak MB  s/step  (first)  | batch-1 latency ms (median)")
    for r in train:
        l = lat_d[(r["model"], r["image_size"])]["latency_ms_median"]
        pk = "-" if r["peak_mem_mb"] is None else f"{r['peak_mem_mb']:.0f}"
        print(f"{r['model']:<6} {params[r['model']]:>9,} {r['image_size']:>5} {r['batch']:>5}  {r['status']:<6} "
              f"{pk:>7}  {str(r['s_per_step']):>6}  {str(r['s_per_step_first']):>6}   | {l}")
    print(f"\nWritten to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
