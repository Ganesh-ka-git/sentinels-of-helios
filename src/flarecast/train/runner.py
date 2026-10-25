"""Module 6 — training runner (decisions.md §9).

One run = one (model, lr, seed). AdamW (wd 1e-4), fp16 autocast + GradScaler, cross-entropy (class-weighted for E5,
weights from the TRAIN split only), early stopping (patience 5) on validation TSS>=M, best checkpoint = best validation
TSS>=M. After training, softmax probabilities of the best checkpoint are written for train, val and test.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..dataset import class_weights, filter_available, focal_alpha, load_samples, make_loaders
from ..dataset.flare_dataset import SPLITS
from ..evaluate.metrics import probs_to_pred, summarise
from ..models import build_model, count_params

log = logging.getLogger(__name__)

MODE_OF = {"E1": "static", "E2": "single", "E3": "dual", "E4": "dual", "E5": "dual",
           "E6": "dual", "E7": "single", "E8": "static"}
FOCAL_MODELS = ("E6", "E7", "E8")   # exploratory focal-loss models (Change Log 16)
FOCAL_GAMMA = 2.0                   # Lin et al. 2017, fixed
WEIGHT_DECAY = 1e-4
PATIENCE = 5
MIN_AVAILABLE = 0.95
HISTORY_COLS = ["epoch", "train_loss", "val_loss", "val_tss_ge_m", "val_recall_M", "val_recall_X", "seconds", "data_wait_s", "is_best"]
RUN_FILES = ["history.csv", "best.pt", "cost.json", "run.json", "DONE", "train.log",
             *[f"preds_{s}.csv" for s in SPLITS]]


# ---------------------------------------------------------------- naming / seeding
def run_name(model: str, lr: float, seed: int, tag: str | None = None) -> str:
    """'E4_lr0.0003_s0' (+ '_<tag>')."""
    return f"{model}_lr{lr:g}_s{seed}" + (f"_{tag}" if tag else "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Seed python/numpy in each DataLoader worker from the torch worker seed (derived from the loader seed)."""
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


# ---------------------------------------------------------------- loss
def loss_weights(model_name: str, samples: pd.DataFrame) -> torch.Tensor | None:
    """Per-class loss factors from the TRAIN rows of ``samples``: E5 w_c = N/(K*N_c) (class-weighted CE);
    E6-E8 focal alpha_c (inverse sqrt class frequency, mean 1); None (plain cross-entropy) otherwise."""
    if model_name == "E5":
        return class_weights(samples)
    if model_name in FOCAL_MODELS:
        return focal_alpha(samples)
    return None


def loss_gamma(model_name: str) -> float | None:
    """Focal gamma for E6-E8, None for cross-entropy models."""
    return FOCAL_GAMMA if model_name in FOCAL_MODELS else None


def focal_loss_sum(logits: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """Per-sample focal loss -alpha_y (1 - p_y)^gamma log p_y (unreduced, shape [B])."""
    logp = F.log_softmax(logits.float(), dim=1).gather(1, y[:, None])[:, 0]
    loss = -((1.0 - logp.exp()) ** gamma) * logp
    return loss if alpha is None else alpha[y] * loss


def batch_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None, gamma: float | None = None):
    """(mean loss for backward, summed loss, normaliser). Cross-entropy (gamma None): with weights the mean is the
    weighted mean (sum w_y * l / sum w_y), identical to ``F.cross_entropy(..., weight=w)``. Focal (gamma set; ``w`` is
    alpha): the plain mean over the batch (normaliser = batch size)."""
    if gamma is not None:
        s = focal_loss_sum(logits, y, w, gamma).sum()
        den = torch.tensor(float(len(y)), device=y.device)
        return s / den, s.detach(), den
    s = F.cross_entropy(logits.float(), y, weight=w, reduction="sum")
    den = w[y].sum() if w is not None else torch.tensor(float(len(y)), device=y.device)
    return s / den, s.detach(), den.detach()


# ---------------------------------------------------------------- forward helpers
def _to_device(x, device):
    if isinstance(x, (tuple, list)):
        return tuple(t.to(device, non_blocking=True) for t in x)
    return x.to(device, non_blocking=True)


def _forward(model, x):
    return model(*x) if isinstance(x, tuple) else model(x)


@torch.no_grad()
def predict(model, loader, device: str, w: torch.Tensor | None = None,
            gamma: float | None = None) -> tuple[pd.DataFrame, float]:
    """Eval-mode softmax probabilities + mean loss (same loss definition as training) over a loader."""
    model.eval()
    rows, s_tot, d_tot = [], 0.0, 0.0
    amp = device == "cuda"
    for x, y, meta in loader:
        x, y = _to_device(x, device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = _forward(model, x)
        _, s, d = batch_loss(logits, y, w, gamma)
        s_tot, d_tot = s_tot + s.item(), d_tot + d.item()
        p = torch.softmax(logits.float(), dim=1).cpu().numpy()
        rows.append(pd.DataFrame({"harpnum": meta["harpnum"].numpy(), "t": list(meta["t"]), "label": y.cpu().numpy(),
                                  "persistence_label": meta["persistence_label"].numpy(),
                                  "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2]}))
    df = pd.concat(rows, ignore_index=True).sort_values(["t", "harpnum"]).reset_index(drop=True)
    return df, s_tot / d_tot


def val_metrics(df: pd.DataFrame) -> dict[str, float]:
    m = summarise(df["label"].to_numpy(), probs_to_pred(df[["p0", "p1", "p2"]].to_numpy()))
    return {"tss_ge_m": m["tss_ge_m"], "recall_M": m["recall_M"], "recall_X": m["recall_X"]}


# ---------------------------------------------------------------- cost
def _load_memory_test():
    """Import scripts/05_memory_test.py (its name is not a valid module name) to reuse its ``latency``."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "05_memory_test.py"
    spec = importlib.util.spec_from_file_location("memory_test_module5", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the run
def train_run(model_name: str, lr: float, seed: int, *, cfg: dict, max_epochs: int = 30, tag: str | None = None,
              num_workers: int | None = None, runs_dir: Path | None = None, samples: pd.DataFrame | None = None,
              frames_dir: Path | None = None, batch_size: int | None = None, force: bool = False,
              use_cache: bool | None = None) -> Path:
    """Train one run end-to-end; returns the run directory. Skips (returns) if a DONE marker exists unless ``force``."""
    if model_name not in MODE_OF:
        raise ValueError(f"model must be one of {sorted(MODE_OF)}, got {model_name!r}")
    runs_dir = Path(runs_dir) if runs_dir else cfg["paths"]["outputs_dir"] / "runs"
    run_dir = runs_dir / run_name(model_name, lr, seed, tag)
    if (run_dir / "DONE").exists() and not force:
        log.info("%s already DONE: skipping", run_dir.name)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    stale = [f for f in RUN_FILES if (run_dir / f).exists()]
    if stale:
        log.warning("Incomplete/previous run in %s; restarting from scratch (removing %s)", run_dir, stale)
        for f in stale:
            (run_dir / f).unlink()
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    try:
        return _train(model_name, lr, seed, cfg, max_epochs, tag, num_workers, run_dir, samples, frames_dir,
                      batch_size, use_cache)
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


def _train(model_name, lr, seed, cfg, max_epochs, tag, num_workers, run_dir, samples, frames_dir, batch_size,
           use_cache) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available: running on CPU (cost.json will have no GPU memory/latency)")
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    mode = MODE_OF[model_name]
    num_workers = cfg["loader"]["num_workers"] if num_workers is None else num_workers
    batch_size = cfg["loader"]["batch_size"] if batch_size is None else batch_size
    frames_dir = Path(frames_dir) if frames_dir else cfg["paths"]["frames_dir"]
    use_cache = cfg["loader"].get("use_cache", True) if use_cache is None else use_cache
    cache_dir = cfg["paths"].get("cache_dir")
    if samples is None:
        all_samples = load_samples(cfg["paths"]["samples_csv"])
        samples = filter_available(all_samples, frames_dir)
        for sp in SPLITS:
            tot, av = (all_samples["split"] == sp).sum(), (samples["split"] == sp).sum()
            if av < MIN_AVAILABLE * tot:
                raise RuntimeError(f"Split {sp}: only {av}/{tot} samples available (< {MIN_AVAILABLE:.0%}); stopping")
    w = loss_weights(model_name, samples)
    w_dev = w.to(device) if w is not None else None
    gamma = loss_gamma(model_name)

    loaders = make_loaders(mode, batch_size, num_workers, samples=samples, frames_dir=frames_dir, seed=seed,
                           worker_init_fn=worker_init_fn if num_workers > 0 else None, use_cache=use_cache,
                           cache_dir=cache_dir)
    for sp in ("train", "val"):
        if sp not in loaders:
            raise RuntimeError(f"No {sp} samples available")
    model = build_model(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    n_params = count_params(model)

    meta = {"model": model_name, "mode": mode, "lr": lr, "seed": seed, "tag": tag, "max_epochs": max_epochs,
            "batch_size": batch_size, "num_workers": num_workers, "use_cache": use_cache, "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
            "loss": "focal" if gamma is not None else ("weighted_ce" if w is not None else "ce"), "focal_gamma": gamma,
            "class_weights": None if w is None else [float(v) for v in w],   # E5 w_c, or focal alpha_c for E6-E8
            "n_samples": {sp: int((samples["split"] == sp).sum()) for sp in SPLITS}, "device": device,
            "torch": torch.__version__, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%s: %s params, mode %s, batch %d, device %s, class weights %s", run_dir.name, f"{n_params:,}", mode,
             batch_size, device, meta["class_weights"])

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_tss, best_epoch, bad = [], -np.inf, 0, 0
    for epoch in range(1, max_epochs + 1):
        tic = time.perf_counter()
        model.train()
        s_tot, d_tot, wait = 0.0, 0.0, 0.0
        it, t_wait = iter(loaders["train"]), time.perf_counter()
        for x, y, _ in it:
            wait += time.perf_counter() - t_wait          # time the GPU loop was blocked on the DataLoader
            x, y = _to_device(x, device), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = _forward(model, x)
            loss, s, d = batch_loss(logits, y, w_dev, gamma)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{run_dir.name}: non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            s_tot, d_tot = s_tot + s.item(), d_tot + d.item()
            t_wait = time.perf_counter()
        val_df, val_loss = predict(model, loaders["val"], device, w_dev, gamma)
        vm = val_metrics(val_df)
        if not np.isfinite(vm["tss_ge_m"]):
            raise ValueError(f"{run_dir.name}: validation TSS>=M undefined (a class is empty in val)")
        is_best = vm["tss_ge_m"] > best_tss
        if is_best:
            best_tss, best_epoch, bad = vm["tss_ge_m"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_tss_ge_m": best_tss, "run": meta},
                       run_dir / "best.pt")
        else:
            bad += 1
        row = {"epoch": epoch, "train_loss": s_tot / d_tot, "val_loss": val_loss, "val_tss_ge_m": vm["tss_ge_m"],
               "val_recall_M": vm["recall_M"], "val_recall_X": vm["recall_X"],
               "seconds": time.perf_counter() - tic, "data_wait_s": wait, "is_best": is_best}
        history.append(row)
        pd.DataFrame(history, columns=HISTORY_COLS).to_csv(run_dir / "history.csv", index=False)
        log.info("epoch %2d  train %.4f  val %.4f  TSS>=M %.3f  recall M %.3f X %.3f  %.0fs (data wait %.0fs)%s", epoch,
                 row["train_loss"], row["val_loss"], vm["tss_ge_m"], vm["recall_M"], vm["recall_X"], row["seconds"], wait,
                 "  *best*" if is_best else "")
        if bad >= PATIENCE:
            log.info("Early stopping: no val TSS>=M improvement for %d epochs (best epoch %d)", PATIENCE, best_epoch)
            break

    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    secs = [r["seconds"] for r in history]

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    for sp, ld in loaders.items():
        df, _ = predict(model, ld, device, w_dev, gamma)
        df.to_csv(run_dir / f"preds_{sp}.csv", index=False)
        log.info("wrote preds_%s.csv (%d rows)", sp, len(df))
    chk = val_metrics(pd.read_csv(run_dir / "preds_val.csv"))["tss_ge_m"]
    if abs(chk - best_tss) > 1e-9:
        raise RuntimeError(f"best.pt val TSS>=M {chk} differs from the recorded best {best_tss}")

    cost = {"model": model_name, "params": n_params,
            "peak_mem_allocated_bytes": peak, "peak_mem_allocated_mb": None if peak is None else round(peak / 2**20, 1),
            "s_per_epoch": float(np.mean(secs)), "total_train_s": float(np.sum(secs)), "epochs_run": len(history),
            "best_epoch": best_epoch, "best_val_tss_ge_m": best_tss, "latency_batch1": None}
    del model, opt, loaders
    if device == "cuda":
        cost["latency_batch1"] = _load_memory_test().latency(model_name, 128, device)
    (run_dir / "cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    (run_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"), encoding="utf-8")
    log.info("%s DONE (best epoch %d, val TSS>=M %.3f)", run_dir.name, best_epoch, best_tss)
    return run_dir
