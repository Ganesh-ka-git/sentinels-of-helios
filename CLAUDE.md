# CLAUDE.md — Solar Flare Forecasting Dissertation

## Who and what
Postgraduate data science dissertation (submission **1 November 2026**). Short-term, active-region-specific solar flare
forecasting from SDO/HMI SHARP Br magnetograms with GOES flare labels. The full research design is in **`decisions.md`**,
which is the single source of truth. Read it before any task.

## Non-negotiable rules
1. **Do not silently change the research design.** If something in `decisions.md` cannot be implemented, stop, explain why,
   and propose the smallest defensible alternative. Every approved change gets a row in the Change Log in `decisions.md`.
2. **Never fabricate** data, labels, associations, sample counts, results or availability. Missing data are reported as missing.
3. **No future information** may enter an input sequence: only records with T_REC ≤ t.
4. **Never treat code as having run successfully** until the user pastes the actual output.
5. Do not add architectures, losses, inputs or experiments beyond E0–E5 (+ optional E1w) without explicit approval.
6. Do not hard-code local paths or the JSOC e-mail; everything comes from `config/config.yaml`.
7. Do not swallow errors. Log them and fail loudly.

## Environment
- **Windows** laptop, NVIDIA GPU with **4 GB** VRAM, CUDA configured. **PyTorch only.**
- Use `pathlib` for all paths. Any script using a multi-worker `DataLoader` must have an `if __name__ == "__main__":` guard
  (Windows uses spawn). Prefer `persistent_workers=True`, and keep `num_workers` configurable (default 4).
- Training data are memory-mapped `.npy` files on **local disk**. Google Drive is for backup only, never for training reads.
- Python ≥ 3.10. Dependencies are in `requirements.txt`. Install torch separately (see README).
- Python environment: conda env flare. Run Python commands as conda run -n flare python ... (or conda run -n flare python -m pytest -q)

## Project layout
```
config/config.yaml        all settings (paths, JSOC email placeholder, pilot month, thresholds)
src/flarecast/            importable package
  config.py               load config + resolve paths
  data/jsoc.py            SHARP keyword queries, export/download
  data/goes.py            GOES flare catalogue (HEK via SunPy), class parsing
  labels/association.py   flare -> HARP association
  labels/samples.py       sample grid, input windows, labels, exclusions
  labels/index.py         month-by-month index, completeness rule, HARP-grouped split
  preprocess/             frames.py, pipeline.py, monitor.py: FITS -> 128x128 float16 .npy
  dataset/flare_dataset.py  FlareDataset (packed cache or per-frame), make_loaders, class_weights
  dataset/cache.py        build_cache / verify_cache (packed float16 frame cache)
  models/models.py        SharedEncoder, E1-E5, build_model, persistence_predict
  train/runner.py         train_run (AMP, early stopping on val TSS>=M, preds, cost.json)
  evaluate/               metrics.py, bootstrap.py, report.py
  (later) explain/
scripts/                  numbered entry points (01_feasibility.py ... 05_memory_test.py, watch_*.py)
tests/                    pytest unit tests (run: python -m pytest -q)
outputs/                  run outputs (reports, CSV/JSON results); gitignored
data/                     raw/processed data; gitignored
```

## Module workflow (follow for every module)
1. State the purpose in a few sentences.
2. Write the code with docstrings, type hints and input/output shape checks.
3. Add or extend unit tests: label boundaries, temporal ordering, shapes, forward passes.
4. Say exactly how to run it on Windows (PowerShell) and what output to expect.
5. Say how to recognise failure.
6. **Wait for the user's real output** before building on it.

## Module plan
| # | Module | Status |
|---|---|---|
| 1 | Feasibility pilot + download benchmark (`01_feasibility.py`, `01b_download_benchmark.py`) | **done**; `tests/test_module1.py` (28 test functions) |
| 2 | Full index + split (`02_build_index.py`, `02b_finalize_index.py`) | **done**; `tests/test_module2.py` (15) |
| 3 | Selective download + preprocessing to per-frame float16 `.npy` (`03_download_preprocess.py`) | **done**, 139,176 frames on disk; `test_module3*.py` (32) + `test_monitor.py` (4) |
| 4 | Dataset/DataLoader (static, single, dual) + loader check (`04_check_loader.py`) | **done**; `tests/test_module4.py` (17) |
| 4b | Packed float16 frame cache (`04b_build_cache.py`, `dataset/cache.py`), Change Log 14 | **done**, 139,176 frames; `tests/test_cache.py` (8) |
| 5 | Models E1–E5 + E0 persistence + GPU memory test (`05_memory_test.py`) | **done**; `tests/test_module5.py` (15) |
| 6 | Training runner (`06_train.py`, `06_run_all.py`, `train/runner.py`) | **code done**, smoke-tested (E1, E4); full queue not started; `tests/test_module6_7.py` |
| 7 | Evaluation (`07_evaluate.py`, `evaluate/`: metrics, cluster bootstrap, tables, figures) | **code done**, smoke-tested on E1/E4 |
| 8 | Grad-CAM + figures and tables for the thesis | pending |

Experiment priority if time is short: E1, E2, E3, then E4, E5, then Grad-CAM, then E1w.

## Current state
- **Period**: kept months **2024-01 … 2025-06** (18 months). Indexed 2023-01 … 2025-06; a month is kept only if ≥ 80% of
  its ≥M flares carry a NOAA number (all 2023 months fail, 0–25%; all 2024-01 … 2025-06 pass, 83–100%). See decisions.md §10, Change Log 10.
- **Final index** (`outputs/index/`): `samples_final.csv` (use rows with `included`), `frames_needed_final.csv`
  (harpnum, time, month, split; 139,176 frames), `index_report_final.json`. Per-month caches in `outputs/index/<YYYY-MM>/`.
  Older `samples_all.csv` / `index_report.json` are superseded.
- **Split counts** (included samples, classes 0/1/2, HARPs): train 2,980 (2727/237/16, 259), val 816 (728/73/15, 59),
  test 856 (775/74/7, 63). Test has only 7 X samples (< 10): report X descriptively (decisions.md §11).
- **Frame cache** (Change Log 14): `data/processed/cache/frames_f16.npy` [139,176,128,128] float16 (4.3 GB) + `frames_index.csv` (harpnum, time=`YYYYMMDDTHHMM`, row). Built by `04b_build_cache.py` (1,000 random rows verified identical); rebuildable. `FlareDataset`/`make_loaders` use it by default (`use_cache=True`, `paths.cache_dir`, `loader.use_cache`) and fail loudly if it is missing or lacks a frame; the memmap is opened lazily per worker.
- **Frames**: `data/processed/frames/<harpnum>/<YYYYMMDD_HHMM>.npy`, one file per frame, float16 128×128 (read-only for training).
  `data/processed/manifest.csv` is append-only and still holds earlier `failed` rows; the `.npy` files are the truth.
- **Loader** (`flarecast.dataset.flare_dataset`): `FlareDataset(samples, frames_dir, mode)` with mode `static` (frame t →
  [1,128,128]), `single` ([31,1,128,128]) or `dual` (coarse [16,1,128,128] = frames[0::2], fine [16,1,128,128] = frames[15:]).
  `make_loaders(mode, batch_size, num_workers, *, samples, frames_dir, seed=0)` → dict of train/val/test `DataLoader`s.
  `class_weights(samples)` → w_c = N/(K·N_c) from the train split only (E5). Helpers: `load_samples`, `filter_available`.
- **Models** (`flarecast.models.models`): `build_model("E1"…"E5")` (E5 = E4 architecture; only the loss differs),
  `persistence_predict(persistence_label)` for E0, `count_params`. Params: E1 249,411; E2 381,507; E3 521,795; E4 538,435.
- **Training config**: **batch 8, 128 px**. Memory test (`outputs/models/memory_test.json`, RTX 3050 4 GB, 3,304 MB free):
  batch 16 at 128 px peaks ≈ 3.0–3.1 GB for E2/E3, batch 8 ≈ 1.5 GB. Batch 8 for **all** models is approved and recorded
  (decisions.md §9, Change Log 12; `loader.batch_size: 8`). Do not use a different batch size per model.
- **Command style**: `conda run -n flare --no-capture-output python scripts/<name>.py` and
  `conda run -n flare --no-capture-output python -m pytest -q` (147 tests pass).
- **Training/eval** (Modules 6-7): runs in `outputs/runs/<model>_lr<lr>_s<seed>[_<tag>]/` (history.csv, best.pt, preds_<split>.csv, cost.json, DONE). `06_run_all.py` queues E1-E5 x lr {1e-3, 3e-4} x seed 0 and skips DONE runs; `07_evaluate.py [--tag T]` selects per model by val TSS>=M only and writes `outputs/results/` and `outputs/figures/`. Runs are not bit-reproducible (cudnn.benchmark + AMP). Without the cache E4 was data-bound (~510 s/epoch); with it ~61 s/epoch (num_workers 4 kept, same as 8).
- **Next**: start the full queue (user runs it), then Module 8 (Grad-CAM for E1-E5 on the same test samples, figures).

## Lessons learned / gotchas
- JSOC returns `QUALITY` as **hex strings** (`0x00000400`); query with drms `convert_numeric=False` and parse with `parse_quality`.
- JSOC allows only **one pending export per user**; do not run two export jobs at once.
- Record-set URLs with more than ~150 timestamps give **HTTP 414**; keep ≤ 120 records per request (`download.export_chunk_records`).
- `url_quick` / `as-is` files are **all named `Br.fits`**: derive local names from the record (`local_filename`), never trust the server name.
- HEK queries can hang or time out; `flares.hek_timeout_s` makes them fail fast and the index build retries a month once.
- 2023 SWPC flares mostly lack NOAA numbers, which is why those months are dropped (completeness rule).
- Windows: any multi-worker `DataLoader` script needs the `if __name__ == "__main__":` guard (spawn).
- The GPU has only **~3.3 GB free** of 4 GB; use `05_memory_test.py` numbers before changing batch size or image size.
- Long background jobs: never stop or restart them without the user's consent; report failures and ask before retrying.
