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
   Approved exploratory additions: focal-loss models E6–E8 (Change Log 16) and the post-hoc tuned-threshold/AUC evaluation (Change Log 15).
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
  explain/                gradcam.py (Grad-CAM, cam_to_patch, explain_cases), cases.py (case selection), figures.py
scripts/                  numbered entry points (01_feasibility.py ... 07_evaluate.py, 04b_build_cache.py, 06b_queue_seeds.py, watch_*.py)
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
| 6 | Training runner (`06_train.py`, `06_run_all.py`, `train/runner.py`) | **done**; seed-0 queue finished (10 runs DONE); `tests/test_module6_7.py` |
| 6b | Seeds 1 and 2 for E1–E5 (`06b_queue_seeds.py`) | superseded by 6c (6c covers E1–E8); `tests/test_multiseed.py` (8) |
| 7 | Evaluation (`07_evaluate.py`, `evaluate/`: metrics, cluster bootstrap, tables, figures) | **done**; multi-seed path run on real output (seeds 0–2, E0–E8) |
| 6c | Focal models E6–E8 (`loss=focal`, gamma 2, alpha = inv. sqrt freq, train only) + queue `06c_queue_focal.py` (stage 1 E6–E8 x 2 lr x seed 0; stage 2 seeds 1, 2 for E1–E8 at the seed-0 lr) | **done: all 24 selected runs (E1–E8 x seeds 0,1,2) DONE; no further training**; `tests/test_posthoc.py` |
| 7b | Post-hoc evaluation (`evaluate/thresholds.py`, `07_evaluate.py`): argmax + two-stage thresholds tuned on val only (tau2 by HSS_X, Change Log 17) + HSS variant + ROC/PR-AUC, pooled multi-seed bootstrap (E1vsE2…E8vsE1, E7vsE2, E6vsE4, E6vsE5, each vs E0), X-flag table, ROC/PR and threshold figures | **done and committed (435049d); results locked** |
| 8 | Grad-CAM E1–E8 (`08_gradcam.py select / download / figures`, `explain/`): 7 test cases from E8 seed 0 at its tuned tau1 (`outputs/explain/cases.csv`: TP-X, TP-M, 2 FN, 2 FA, TN), full-resolution frames in `data/figure_fits/`, figures + `figure_index.md` in `outputs/figures/gradcam/`, attention plots E4/E5/E6; `tests/test_gradcam.py` (18) | **done, committed (0fc0afd)**; thesis tables/figures assembly and thesis text remain |

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
  `make_loaders(mode, batch_size, num_workers, *, samples, frames_dir, seed=0, worker_init_fn=None, use_cache=True, cache_dir=None)` → dict of train/val/test `DataLoader`s.
  `tests/test_module4.py` wraps both with `use_cache=False` (it tests the per-frame path); cache equivalence is in `tests/test_cache.py`.
  `class_weights(samples)` → w_c = N/(K·N_c) from the train split only (E5). Helpers: `load_samples`, `filter_available`.
- **Models** (`flarecast.models.models`): `build_model("E1"…"E8")` (E5 = E4 architecture with class-weighted CE; E6/E7/E8 = E4/E2/E1 architecture
  with focal loss; only the loss differs), `MODE_OF` (single model→input-layout mapping, used by runner and cost tests),
  `persistence_predict(persistence_label)` for E0, `count_params`. Params: E1 249,411; E2 381,507; E3 521,795; E4 538,435.
- **Training config**: **batch 8, 128 px**. Memory test (`outputs/models/memory_test.json`, RTX 3050 4 GB, 3,304 MB free):
  batch 16 at 128 px peaks ≈ 3.0–3.1 GB for E2/E3, batch 8 ≈ 1.5 GB. Batch 8 for **all** models is approved and recorded
  (decisions.md §9, Change Log 12; `loader.batch_size: 8`). Do not use a different batch size per model.
- **Command style**: `conda run -n flare --no-capture-output python scripts/<name>.py` and
  `conda run -n flare --no-capture-output python -m pytest -q` (192 tests pass; `test_train_run_end_to_end_cpu_and_skip` needs the GPU visible).
- **Training/eval** (Modules 6-7): runs in `outputs/runs/<model>_lr<lr>_s<seed>[_<tag>]/` (history.csv incl. `data_wait_s`, best.pt
  = `{"model": state_dict, epoch, ...}`, preds_<split>.csv, cost.json, run.json, DONE). Tagged runs (`smoke`, `cachetest_w*`) are ignored by
  the default evaluation. `06_run_all.py [--num-workers N]` queues E1-E5 x lr {1e-3, 3e-4} x seed 0 and skips DONE runs (one process
  per run; a failed run is reported, never retried). `07_evaluate.py [--tag T] [--seed S] [--seeds 0 1 2]` selects per model by
  **val TSS>=M only** (lr chosen on `--seed`, reused for other seeds), E0 comes from `persistence_label` in the preds files, and
  writes `outputs/results/` (CSVs + `results_summary.md`; `results/<tag>/` for tagged runs) and `outputs/figures/` (300 dpi PNG+PDF:
  per-run loss curves, val TSS per epoch, confusion matrices val+test). Bootstrap = paired cluster bootstrap over test HARPs
  (2,000 resamples, seed 0, difference = second - first, "supported" only if CI lower bound > 0; recall X "descriptive only" while
  test X < 10). Multi-seed verdicts additionally require all per-seed estimates to have the same sign.
- **Speed**: epoch time was data-bound until the frame cache (E4 ~510 s/epoch from per-frame files, cold reads ~600 files/s;
  ~61 s/epoch with the cache, `num_workers` 0/4/8 = 66/61/61 s, 4 kept). A 30-epoch dual-branch run is <= ~30 min.
- **Reproducibility**: runs are NOT bit-reproducible (`cudnn.benchmark=True` + AMP; identical E1 smoke runs gave val TSS 0.189 vs 0.125).
  Treat single-seed differences with caution.
- **Seed-0 results** (full queue, selected lr per model; test n=856, 63 HARPs, 7 X; from `outputs/results/results_summary.md`):
  test TSS>=M E0 0.405, E1 0.479 (lr 3e-4), E2 0.370 (3e-4), E3 0.525 (3e-4), E4 0.508 (1e-3), E5 0.523 (3e-4); recall M E0 0.338, E1 0.541,
  E2 0.432, E3 0.622, E4 0.622, E5 0.865 (precision >=M falls 0.35 -> 0.21, HSS 0.375 -> 0.213); recall X = 0 for E1-E5 (E0: 2/7, descriptive).
  Bootstrap (95% CI of test TSS>=M difference): E2-E1 -0.109 [-0.179, -0.001] (reverse, H1 not supported); E3-E2 +0.155 [0.019, 0.287]
  (supported); E4-E3 -0.017 [-0.091, 0.071] and E5-E4 +0.015 [-0.133, 0.189] (not supported); no model differs from E0 (all CIs include 0);
  E5-E4 recall M +0.243 [0.089, 0.411] (supported). Train TSS 0.67-0.77 vs test 0.37-0.53 (overfits quickly; best epochs 3-11).
  These single-seed argmax verdicts are superseded by the multi-seed results below.
- **Final multi-seed results (LOCKED; seeds 0,1,2, E0–E8, lr fixed from seed-0 val argmax TSS>=M: 3e-4 for all except E4 1e-3; produced by
  `07_evaluate.py --seeds 0 1 2`, commit 435049d, `outputs/results/results_summary.md`)**. Test mean TSS>=M, argmax / tuned (tau1 on val TSS>=M):
  E0 0.405; E1 0.425/0.533; E2 0.356/0.484; E3 0.436/0.545; E4 0.333/0.558; E5 0.532/0.573; E6 0.536/0.525; E7 0.591/0.566; E8 0.569/0.571.
  ROC-AUC 0.837–0.868 for E1–E8 vs 0.702 for E0. Pooled verdicts (CI excludes 0 AND same sign in all 3 seeds), tuned TSS>=M: E2-E1 reverse
  (H1 not supported); E3-E2 supported; E4-E3, E5-E4, E8-E1, E6-E4, E6-E5 not supported; E7-E2 supported; vs E0 supported for E3, E4, E5, E7, E8
  (not E1, E2, E6). ROC-AUC: all models beat E0, E2<E1 reverse, nothing else differs; PR-AUC: E8-E1 and E7-E2 supported. Argmax (secondary):
  E2-E1 and E4-E3 reverse; E5-E4, E8-E1, E7-E2, E6-E4 supported; vs E0 only E7, E8. H4 (tuned TSS>=M) supported only for E7 vs E2 (E8 vs E1, E6 vs E4 not; E8 and E7 do gain on PR-AUC and argmax TSS); H5 reverse under
  tuned thresholds (E6 has lower HSS and precision than E5). X is descriptive only (7 test X): under tuned thresholds all 7 are flagged >=M for
  E3–E8 in every seed, but the number of X predictions is unstable across seeds (e.g. E3 137/7/178, E8 18/280/1), and argmax almost never
  predicts X; the tuned per-class/X numbers are not trustworthy. No further training; do not change the selection or thresholds.
- **Grad-CAM (Module 8, done)**: seed-0 selected runs of E1–E8; layer = block-4 ReLU of the shared encoder (16x16, before pooling), frame at t;
  target log P(>=M); dual models sum the gradients of both encoder calls; CAM mapped back to the original patch (upsample to padded square,
  crop padding). Cases chosen from E8 seed 0 test predictions at its validation-tuned tau1 = 0.62 (rules in `explain/cases.py`), reused for every
  model; each model's decision uses its own tuned tau1 (all 8 models decide identically on all 7 cases). Findings to report carefully: some CAMs are
  empty inside the patch (e.g. 03_FN E8, evidence only in the padding); attention weights of E4/E5/E6 are near uniform (1/16) except more weight on
  the oldest frames (checked on case 01 only). The earlier "X predicted as M" case was replaced by the X true-positive case (01_TP-X).
- **Next**: (1) assemble thesis figures/tables from `outputs/results/`, `outputs/figures/` and `outputs/figures/gradcam/`; (2) thesis text must state the TAI/UTC note (decisions.md §4), the
  catalogue-completeness rule, the batch-8 and cache deviations, and that X is descriptive only.

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
- `conda run ... python -c "<multi-line>"` fails (conda rejects newlines in arguments); write a script file (use the scratchpad) instead.
  Large Bash heredocs containing quotes/backslashes can also be rejected by the shell tool: write the file with the Write tool, then run it.
- Source files mix LF/CRLF; when patching with scripts read with `newline=""` and keep the original line ending.
- `Path.with_suffix` on names containing a dot (e.g. `loss_E1_lr0.0003_s0`) truncates the name; build `f"{base}.png"` instead.
- First batch of a persistent-worker `DataLoader` takes ~20 s on Windows (spawn); exclude it when timing. A 20-minute foreground command
  is cut off by the tool timeout (10 min): run long jobs in the background.
- Cold per-frame reads are the training bottleneck; always train with the packed cache (`04b_build_cache.py`; rebuild if frames change).
- Bootstrap CIs on 63 test HARPs are wide (about +/-0.1-0.3 TSS), so most between-model differences are not detectable; do not over-interpret point estimates.
