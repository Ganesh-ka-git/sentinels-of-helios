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
  (later) preprocess/, dataset/, models/, train/, evaluate/, explain/
scripts/                  numbered entry points (01_feasibility.py, 02_..., ...)
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
| 1 | Feasibility pilot (one month: keywords, flares, association, samples, download test) | written, awaiting run |
| 2 | Full dataset: sample index for the chosen period, splits | pending pilot |
| 3 | Selective download + preprocessing to per-HARP float16 arrays | pending |
| 4 | Dataset/DataLoader (E1, E2, dual-branch views) + GPU memory test | pending |
| 5 | Models E1–E5 (shared encoder) + E0 persistence | pending |
| 6 | Training runner (config-driven, AMP, early stopping on val TSS≥M, checkpoints) | pending |
| 7 | Evaluation (TSS, P/R/F1, HSS, cluster bootstrap CIs, cost) | pending |
| 8 | Grad-CAM + figures and tables for the thesis | pending |

Experiment priority if time is short: E1, E2, E3, then E4, E5, then Grad-CAM, then E1w.
