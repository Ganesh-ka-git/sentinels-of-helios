# Solar Flare Forecasting: Dissertation Code

Short-term (6–24 h) active-region flare forecasting from SDO/HMI SHARP Br magnetograms. The design is in `decisions.md`,
and the instructions for Claude Code are in `CLAUDE.md`.

## Setup on Windows (PowerShell, once)
```powershell
cd path\to\solar-flare-forecasting
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
# 1) PyTorch with CUDA. Check your CUDA driver version with `nvidia-smi` and pick the matching index URL
#    (cu121, cu124 or cu126) from https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 2) Everything else
pip install -r requirements.txt
# 3) Check that the GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Then open `config/config.yaml` and replace `YOUR_JSOC_EMAIL` with the e-mail address you registered at JSOC.

## Module 1: feasibility pilot
```powershell
python -m pytest -q                       # unit tests (no internet needed); expect: 20 passed
python scripts/01_feasibility.py          # needs internet; roughly 5–20 min depending on JSOC
```
Outputs go to `outputs/feasibility/2024-05/`:
- `report.json`
- `sharp_keywords.csv`
- `flares_associated.csv`
- `samples.csv`
- `frames_needed.csv`

The test FITS files go to `data/pilot_fits/`.
