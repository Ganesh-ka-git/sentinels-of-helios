"""Grad-CAM on the shared encoder's last conv block for models E1-E8 (decisions.md §11, Change Log 13).

Layer: output of block 4 (Conv-BN-ReLU) of ``model.encoder.features``, i.e. the [128,16,16] map just before the last
max-pool and the global average pool. Target (identical for every model): log P(>=M) = logsumexp(z1,z2) -
logsumexp(z0,z1,z2). The CAM is taken for the frame at t only: for the temporal models it is the last frame of every
sequence, and the gradient reaches it through the LSTM / attention. The dual-branch models call the shared encoder twice
(coarse and fine); frame t has the same activation in both calls, so the gradients of the two paths are summed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..models import build_model

CAM_LAYER = 14        # features[12:16] = block 4 [Conv, BN, ReLU, MaxPool]; index 14 = its ReLU (before pooling)
CAM_SIZE = 16         # spatial size of the block-4 map for a 128 x 128 input


def log_p_ge_m(logits: torch.Tensor) -> torch.Tensor:
    """log P(>=M) = logsumexp(z1,z2) - logsumexp(z0,z1,z2) for logits [B,3] -> [B]."""
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"expected logits [B,3], got {tuple(logits.shape)}")
    return torch.logsumexp(logits[:, 1:], dim=1) - torch.logsumexp(logits, dim=1)


def parse_frame_time(frame_id: str) -> pd.Timestamp:
    """'20240101T0024' -> Timestamp."""
    return pd.Timestamp(f"{frame_id[:4]}-{frame_id[4:6]}-{frame_id[6:8]} {frame_id[9:11]}:{frame_id[11:13]}")


def check_no_future_frames(frame_ids: list[str], t: pd.Timestamp) -> None:
    """Raise if any input frame is later than t, or the last frame is not t."""
    times = [parse_frame_time(f) for f in frame_ids]
    if max(times) > pd.Timestamp(t):
        raise ValueError(f"input window contains a frame after t={t}: {max(times)}")
    if times[-1] != pd.Timestamp(t):
        raise ValueError(f"last frame {times[-1]} is not t={t}")


def load_model(name: str, ckpt_path, device: str = "cpu") -> torch.nn.Module:
    """Build model ``name`` and load ``best.pt['model']`` (eval mode, fp32)."""
    model = build_model(name).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def _batch(x, device: str):
    """Add a batch dimension of 1 to one sample's inputs (tensor or (coarse, fine))."""
    if isinstance(x, (tuple, list)):
        return tuple(t.unsqueeze(0).to(device) for t in x)
    return x.unsqueeze(0).to(device)


def gradcam(model: torch.nn.Module, x, device: str = "cpu") -> dict:
    """Grad-CAM of the frame at t for ONE sample ``x`` (as returned by FlareDataset, no batch dimension).

    Returns {"cam": [16,16] float32 (ReLU'd, un-normalised), "target": log P(>=M), "p_ge_m", "logits" [3],
    "attention": {"coarse": [16], "fine": [16]} for the attention models else None}.
    """
    model.eval()
    layer = model.encoder.features[CAM_LAYER]
    acts: list[torch.Tensor] = []
    grads: dict[int, torch.Tensor] = {}

    def fwd_hook(_m, _inp, out):
        k = len(acts)
        acts.append(out.detach())
        out.register_hook(lambda g, k=k: grads.__setitem__(k, g.detach()))

    h = layer.register_forward_hook(fwd_hook)
    try:
        xb = _batch(x, device)
        with torch.enable_grad():
            attn_model = hasattr(model, "attn_coarse")
            out = model(*xb, return_attention=True) if attn_model else (model(*xb) if isinstance(xb, tuple) else model(xb))
            logits, attn = out if attn_model else (out, None)
            target = log_p_ge_m(logits)[0]
            model.zero_grad(set_to_none=True)
            target.backward()
    finally:
        h.remove()
    if not acts or len(grads) != len(acts):
        raise RuntimeError(f"hook captured {len(acts)} activations and {len(grads)} gradients")
    mode = model.input_mode
    n_calls = {"static": 1, "single": 1, "dual": 2}[mode]
    if len(acts) != n_calls:
        raise RuntimeError(f"expected {n_calls} encoder call(s) for mode {mode!r}, got {len(acts)}")
    a_t = acts[0][-1]                                         # activation of frame t (last frame of the first call)
    for a in acts[1:]:
        if not torch.allclose(a[-1], a_t, atol=1e-5):
            raise RuntimeError("frame t has different activations in the two encoder calls")
    g_t = sum(g[-1] for g in grads.values())                  # total gradient wrt the activation of frame t
    if a_t.shape != (128, CAM_SIZE, CAM_SIZE):
        raise RuntimeError(f"unexpected activation shape {tuple(a_t.shape)}")
    weights = g_t.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * a_t).sum(0)).cpu().numpy().astype(np.float32)
    return {"cam": cam, "target": float(target.item()), "p_ge_m": float(torch.exp(target).item()),
            "logits": logits[0].detach().cpu().numpy(),
            "attention": None if attn is None else {k: v[0].detach().cpu().numpy() for k, v in attn.items()}}


def cam_to_patch(cam: np.ndarray, patch_hw: tuple[int, int]) -> np.ndarray:
    """Map a [16,16] CAM back to the original (h, w) patch: normalise by the CAM maximum (0 stays 0), bilinear upsample to
    the padded square (side max(h, w)), then remove the zero padding with the same offsets as ``pad_to_square``."""
    h, w = int(patch_hw[0]), int(patch_hw[1])
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or cam.shape[0] != cam.shape[1]:
        raise ValueError(f"expected a square 2-D CAM, got {cam.shape}")
    m = float(cam.max())
    cam = cam / m if m > 0 else cam
    s = max(h, w)
    up = F.interpolate(torch.from_numpy(cam)[None, None], size=(s, s), mode="bilinear", align_corners=False)[0, 0].numpy()
    top, left = (s - h) // 2, (s - w) // 2
    out = np.clip(up[top:top + h, left:left + w], 0.0, 1.0)
    if out.shape != (h, w):
        raise AssertionError(f"CAM patch shape {out.shape} != patch shape {(h, w)}")
    return out


def explain_cases(models: dict[str, torch.nn.Module], cases: pd.DataFrame, samples: pd.DataFrame, frames_dir,
                  use_cache: bool = True, cache_dir=None, device: str = "cpu") -> dict[tuple[str, str], dict]:
    """Grad-CAM of every model on EVERY case (same cases for all models). ``samples`` = the sample index
    (``load_samples``); each case is looked up by (harpnum, t) and must match exactly one row. The input window of every
    case is checked to contain no frame after t. Returns {(model, case_id): gradcam() result + meta}."""
    from ..dataset import FlareDataset

    names = {"<M": 0, "M": 1, "X": 2}
    out: dict[tuple[str, str], dict] = {}
    for c in cases.itertuples():
        t = pd.Timestamp(c.t)
        row = samples[(samples["harpnum"] == c.harpnum) & (samples["t"] == t)]
        if len(row) != 1:
            raise ValueError(f"case {c.case}: {len(row)} sample rows for HARP {c.harpnum} t={t}")
        check_no_future_frames(list(row.iloc[0]["frame_ids"]), t)
        for name, model in models.items():
            ds = FlareDataset(row, frames_dir, model.input_mode, use_cache, cache_dir)
            x, y, meta = ds[0]
            if int(y) != names[c.true_class] or meta["harpnum"] != c.harpnum or meta["t"] != t.strftime("%Y-%m-%d %H:%M"):
                raise ValueError(f"case {c.case}: dataset sample does not match the case list")
            out[(name, c.case)] = gradcam(model, x, device)
    return out
