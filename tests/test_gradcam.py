"""Tests for the Grad-CAM module (flarecast.explain): target, CAM-to-patch mapping, case selection, no future frames.
Synthetic data and randomly initialised models, CPU only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from flarecast.dataset.flare_dataset import frame_filename  # noqa: E402
from flarecast.explain.cases import CASE_SPEC, select_cases  # noqa: E402
from flarecast.explain.gradcam import (CAM_SIZE, cam_to_patch, check_no_future_frames, explain_cases,  # noqa: E402
                                       gradcam, log_p_ge_m)
from flarecast.models import build_model  # noqa: E402
from test_module4 import T0, make_row  # noqa: E402

MODELS = [f"E{i}" for i in range(1, 9)]


# ---------------------------------------------------------------- target
def test_target_is_log_p_ge_m_from_softmax():
    z = torch.randn(6, 3, dtype=torch.float64) * 3
    p = torch.softmax(z, dim=1)
    assert torch.allclose(log_p_ge_m(z), torch.log(p[:, 1] + p[:, 2]), atol=1e-12)
    assert torch.allclose(log_p_ge_m(z), torch.logsumexp(z[:, 1:], 1) - torch.logsumexp(z, 1))
    with pytest.raises(ValueError):
        log_p_ge_m(torch.zeros(2, 4))


# ---------------------------------------------------------------- CAM -> original patch
@pytest.mark.parametrize("hw", [(60, 100), (100, 60), (377, 745), (128, 128), (31, 17)])
def test_cam_shape_equals_original_patch(hw):
    cam = np.random.default_rng(0).random((CAM_SIZE, CAM_SIZE)).astype(np.float32)
    out = cam_to_patch(cam, hw)
    assert out.shape == hw and out.min() >= 0 and out.max() <= 1


def test_cam_padding_removed_and_zero_cam():
    cam = np.zeros((CAM_SIZE, CAM_SIZE), dtype=np.float32)
    cam[-1, :] = 1.0                                   # hot bottom row of the padded square
    h, w = 60, 100                                     # padded side 100, 20 rows of padding top and bottom
    out = cam_to_patch(cam, (h, w))
    full = F.interpolate(torch.from_numpy(cam)[None, None], size=(100, 100), mode="bilinear", align_corners=False)[0, 0].numpy()
    assert np.allclose(out, np.clip(full[20:80, :], 0, 1))      # crop = same offsets as pad_to_square
    assert out.max() < full.max()                               # the hottest rows lie in the removed padding
    assert (cam_to_patch(np.zeros((CAM_SIZE, CAM_SIZE)), (h, w)) == 0).all()


# ---------------------------------------------------------------- Grad-CAM on every model
@pytest.mark.parametrize("name", MODELS)
def test_gradcam_every_model(name):
    torch.manual_seed(0)
    model = build_model(name).eval()
    g = torch.Generator().manual_seed(1)
    if model.input_mode == "static":
        x = torch.rand(1, 128, 128, generator=g) * 2 - 1
    elif model.input_mode == "single":
        x = torch.rand(31, 1, 128, 128, generator=g) * 2 - 1
    else:
        coarse, fine = (torch.rand(16, 1, 128, 128, generator=g) * 2 - 1, torch.rand(16, 1, 128, 128, generator=g) * 2 - 1)
        fine[-1] = coarse[-1]                      # both branches end at the same frame t
        x = (coarse, fine)
    r = gradcam(model, x)
    with torch.no_grad():
        xb = tuple(t[None] for t in x) if isinstance(x, tuple) else x[None]
        p = torch.softmax(model(*xb) if isinstance(xb, tuple) else model(xb), dim=1)[0]
    assert r["cam"].shape == (CAM_SIZE, CAM_SIZE) and (r["cam"] >= 0).all()
    assert r["target"] == pytest.approx(float(torch.log(p[1] + p[2])), abs=1e-5)
    assert r["p_ge_m"] == pytest.approx(float(p[1] + p[2]), abs=1e-5)
    if name in ("E4", "E5", "E6"):
        for br in ("coarse", "fine"):
            assert r["attention"][br].shape == (16,) and r["attention"][br].sum() == pytest.approx(1.0, abs=1e-5)
    else:
        assert r["attention"] is None


# ---------------------------------------------------------------- no future information
def test_check_no_future_frames():
    ids = make_row(1, T0, 0, "test")["frame_ids"]
    check_no_future_frames(ids, T0)                                    # ok: all <= t, last == t
    later = ids[:-1] + [(T0 + pd.Timedelta(minutes=24)).strftime("%Y%m%dT%H%M")]
    with pytest.raises(ValueError, match="after t"):
        check_no_future_frames(later, T0)
    with pytest.raises(ValueError, match="not t"):
        check_no_future_frames(ids[:-1], T0)


# ---------------------------------------------------------------- same cases for all models
def _preds(n_per=8):
    rng = np.random.default_rng(0)
    rows = []
    for h in range(1, 41):
        label = 2 if h <= 3 else (1 if h <= 12 else 0)
        s = float(rng.random())
        p2 = s * (0.7 if label == 2 else 0.2)
        rows.append({"harpnum": h, "t": T0, "label": label, "p0": 1 - s, "p1": s - p2, "p2": p2})
    return pd.DataFrame(rows)


def test_select_cases_rules():
    d, tau1 = _preds(), 0.5
    c = select_cases(d, tau1)
    assert list(c["category"]) == ["TP-X", "TP-M", "FN", "FN", "FA", "FA", "TN"] and len(c) == sum(n for _, n in CASE_SPEC)
    assert c["harpnum"].is_unique
    s = d.set_index("harpnum")["p1"] + d.set_index("harpnum")["p2"]
    lab = d.set_index("harpnum")["label"]
    for r in c.itertuples():
        assert r.e8_p_ge_m == pytest.approx(s[r.harpnum]) and r.e8_tau1 == tau1
        pos, hit = lab[r.harpnum] >= 1, s[r.harpnum] >= tau1
        assert {"TP-X": pos and hit and lab[r.harpnum] == 2, "TP-M": pos and hit and lab[r.harpnum] == 1,
                "FN": pos and not hit, "FA": (not pos) and hit, "TN": (not pos) and not hit}[r.category]
    pd.testing.assert_frame_equal(c, select_cases(d.sample(frac=1, random_state=3), tau1))   # order-independent
    with pytest.raises(ValueError, match="Only"):
        select_cases(d[d["label"] < 2], tau1)                  # no X in the pool -> fail loudly, no silent substitute


def test_explain_cases_same_cases_for_all_models(tmp_path):
    rng = np.random.default_rng(0)
    rows = [make_row(h, T0, lab, "test") for h, lab in ((1, 2), (2, 0), (3, 1))]
    for r in rows:
        d = tmp_path / str(r["harpnum"])
        d.mkdir()
        for fid in r["frame_ids"]:
            np.save(d / frame_filename(fid), (rng.random((128, 128)) * 2 - 1).astype(np.float16))
    samples = pd.DataFrame(rows)
    cases = pd.DataFrame({"case": ["01_TP-X", "02_TN", "03_TP-M"], "harpnum": [1, 2, 3], "t": [T0] * 3,
                          "true_class": ["X", "<M", "M"]})
    torch.manual_seed(0)
    models = {m: build_model(m).eval() for m in ("E1", "E2", "E4", "E8")}
    res = explain_cases(models, cases, samples, tmp_path, use_cache=False)
    assert set(res) == {(m, c) for m in models for c in cases["case"]}           # every model on every case
    # a case that does not match the sample (wrong true class) must fail loudly
    bad = cases.copy()
    bad.loc[0, "true_class"] = "M"
    with pytest.raises(ValueError, match="does not match"):
        explain_cases(models, bad, samples, tmp_path, use_cache=False)
    # a sample whose window contains a frame after t must be refused
    fut = samples.copy()
    fut.at[0, "frame_ids"] = list(fut.at[0, "frame_ids"][:-1]) + [(T0 + pd.Timedelta(minutes=24)).strftime("%Y%m%dT%H%M")]
    with pytest.raises(ValueError, match="after t"):
        explain_cases(models, cases, fut, tmp_path, use_cache=False)
