"""Unit tests for Module 5 (models E0-E5). CPU, tiny images, no data."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flarecast.models import (DualLSTM, DualLSTMAttn, SharedEncoder, SingleLSTM, StaticCNN, build_model,  # noqa: E402
                              count_params, persistence_predict)

B, S = 3, 32  # batch, image size (the architecture is size-agnostic thanks to global average pooling)


def inputs(name, b=B, s=S):
    g = torch.Generator().manual_seed(0)
    r = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    return {"E1": (r(b, 1, s, s),), "E2": (r(b, 31, 1, s, s),),
            "E3": (r(b, 16, 1, s, s), r(b, 16, 1, s, s)), "E4": (r(b, 16, 1, s, s), r(b, 16, 1, s, s)),
            "E5": (r(b, 16, 1, s, s), r(b, 16, 1, s, s))}[name]


@pytest.mark.parametrize("name", ["E1", "E2", "E3", "E4", "E5"])
def test_forward_shape_and_finite(name):
    m = build_model(name).eval()
    out = m(*inputs(name))
    assert out.shape == (B, 3) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["E1", "E2", "E3", "E4"])
def test_backward_reaches_all_parameters(name):
    m = build_model(name).train()
    m(*inputs(name, b=4)).sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_encoder_architecture():
    enc = SharedEncoder()
    convs = [mod for mod in enc.modules() if isinstance(mod, torch.nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.padding) for c in convs] == \
        [(1, 32, (3, 3), (1, 1)), (32, 64, (3, 3), (1, 1)), (64, 128, (3, 3), (1, 1)), (128, 128, (3, 3), (1, 1))]
    assert sum(isinstance(mod, torch.nn.BatchNorm2d) for mod in enc.modules()) == 4
    assert sum(isinstance(mod, torch.nn.MaxPool2d) for mod in enc.modules()) == 4
    assert enc(torch.randn(2, 1, 64, 64)).shape == (2, 128) and enc(torch.randn(2, 1, 128, 128)).shape == (2, 128)
    with pytest.raises(ValueError):
        enc(torch.randn(2, 3, 64, 64))


def test_sequence_encoding_is_per_frame_with_same_weights():
    enc = SharedEncoder().eval()
    x = torch.randn(2, 5, 1, S, S)
    seq = enc.encode_sequence(x)
    assert seq.shape == (2, 5, 128)
    assert torch.allclose(seq[1, 3], enc(x[1, 3][None])[0], atol=1e-5)


def test_head_structure_is_identical_across_models():
    for name, in_dim in [("E1", 128), ("E2", 128), ("E3", 256), ("E4", 256), ("E5", 256)]:
        h = build_model(name).head
        lin = [m for m in h if isinstance(m, torch.nn.Linear)]
        assert [(l.in_features, l.out_features) for l in lin] == [(in_dim, 64), (64, 3)]
        assert [type(m).__name__ for m in h] == ["Linear", "ReLU", "Dropout", "Linear"] and h[2].p == 0.3


@pytest.mark.parametrize("cls", [DualLSTM, DualLSTMAttn])
def test_dual_models_use_one_shared_encoder_instance(cls):
    m = cls().eval()
    assert sum(isinstance(mod, SharedEncoder) for mod in m.modules()) == 1   # one encoder, not two copies
    calls = []
    m.encoder.register_forward_hook(lambda mod, i, o: calls.append(id(mod)))
    m(*inputs("E3"))
    assert len(calls) == 2 and len(set(calls)) == 1                          # coarse and fine pass through it
    # same frames in both branches -> identical features (same parameters)
    x = torch.randn(2, 16, 1, S, S)
    assert torch.equal(m.encoder.encode_sequence(x), m.encoder.encode_sequence(x.clone()))
    # parameter budget = ONE encoder + two LSTMs (+ attention) + head
    enc = count_params(m.encoder)
    lstms = count_params(m.lstm_coarse) + count_params(m.lstm_fine)
    attn = (count_params(m.attn_coarse) + count_params(m.attn_fine)) if cls is DualLSTMAttn else 0
    assert count_params(m) == enc + lstms + attn + count_params(m.head)
    assert m.lstm_coarse is not m.lstm_fine and m.lstm_coarse.weight_ih_l0 is not m.lstm_fine.weight_ih_l0


def test_attention_weights_sum_to_one_over_time():
    m = DualLSTMAttn().eval()
    logits, attn = m(*inputs("E4"), return_attention=True)
    assert logits.shape == (B, 3) and set(attn) == {"coarse", "fine"}
    for a in attn.values():
        assert a.shape == (B, 16) and (a >= 0).all()
        assert torch.allclose(a.sum(dim=1), torch.ones(B), atol=1e-5)
    assert torch.equal(logits, m(*inputs("E4")))  # flag does not change the prediction


def test_attention_formula():
    m = DualLSTMAttn()
    lin_w = m.attn_coarse.W
    assert (lin_w.in_features, lin_w.out_features) == (128, 64)
    assert (m.attn_coarse.v.in_features, m.attn_coarse.v.out_features, m.attn_coarse.v.bias) == (64, 1, None)
    h = torch.randn(2, 16, 128)
    ctx, a = m.attn_coarse(h)
    e = (torch.tanh(h @ lin_w.weight.T + lin_w.bias) @ m.attn_coarse.v.weight.T).squeeze(-1)
    assert torch.allclose(a, torch.softmax(e, 1), atol=1e-6)
    assert torch.allclose(ctx, (a.unsqueeze(-1) * h).sum(1), atol=1e-6)


def test_e1_sees_only_frame_t():
    m = StaticCNN().eval()
    seq = torch.randn(2, 31, 1, S, S)
    out_t = m(seq[:, -1])
    seq2 = seq.clone()
    seq2[:, :-1] = torch.randn_like(seq2[:, :-1])          # change every earlier frame
    assert torch.equal(m(seq2[:, -1]), out_t)
    with pytest.raises(ValueError, match="single frame"):
        m(seq)                                              # a sequence is rejected, not silently reduced


def test_sequence_models_reject_wrong_lengths():
    with pytest.raises(ValueError):
        SingleLSTM()(torch.randn(2, 16, 1, S, S))
    with pytest.raises(ValueError):
        DualLSTM()(torch.randn(2, 31, 1, S, S), torch.randn(2, 16, 1, S, S))


def test_e5_has_e4_architecture_but_is_a_separate_instance():
    e4, e5 = build_model("E4"), build_model("E5")
    assert type(e4) is type(e5) and e4 is not e5 and count_params(e4) == count_params(e5)
    assert [n for n, _ in e4.named_parameters()] == [n for n, _ in e5.named_parameters()]


def test_param_counts_positive_and_ordered():
    p = {n: count_params(build_model(n)) for n in ("E1", "E2", "E3", "E4")}
    assert all(v > 0 for v in p.values())
    assert p["E1"] < p["E2"] < p["E3"] < p["E4"]
    enc = 1 * 32 * 9 + 32 + 2 * 32  # first conv + BN
    assert count_params(SharedEncoder()) > enc


def test_build_model_rejects_unknown_names():
    for bad in ("E0", "E6", "resnet", "", "E"):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(bad)
    assert type(build_model("e1")) is StaticCNN  # case-insensitive


def test_persistence_predict():
    y = persistence_predict([0, 1, 2, 1])
    assert y.dtype == torch.long and y.tolist() == [0, 1, 2, 1]
    assert persistence_predict(torch.tensor([0.0, 2.0])).tolist() == [0, 2]
    for bad in ([3], [-1], [0.5]):
        with pytest.raises(ValueError):
            persistence_predict(bad)


def test_models_accept_dataset_batch_layout(tmp_path):
    """The model input layouts equal the DataLoader outputs of Module 4 (static / single / dual)."""
    assert StaticCNN.input_mode == "static" and SingleLSTM.input_mode == "single"
    assert DualLSTM.input_mode == DualLSTMAttn.input_mode == "dual"
