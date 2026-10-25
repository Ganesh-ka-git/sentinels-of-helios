"""Models E0-E5 exactly as decisions.md §8.

Input conventions (match flarecast.dataset.FlareDataset):
    E1  StaticCNN       x [B,1,H,W]                (frame t)
    E2  SingleLSTM      x [B,31,1,H,W]             (oldest -> newest)
    E3  DualLSTM        coarse [B,16,1,H,W], fine [B,16,1,H,W]
    E4  DualLSTMAttn    same as E3; ``return_attention=True`` also returns the attention weights
    E5  = E4 architecture; it differs only in the (class-weighted) loss, which the training code handles.
    E6/E7/E8 = E4/E2/E1 architecture with focal loss (exploratory, Change Log 16); the loss is in the training code.
All models return logits [B,3]. E0 (persistence) is :func:`persistence_predict`.
"""
from __future__ import annotations

import torch
from torch import nn

N_CLASSES = 3
FEAT = 128
SEQ_LEN = {"single": 31, "coarse": 16, "fine": 16}


class SharedEncoder(nn.Module):
    """4 x [Conv3x3(pad 1) -> BatchNorm -> ReLU -> MaxPool2], channels 32/64/128/128, global average pool -> 128-d."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 128)):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N,1,H,W] -> [N,128]"""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"encoder expects [N,1,H,W], got {tuple(x.shape)}")
        return self.pool(self.features(x)).flatten(1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,1,H,W] -> [B,T,128]; every frame goes through the same weights (reshaped to [B*T,1,H,W])."""
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError(f"expected [B,T,1,H,W], got {tuple(x.shape)}")
        b, t = x.shape[:2]
        return self(x.reshape(b * t, *x.shape[2:])).reshape(b, t, -1)


def make_head(in_dim: int) -> nn.Sequential:
    """Classifier head shared by all models: Linear(in,64) -> ReLU -> Dropout(0.3) -> Linear(64,3)."""
    return nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, N_CLASSES))


def _check_seq(x: torch.Tensor, t: int, name: str) -> None:
    if x.ndim != 5 or x.shape[1] != t or x.shape[2] != 1:
        raise ValueError(f"{name}: expected [B,{t},1,H,W], got {tuple(x.shape)}")


class StaticCNN(nn.Module):
    """E1: encoder(frame t) -> head. Only ever sees one frame."""
    input_mode = "static"

    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.head = make_head(FEAT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"E1 takes a single frame [B,1,H,W], got {tuple(x.shape)}")
        return self.head(self.encoder(x))


class SingleLSTM(nn.Module):
    """E2: encoder per frame (31) -> LSTM(128, 1 layer) -> last hidden state -> head."""
    input_mode = "single"

    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.lstm = nn.LSTM(FEAT, FEAT, num_layers=1, batch_first=True)
        self.head = make_head(FEAT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_seq(x, SEQ_LEN["single"], "E2")
        _, (h, _) = self.lstm(self.encoder.encode_sequence(x))
        return self.head(h[-1])


class AdditiveAttention(nn.Module):
    """e_t = v^T tanh(W h_t), alpha = softmax over time, c = sum_t alpha_t h_t  (W: 128->64, v: 64->1)."""

    def __init__(self, dim: int = FEAT, hidden: int = 64):
        super().__init__()
        self.W = nn.Linear(dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [B,T,D] -> (context [B,D], alpha [B,T])"""
        e = self.v(torch.tanh(self.W(h))).squeeze(-1)
        alpha = torch.softmax(e.float(), dim=1).to(h.dtype)  # softmax in fp32 for AMP stability
        return (alpha.unsqueeze(-1) * h).sum(dim=1), alpha


class DualLSTM(nn.Module):
    """E3: ONE shared encoder for both branches; an LSTM(128) per branch; concat of last hiddens (256) -> head."""
    input_mode = "dual"

    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()                       # single instance, used by coarse and fine
        self.lstm_coarse = nn.LSTM(FEAT, FEAT, num_layers=1, batch_first=True)
        self.lstm_fine = nn.LSTM(FEAT, FEAT, num_layers=1, batch_first=True)
        self.head = make_head(2 * FEAT)

    def forward(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        _check_seq(coarse, SEQ_LEN["coarse"], "E3 coarse")
        _check_seq(fine, SEQ_LEN["fine"], "E3 fine")
        _, (hc, _) = self.lstm_coarse(self.encoder.encode_sequence(coarse))
        _, (hf, _) = self.lstm_fine(self.encoder.encode_sequence(fine))
        return self.head(torch.cat([hc[-1], hf[-1]], dim=1))


class DualLSTMAttn(nn.Module):
    """E4 (and E5): E3 with additive attention pooling over each branch's LSTM outputs instead of the last hidden."""
    input_mode = "dual"

    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.lstm_coarse = nn.LSTM(FEAT, FEAT, num_layers=1, batch_first=True)
        self.lstm_fine = nn.LSTM(FEAT, FEAT, num_layers=1, batch_first=True)
        self.attn_coarse = AdditiveAttention()
        self.attn_fine = AdditiveAttention()
        self.head = make_head(2 * FEAT)

    def forward(self, coarse: torch.Tensor, fine: torch.Tensor, return_attention: bool = False):
        _check_seq(coarse, SEQ_LEN["coarse"], "E4 coarse")
        _check_seq(fine, SEQ_LEN["fine"], "E4 fine")
        oc, _ = self.lstm_coarse(self.encoder.encode_sequence(coarse))   # [B,16,128]
        of, _ = self.lstm_fine(self.encoder.encode_sequence(fine))
        cc, a_c = self.attn_coarse(oc)
        cf, a_f = self.attn_fine(of)
        logits = self.head(torch.cat([cc, cf], dim=1))
        return (logits, {"coarse": a_c, "fine": a_f}) if return_attention else logits


def persistence_predict(persistence_label) -> torch.Tensor:
    """E0: predict the max associated class in [t-24h, t), i.e. the sample's ``persistence_label``. No learning."""
    y = torch.as_tensor(persistence_label)
    if y.dtype.is_floating_point:
        if not torch.equal(y, y.round()):
            raise ValueError("persistence labels must be integers")
    y = y.long()
    if ((y < 0) | (y >= N_CLASSES)).any():
        raise ValueError(f"persistence labels must be in 0..{N_CLASSES - 1}")
    return y


# E5 = E4 architecture + class-weighted CE; E6 = E4 + focal; E7 = E2 + focal; E8 = E1 + focal (Change Logs 15-16).
_MODELS = {"E1": StaticCNN, "E2": SingleLSTM, "E3": DualLSTM, "E4": DualLSTMAttn, "E5": DualLSTMAttn,
           "E6": DualLSTMAttn, "E7": SingleLSTM, "E8": StaticCNN}

# Input layout per model, derived from each class's ``input_mode`` (the single source for loaders, training and cost tests).
MODE_OF = {name: cls.input_mode for name, cls in _MODELS.items()}


def build_model(name: str) -> nn.Module:
    """Factory for the trainable models E1-E8 (E5 = E4, E6 = E4, E7 = E2, E8 = E1 architecture; only the loss differs).
    E0 is not a model."""
    key = str(name).upper()
    if key not in _MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(_MODELS)} "
                         f"(E0 persistence is persistence_predict, not a trainable model)")
    return _MODELS[key]()


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
