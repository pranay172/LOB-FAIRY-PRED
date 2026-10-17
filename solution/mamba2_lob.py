import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAW_DIM = 32
ENGINEERED_DIM = 69


@dataclass
class ModelConfig:
    raw_dim: int = RAW_DIM
    input_dim: int = ENGINEERED_DIM
    d_model: int = 128
    n_layers: int = 4
    ff_mult: int = 2
    dropout: float = 0.1


class Mamba2SSMBlock(nn.Module):
    """A lightweight Mamba2-style state-space block with streaming step support."""

    def __init__(self, d_model: int, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.a_log = nn.Parameter(torch.zeros(d_model))
        self.d_skip = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

        ff_hidden = ff_mult * d_model
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout),
        )

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

    def _recurrent_step(self, z_t: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_t, g_t, dt_t = self.in_proj(z_t).chunk(3, dim=-1)
        dt_t = F.softplus(dt_t) + 1e-4

        a = torch.exp(-dt_t * torch.exp(self.a_log).unsqueeze(0))
        new_state = a * state + (1.0 - a) * u_t

        y_t = torch.sigmoid(g_t) * new_state + self.d_skip.unsqueeze(0) * z_t
        return y_t, new_state

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        bsz, tlen, _ = x.shape
        z = self.norm(x)

        h = state if state is not None else self._init_state(bsz, x.device, x.dtype)
        outputs = []
        for t in range(tlen):
            y_t, h = self._recurrent_step(z[:, t, :], h)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        x = x + self.out_proj(y)
        x = x + self.ff(self.ff_norm(x))
        return x, h

    def step(self, x_t: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_t: (B, D)
        z_t = self.norm(x_t)
        h = state if state is not None else self._init_state(x_t.shape[0], x_t.device, x_t.dtype)
        y_t, h = self._recurrent_step(z_t, h)
        x_t = x_t + self.out_proj(y_t)
        x_t = x_t + self.ff(self.ff_norm(x_t))
        return x_t, h


class RegressionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Mamba2LOBModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.input_proj = nn.Linear(self.config.input_dim, self.config.d_model)
        self.blocks = nn.ModuleList(
            [
                Mamba2SSMBlock(
                    d_model=self.config.d_model,
                    ff_mult=self.config.ff_mult,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.head_t0 = RegressionHead(self.config.d_model)
        self.head_t1 = RegressionHead(self.config.d_model)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        x = self.final_norm(x)
        t0 = self.head_t0(x).squeeze(-1)
        t1 = self.head_t1(x).squeeze(-1)
        return torch.stack([t0, t1], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        states: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # features: (B, T, F)
        x = self.input_proj(features)

        out_states: List[torch.Tensor] = []
        for ix, block in enumerate(self.blocks):
            block_state = None if states is None else states[ix]
            x, new_state = block(x, block_state)
            out_states.append(new_state)

        pred = self._heads(x[:, -1, :])
        return pred, out_states

    def step(
        self,
        features_t: torch.Tensor,
        states: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # features_t: (B, F)
        x_t = self.input_proj(features_t)

        out_states: List[torch.Tensor] = []
        for ix, block in enumerate(self.blocks):
            block_state = None if states is None else states[ix]
            x_t, new_state = block.step(x_t, block_state)
            out_states.append(new_state)

        pred = self._heads(x_t)
        return pred, out_states


def target_transform(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.log1p(np.abs(y))


def inverse_target_transform(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * (np.expm1(np.abs(y)))


def weighted_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target: (B, 2)
    pred = torch.clamp(pred, -6.0, 6.0)
    w = torch.abs(target) + eps

    w_sum = torch.sum(w, dim=0, keepdim=True)
    mu_p = torch.sum(w * pred, dim=0, keepdim=True) / w_sum
    mu_t = torch.sum(w * target, dim=0, keepdim=True) / w_sum

    dp = pred - mu_p
    dt = target - mu_t

    cov = torch.sum(w * dp * dt, dim=0) / torch.clamp(w_sum.squeeze(0), min=eps)
    var_p = torch.sum(w * dp * dp, dim=0) / torch.clamp(w_sum.squeeze(0), min=eps)
    var_t = torch.sum(w * dt * dt, dim=0) / torch.clamp(w_sum.squeeze(0), min=eps)

    corr = cov / (torch.sqrt(var_p * var_t) + eps)
    return -torch.mean(corr)


def build_engineered_features_torch(raw: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """
    raw:   (B, T, 32)
    steps: (B, T)
    returns: (B, T, 69)
    """
    delta = raw - torch.roll(raw, shifts=1, dims=1)
    delta[:, 0, :] = 0.0

    bid_vol = raw[:, :, 12:18].sum(dim=-1, keepdim=True)
    ask_vol = raw[:, :, 18:24].sum(dim=-1, keepdim=True)
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
    spread = (raw[:, :, 6:7] - raw[:, :, 0:1])
    total_vol = bid_vol + ask_vol

    s = steps.float().unsqueeze(-1)
    sin_pos = torch.sin(s / 1000.0)
    cos_pos = torch.cos(s / 1000.0)

    return torch.cat([raw, delta, imbalance, spread, total_vol, sin_pos, cos_pos], dim=-1)


def build_engineered_features_numpy(
    raw: np.ndarray,
    delta: np.ndarray,
    step_in_seq: int,
) -> np.ndarray:
    """
    raw:   (32,)
    delta: (32,)
    returns: (69,)
    """
    bid_vol = float(np.sum(raw[12:18]))
    ask_vol = float(np.sum(raw[18:24]))
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
    spread = float(raw[6] - raw[0])
    total_vol = bid_vol + ask_vol

    s = float(step_in_seq)
    sin_pos = math.sin(s / 1000.0)
    cos_pos = math.cos(s / 1000.0)

    extra = np.asarray([imbalance, spread, total_vol, sin_pos, cos_pos], dtype=np.float32)
    return np.concatenate([raw.astype(np.float32), delta.astype(np.float32), extra], axis=0)
