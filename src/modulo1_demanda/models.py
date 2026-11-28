"""Arquitecturas neuronales globales de pronóstico para el Módulo 1 (PyTorch, CPU).

Responsabilidad:
    - ``GlobalMLPForecaster``: aplana la ventana histórica (``lookback_window`` días) + vector de
      calendario futuro (``forecast_horizon`` días) + embedding de ruta y predice los H pasos en
      paralelo (direct multi-step).
    - ``GlobalGRUForecaster``: encoder GRU sobre los días históricos causales; el estado final se
      concatena con el embedding de ruta y con el calendario de cada día futuro h para proyectar
      la salida de cada paso del horizonte con una cabeza compartida.
    - Ambos modelos son globales (las 8 rutas juntas, ``nn.Embedding(num_routes, emb_dim)``).

Salidas por paso h (en espacio normalizado por ventana; ver ``nn_data.build_windows``):
    mu_h  = Softplus(z_mu_h) + 1e-4                 (pronóstico de conteo, PoissonNLL)
    q90_h = mu_h + Softplus(z_delta_h)              (cuantil P90, Pinball; garantiza q90 >= mu)

Pérdida: ``poisson_nll(y, mu) + lambda_pinball * pinball_P90(y, q90)``.
Nota: al entrenar sobre y/s (s = escala de la ventana), la NLL de Poisson actúa como
cuasi-verosimilitud: es proporcional a la NLL en conteos por muestra (salvo una constante), de modo
que el óptimo sigue siendo la media condicional y cada ruta/ventana pesa de forma comparable.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import nn

EPS_MU: float = 1e-4
_MU_BIAS_INIT: float = math.log(math.e - 1.0)  # softplus(b) = 1  -> mu inicial = escala de la ventana
_DELTA_BIAS_INIT: float = math.log(math.exp(0.3) - 1.0)  # softplus(b) = 0.3 -> q90 inicial = 1.3 * mu


def decode_outputs(raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convierte logits ``[B, H, 2]`` en ``(mu, q90)`` con ``q90 >= mu`` y ``mu > 0``."""
    mu = F.softplus(raw[..., 0]) + EPS_MU
    q90 = mu + F.softplus(raw[..., 1])
    return mu, q90


def count_trainable_parameters(model: nn.Module) -> int:
    """Cuenta exactamente los parámetros entrenables de un modelo."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class GlobalMLPForecaster(nn.Module):
    """MLP global direct multi-step: historia aplanada + calendario futuro + embedding de ruta."""

    def __init__(
        self,
        num_routes: int,
        lookback: int,
        horizon: int,
        n_cal: int,
        emb_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.embedding = nn.Embedding(num_routes, emb_dim)
        in_dim = lookback + horizon * n_cal + emb_dim + 1  # historia + calendario futuro + emb + log_level
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(int(num_layers)):
            layers += [nn.Linear(dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim, 2 * horizon)
        with torch.no_grad():
            bias = self.head.bias.view(horizon, 2)
            bias[:, 0] = _MU_BIAS_INIT
            bias[:, 1] = _DELTA_BIAS_INIT

    def forward(
        self,
        hist_y: torch.Tensor,
        hist_cal: torch.Tensor,
        fut_cal: torch.Tensor,
        route_idx: torch.Tensor,
        log_level: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del hist_cal  # el MLP usa solo la historia del objetivo + calendario futuro
        emb = self.embedding(route_idx)
        x = torch.cat([hist_y, fut_cal.flatten(1), emb, log_level], dim=1)
        raw = self.head(self.backbone(x)).view(-1, self.horizon, 2)
        return decode_outputs(raw)


class GlobalGRUForecaster(nn.Module):
    """Encoder GRU causal + decodificador por paso con calendario futuro y embedding de ruta."""

    def __init__(
        self,
        num_routes: int,
        lookback: int,
        horizon: int,
        n_cal: int,
        emb_dim: int = 8,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.embedding = nn.Embedding(num_routes, emb_dim)
        self.encoder = nn.GRU(
            input_size=1 + n_cal,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.state_dropout = nn.Dropout(dropout)
        dec_in = hidden_size + emb_dim + 1 + n_cal + 1  # estado + emb + log_level + cal futuro + paso h
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )
        self.register_buffer("step_enc", torch.linspace(1.0 / horizon, 1.0, horizon).view(1, horizon, 1))
        with torch.no_grad():
            last: nn.Linear = self.decoder[-1]  # type: ignore[assignment]
            last.bias[0] = _MU_BIAS_INIT
            last.bias[1] = _DELTA_BIAS_INIT

    def forward(
        self,
        hist_y: torch.Tensor,
        hist_cal: torch.Tensor,
        fut_cal: torch.Tensor,
        route_idx: torch.Tensor,
        log_level: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = hist_y.shape[0]
        seq = torch.cat([hist_y.unsqueeze(-1), hist_cal], dim=-1)  # [B, L, 1 + C]
        _, h_n = self.encoder(seq)
        state = self.state_dropout(h_n[-1])  # [B, hidden]
        emb = self.embedding(route_idx)
        ctx = torch.cat([state, emb, log_level], dim=-1).unsqueeze(1).expand(batch, self.horizon, -1)
        step = self.step_enc.expand(batch, self.horizon, 1)
        raw = self.decoder(torch.cat([ctx, fut_cal, step], dim=-1))  # [B, H, 2]
        return decode_outputs(raw)


def build_model(
    arch: str,
    hparams: Mapping[str, Any],
    num_routes: int,
    lookback: int,
    horizon: int,
    n_cal: int,
    emb_dim: int,
) -> nn.Module:
    """Instancia la arquitectura ``mlp`` o ``gru`` con los hiperparámetros dados."""
    common = dict(
        num_routes=int(num_routes),
        lookback=int(lookback),
        horizon=int(horizon),
        n_cal=int(n_cal),
        emb_dim=int(emb_dim),
        hidden_size=int(hparams["hidden_size"]),
        num_layers=int(hparams["num_layers"]),
        dropout=float(hparams["dropout"]),
    )
    if arch == "mlp":
        return GlobalMLPForecaster(**common)
    if arch == "gru":
        return GlobalGRUForecaster(**common)
    raise ValueError(f"Arquitectura desconocida '{arch}'. Opciones: mlp, gru.")


def pinball_loss(y: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball loss promedio para el cuantil ``tau``."""
    diff = y - q
    return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))


class MultiHorizonLoss(nn.Module):
    """``poisson_nll(y, mu) + lambda_pinball * pinball(y, q90)`` sobre todos los pasos del horizonte."""

    def __init__(self, lambda_pinball: float = 1.0, tau: float = 0.90) -> None:
        super().__init__()
        self.lambda_pinball = float(lambda_pinball)
        self.tau = float(tau)
        self.poisson = nn.PoissonNLLLoss(log_input=False, full=False, eps=1e-8, reduction="mean")

    def forward(
        self, mu: torch.Tensor, q90: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poisson = self.poisson(mu, y)
        pinball = pinball_loss(y, q90, self.tau)
        return poisson + self.lambda_pinball * pinball, poisson, pinball
