"""Datos, ventanas causales, predictor rolling-origin y checkpoints de las redes de M1.

Responsabilidad (100% offline sobre ``data/processed/m1/demanda_diaria.parquet``):
    - Cargar el panel (8 rutas x 1,095 días) como arreglos NumPy y el calendario exógeno.
    - Construir ventanas por origen ``t_0``: historia ``[t_0-L+1, ..., t_0]`` (causal) y calendario
      conocido de ``[t_0+1, ..., t_0+H]``. Ningún valor del objetivo posterior a ``t_0`` entra al input.
    - Normalizar por ventana: ``s = max(media_L, scale_floor_frac * media_train_ruta, 1)``; historia y
      objetivo se dividen por ``s``; ``log(s / media_train_ruta)`` se pasa como feature de nivel.
    - ``NNPredictor``: interfaz ``predict_from_origin`` idéntica a la de los baselines para reutilizar
      ``run_rolling_origin_evaluation`` / ``evaluate_forecast_records`` de ``baselines.py``.
    - Guardar/cargar checkpoints en ``models/m1/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.modulo1_demanda.baselines import (
    FUTURE_CALENDAR_COLS,
    compute_mase_scales,
    generate_rolling_origins,
)
from src.modulo1_demanda.models import build_model, count_trainable_parameters
from src.utils.config import resolve_path

WindowDict = Dict[str, torch.Tensor]


@dataclass
class M1Arrays:
    """Panel de demanda y calendario en formato matricial."""

    df: pd.DataFrame
    dates: pd.DatetimeIndex
    route_ids: List[str]
    y: np.ndarray  # [R, T] float32 (pasajeros)
    cal: np.ndarray  # [T, C] float32 (calendario conocido a futuro)
    route_train_mean: np.ndarray  # [R] media diaria en train
    date_to_idx: Dict[pd.Timestamp, int]


def load_arrays(config: Dict[str, Any]) -> M1Arrays:
    """Carga el parquet procesado y lo convierte en arreglos ``[ruta, día]``."""
    parquet_path = resolve_path(config["paths"]["processed_dataset_file"])
    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe {parquet_path}. Ejecuta primero preprocess.py.")
    df = pd.read_parquet(parquet_path)
    df["fecha_despacho"] = pd.to_datetime(df["fecha_despacho"])
    df = df.sort_values(["route_idx", "fecha_despacho"]).reset_index(drop=True)

    route_meta = df.drop_duplicates("route_idx").sort_values("route_idx")
    route_ids = route_meta["route_id"].astype(str).tolist()
    pivot = df.pivot(index="route_idx", columns="fecha_despacho", values="pasajeros").sort_index()
    dates = pd.DatetimeIndex(pivot.columns)
    first_route = df[df["route_idx"] == route_meta["route_idx"].iloc[0]].sort_values("fecha_despacho")
    cal = first_route[FUTURE_CALENDAR_COLS].to_numpy(dtype=np.float32)
    train_mean = (
        df[df["split"] == "train"].groupby("route_idx")["pasajeros"].mean().sort_index().to_numpy(dtype=np.float32)
    )
    return M1Arrays(
        df=df,
        dates=dates,
        route_ids=route_ids,
        y=pivot.to_numpy(dtype=np.float32),
        cal=cal,
        route_train_mean=train_mean,
        date_to_idx={d: i for i, d in enumerate(dates)},
    )


def build_windows(
    arrays: M1Arrays,
    origin_idx: np.ndarray,
    lookback: int,
    horizon: int,
    scale_floor_frac: float,
    route_mean: np.ndarray,
) -> WindowDict:
    """Construye tensores de entrada/objetivo para todos los (origen, ruta). Orden: ruta-mayor."""
    y, cal = arrays.y, arrays.cal
    n_routes = y.shape[0]
    n_orig = len(origin_idx)
    hidx = origin_idx[:, None] + np.arange(-lookback + 1, 1)[None, :]  # [N, L]  (<= t_0)
    fidx = origin_idx[:, None] + np.arange(1, horizon + 1)[None, :]  # [N, H]  (> t_0)
    if hidx.min() < 0 or fidx.max() >= y.shape[1]:
        raise IndexError("Ventana fuera del rango del dataset.")

    hist = y[:, hidx]  # [R, N, L]
    y_future = y[:, fidx]  # [R, N, H]
    hist_cal = cal[hidx]  # [N, L, C]
    fut_cal = cal[fidx]  # [N, H, C]

    floor = (scale_floor_frac * route_mean)[:, None]
    scale = np.maximum(np.maximum(hist.mean(axis=-1), floor), 1.0)  # [R, N]
    n_cal = cal.shape[1]

    def flat(a: np.ndarray, *shape: int) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32).reshape(n_routes * n_orig, *shape))

    return {
        "hist_y": flat(hist / scale[..., None], lookback),
        "hist_cal": flat(np.broadcast_to(hist_cal[None], (n_routes, n_orig, lookback, n_cal)), lookback, n_cal),
        "fut_cal": flat(np.broadcast_to(fut_cal[None], (n_routes, n_orig, horizon, n_cal)), horizon, n_cal),
        "log_level": flat(np.log(scale / route_mean[:, None]), 1),
        "scale": flat(scale, 1),
        "y_scaled": flat(y_future / scale[..., None], horizon),
        "y_future": flat(y_future, horizon),
        "route": torch.from_numpy(np.repeat(np.arange(n_routes), n_orig).astype(np.int64)),
    }


def origin_indices(arrays: M1Arrays, config: Dict[str, Any], split: str) -> np.ndarray:
    """Orígenes (paso ``stride``) cuyas ventanas de objetivo caen 100% dentro del split."""
    sd = config["data"]["split_dates"]
    lookback = int(config["time_series"]["lookback_window"])
    horizon = int(config["time_series"]["forecast_horizon"])
    stride = int(config["time_series"].get("stride", 1))
    idx = arrays.date_to_idx
    if split == "train":
        lo, hi = lookback - 1, idx[pd.Timestamp(sd["train_end"])] - horizon
    else:
        lo = idx[pd.Timestamp(sd[f"{split}_start"])] - 1
        hi = idx[pd.Timestamp(sd[f"{split}_end"])] - horizon
    return np.arange(lo, hi + 1, stride)


def prepare_context(config: Dict[str, Any], with_train_windows: bool = True) -> Dict[str, Any]:
    """Carga datos y deja listo todo lo compartido por train/evaluate (ventanas, escalas MASE, orígenes)."""
    arrays = load_arrays(config)
    ts, ev, data_cfg = config["time_series"], config["evaluation"], config["data"]
    lookback, horizon = int(ts["lookback_window"]), int(ts["forecast_horizon"])
    floor_frac = float(config["model"]["scale_floor_frac"])
    step_days = int(ev.get("rolling_origin_step_days", 7))
    sd = data_cfg["split_dates"]

    windows: Dict[str, WindowDict] = {}
    if with_train_windows:
        for split in ("train", "val"):
            windows[split] = build_windows(
                arrays, origin_indices(arrays, config, split), lookback, horizon, floor_frac, arrays.route_train_mean
            )
    return {
        "arrays": arrays,
        "windows": windows,
        "lookback": lookback,
        "horizon": horizon,
        "floor_frac": floor_frac,
        "mase_scales": compute_mase_scales(arrays.df, seasonality=int(ev.get("mase_seasonality", 7))),
        "val_origins": generate_rolling_origins(sd["val_start"], sd["val_end"], horizon, step_days),
        "test_origins": generate_rolling_origins(sd["test_start"], sd["test_end"], horizon, step_days),
        "horizon_blocks": ev["horizon_blocks"],
        "quantile": float(ev.get("target_quantile", 0.90)),
        "n_cal": int(arrays.cal.shape[1]),
        "emb_dim": int(config["model"]["route_embedding_dim"]),
    }


def model_forward(model: nn.Module, w: WindowDict, sl: slice) -> Tuple[torch.Tensor, torch.Tensor]:
    return model(w["hist_y"][sl], w["hist_cal"][sl], w["fut_cal"][sl], w["route"][sl], w["log_level"][sl])


@torch.no_grad()
def predict_windows(model: nn.Module, w: WindowDict, batch_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predice ``(mu, q90)`` en espacio normalizado para todas las muestras de ``w``."""
    model.eval()
    n = w["route"].shape[0]
    mus, qs = [], []
    for start in range(0, n, batch_size):
        mu, q = model_forward(model, w, slice(start, start + batch_size))
        mus.append(mu)
        qs.append(q)
    return torch.cat(mus), torch.cat(qs)


class NNPredictor:
    """Predictor con la misma interfaz ``predict_from_origin`` que los baselines (rolling origin)."""

    def __init__(
        self,
        model: nn.Module,
        arrays: M1Arrays,
        lookback: int,
        horizon: int,
        floor_frac: float,
        route_mean: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model.eval()
        self.arrays = arrays
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.floor_frac = float(floor_frac)
        self.route_mean = arrays.route_train_mean if route_mean is None else np.asarray(route_mean, dtype=np.float32)

    def predict_from_origin(self, df_all: pd.DataFrame, origin_dt: pd.Timestamp, horizon: int = 30) -> pd.DataFrame:
        del df_all  # los datos ya están en ``arrays``; solo se leen índices <= t_0 para el input
        if int(horizon) != self.horizon:
            raise ValueError(f"El modelo fue entrenado para H={self.horizon}, se pidió H={horizon}.")
        i0 = self.arrays.date_to_idx[pd.Timestamp(origin_dt).normalize()]
        w = build_windows(
            self.arrays, np.array([i0]), self.lookback, self.horizon, self.floor_frac, self.route_mean
        )
        mu_s, q_s = predict_windows(self.model, w)
        mu = (mu_s * w["scale"]).numpy()
        q90 = (q_s * w["scale"]).numpy()
        y_true = w["y_future"].numpy()
        target_dates = self.arrays.dates[i0 + 1 : i0 + 1 + self.horizon]
        frames = []
        for r, rid in enumerate(self.arrays.route_ids):
            frames.append(
                pd.DataFrame(
                    {
                        "route_id": rid,
                        "origin_date": pd.Timestamp(origin_dt),
                        "target_date": target_dates,
                        "horizon_step": np.arange(1, self.horizon + 1, dtype=np.int16),
                        "y_true": y_true[r].astype(np.float64),
                        "y_pred_p50": mu[r].astype(np.float64),
                        "y_pred_p90": q90[r].astype(np.float64),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    arch: str,
    hparams: Dict[str, Any],
    ctx: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    """Guarda pesos + escaladores + metadatos en un único ``.pt`` (solo tipos básicos y tensores)."""
    arrays: M1Arrays = ctx["arrays"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "hparams": dict(hparams),
            "state_dict": model.state_dict(),
            "route_ids": list(arrays.route_ids),
            "route_train_mean": [float(v) for v in arrays.route_train_mean],
            "calendar_cols": list(FUTURE_CALENDAR_COLS),
            "lookback": int(ctx["lookback"]),
            "horizon": int(ctx["horizon"]),
            "emb_dim": int(ctx["emb_dim"]),
            "scale_floor_frac": float(ctx["floor_frac"]),
            "n_parameters": count_trainable_parameters(model),
            "meta": dict(meta),
        },
        path,
    )


def load_checkpoint(path: Path) -> Tuple[nn.Module, Dict[str, Any]]:
    """Carga un checkpoint y reconstruye el modelo (sin reentrenar)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No existe el checkpoint {path}. Entrena primero (--mode single).")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(
        ckpt["arch"],
        ckpt["hparams"],
        num_routes=len(ckpt["route_ids"]),
        lookback=ckpt["lookback"],
        horizon=ckpt["horizon"],
        n_cal=len(ckpt["calendar_cols"]),
        emb_dim=ckpt["emb_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
