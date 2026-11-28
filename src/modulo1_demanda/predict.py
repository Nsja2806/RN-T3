"""Contrato de inferencia pura para el Módulo 1 (consumido por Streamlit).

Responsabilidad:
    - Cargar exclusivamente pesos entrenados desde ``models/m1/`` y series/calendario
      procesados desde ``data/processed/m1/`` (100% offline, cero reentrenamiento).
    - Soportar inferencia inmediata para cualquier ruta (o todas las rutas), fecha de origen
      (incluyendo el cierre ``2023-12-31`` proyectando enero de 2024 usando
      ``calendario_exogenas.csv``), nivel de servicio (Puntual ``p50`` vs ``p90``),
      ocupación objetivo, multiplicador de demanda y tasa de ausentismo, integrando
      ``compute_fleet_plan``.
    - Exponer ``load_precomputed_reports`` para consultar en la UI las métricas precalculadas
      de 30 semillas y el caso de negocio sin recalcular entrenamientos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulo1_demanda.baselines import FUTURE_CALENDAR_COLS
from src.modulo1_demanda.business_fleet import compute_fleet_plan
from src.modulo1_demanda.nn_data import load_arrays, load_checkpoint
from src.utils.config import load_module_config, resolve_path


def load_artifacts(
    config: Optional[Dict[str, Any]] = None,
    arch: str = "gru",
) -> Dict[str, Any]:
    """Carga pesos preentrenados de M1, panel histórico y calendario extendido offline."""
    cfg = config if config is not None else load_module_config("m1")
    model_dir = resolve_path(cfg["paths"]["model_dir"])

    models_loaded: Dict[str, Any] = {}
    ckpts_loaded: Dict[str, Any] = {}
    for cand in ("gru", "mlp"):
        p = model_dir / f"{cand}_best.pt"
        if p.exists():
            m, ck = load_checkpoint(p)
            models_loaded[cand] = m
            ckpts_loaded[cand] = ck

    if arch not in models_loaded:
        raise FileNotFoundError(
            f"No se encontró el checkpoint '{arch}_best.pt' en {model_dir}. "
            "Ejecuta previamente: python run.py train --module m1 --mode single"
        )

    arrays = load_arrays(cfg)

    cal_csv_path = resolve_path(cfg["paths"]["calendar_exog_file"])
    if not cal_csv_path.exists():
        raise FileNotFoundError(f"No existe el calendario exógeno offline: {cal_csv_path}")
    cal_df = pd.read_csv(cal_csv_path)
    cal_df["fecha_despacho"] = pd.to_datetime(cal_df["fecha_despacho"]).dt.normalize()
    cal_df = cal_df.sort_values("fecha_despacho").reset_index(drop=True)

    route_meta = {str(r["route_id"]): dict(r) for r in cfg["data"]["selected_routes"]}

    return {
        "config": cfg,
        "default_arch": arch,
        "models": models_loaded,
        "checkpoints": ckpts_loaded,
        "arrays": arrays,
        "calendar_df": cal_df,
        "route_meta": route_meta,
    }


def load_precomputed_reports(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Carga los resúmenes JSON precalculados (baselines, eval30 y negocio) para mostrar en UI."""
    cfg = config if config is not None else load_module_config("m1")
    paths_cfg = cfg["paths"]

    out: Dict[str, Any] = {}
    for key, cfg_key in (
        ("baselines", "baselines_metrics_file"),
        ("eval30", "eval30_json"),
        ("nn_test", "nn_metrics_file"),
        ("business", "business_summary_file"),
    ):
        f_path = resolve_path(paths_cfg[cfg_key])
        out[key] = json.loads(f_path.read_text(encoding="utf-8")) if f_path.exists() else None
    return out


@torch.no_grad()
def predict_demand(
    history_window: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    route_id: Optional[str | Sequence[str]] = None,
    origin_date: Optional[str | pd.Timestamp] = None,
    service_level: str = "p90",
    target_occupancy_ratio: Optional[float] = None,
    demand_multiplier: float = 1.0,
    absenteeism_rate: Optional[float] = None,
    arch: Optional[str] = None,
    include_reverse_pair_for_fleet: bool = True,
) -> Dict[str, Any]:
    """Ejecuta inferencia pura a 30 días + plan de flota y conductores sin reentrenar."""
    art = artifacts if artifacts is not None else load_artifacts(config=config, arch=arch or "gru")
    cfg = config if config is not None else art["config"]
    chosen_arch = (arch or art["default_arch"]).lower()
    if chosen_arch not in art["models"]:
        raise ValueError(f"Arquitectura '{chosen_arch}' no disponible en artefactos: {list(art['models'])}")

    model = art["models"][chosen_arch]
    ckpt = art["checkpoints"][chosen_arch]
    arrays = art["arrays"]
    cal_df: pd.DataFrame = art["calendar_df"]
    route_meta_map: Dict[str, Dict[str, Any]] = art["route_meta"]

    lookback = int(ckpt["lookback"])
    horizon = int(ckpt["horizon"])
    floor_frac = float(ckpt["scale_floor_frac"])
    route_train_mean = np.asarray(ckpt["route_train_mean"], dtype=np.float32)

    if origin_date is None:
        origin_dt = pd.Timestamp(cfg["fleet_business"]["analysis"]["forecast_30d_origin_date"]).normalize()
    else:
        origin_dt = pd.Timestamp(origin_date).normalize()

    if origin_dt not in arrays.date_to_idx:
        raise ValueError(
            f"origin_date={origin_dt.strftime('%Y-%m-%d')} fuera del rango histórico disponible "
            f"({arrays.dates[lookback-1].strftime('%Y-%m-%d')} a {arrays.dates[-1].strftime('%Y-%m-%d')})."
        )
    i0 = arrays.date_to_idx[origin_dt]
    if i0 < lookback - 1:
        raise ValueError(
            f"origin_date={origin_dt.strftime('%Y-%m-%d')} requiere al menos {lookback} días previos."
        )

    all_route_ids = list(arrays.route_ids)
    if route_id is None:
        requested_routes = list(all_route_ids)
    elif isinstance(route_id, str):
        requested_routes = [route_id]
    else:
        requested_routes = [str(r) for r in route_id]

    for r in requested_routes:
        if r not in route_meta_map:
            raise ValueError(f"route_id desconocido '{r}'. Válidos: {all_route_ids}")

    eval_routes = list(requested_routes)
    if include_reverse_pair_for_fleet:
        target_corridors = {route_meta_map[r]["corridor_id"] for r in requested_routes}
        for r in all_route_ids:
            if route_meta_map[r]["corridor_id"] in target_corridors and r not in eval_routes:
                eval_routes.append(r)

    hist_dates = pd.date_range(origin_dt - pd.Timedelta(days=lookback - 1), origin_dt, freq="1D")
    fut_dates = pd.date_range(origin_dt + pd.Timedelta(days=1), periods=horizon, freq="1D")

    cal_indexed = cal_df.set_index("fecha_despacho")[FUTURE_CALENDAR_COLS]
    if not fut_dates.isin(cal_indexed.index).all():
        raise ValueError(
            f"El calendario exógeno offline no cubre hasta {fut_dates[-1].strftime('%Y-%m-%d')}."
        )

    hist_cal_np = cal_indexed.loc[hist_dates].to_numpy(dtype=np.float32)
    fut_cal_np = cal_indexed.loc[fut_dates].to_numpy(dtype=np.float32)

    rid_to_idx = {r: idx for idx, r in enumerate(all_route_ids)}
    r_indices = [rid_to_idx[r] for r in eval_routes]

    hist_y_list: List[np.ndarray] = []
    history_frames: List[pd.DataFrame] = []
    for r_i, r_code in zip(r_indices, eval_routes):
        if history_window is not None and len(requested_routes) == 1 and r_code == requested_routes[0]:
            hw = np.asarray(history_window, dtype=np.float32).reshape(-1)
            if len(hw) != lookback:
                raise ValueError(f"history_window debe tener longitud {lookback}, recibido={len(hw)}.")
            hist_vals = hw
        else:
            hist_vals = arrays.y[r_i, i0 - lookback + 1 : i0 + 1].astype(np.float32)
        hist_y_list.append(hist_vals)

        if r_code in requested_routes:
            history_frames.append(
                pd.DataFrame(
                    {
                        "route_id": r_code,
                        "route_label": route_meta_map[r_code]["route_label"],
                        "fecha_despacho": hist_dates,
                        "pasajeros_hist": hist_vals.astype(np.float64),
                    }
                )
            )

    hist_mat = np.stack(hist_y_list, axis=0)
    rmean_sub = route_train_mean[r_indices]
    scales = np.maximum(np.maximum(hist_mat.mean(axis=1), floor_frac * rmean_sub), 1.0)

    b_size = len(eval_routes)
    t_hist_y = torch.from_numpy(hist_mat / scales[:, None])
    t_hist_cal = torch.from_numpy(
        np.broadcast_to(hist_cal_np[None], (b_size, lookback, hist_cal_np.shape[1])).copy()
    )
    t_fut_cal = torch.from_numpy(
        np.broadcast_to(fut_cal_np[None], (b_size, horizon, fut_cal_np.shape[1])).copy()
    )
    t_route = torch.tensor(r_indices, dtype=torch.int64)
    t_log_lvl = torch.from_numpy(np.log(scales / rmean_sub)[:, None].astype(np.float32))

    model.eval()
    mu_s, q90_s = model(t_hist_y, t_hist_cal, t_fut_cal, t_route, t_log_lvl)
    pred_p50 = (mu_s.numpy() * scales[:, None]).astype(np.float64)
    pred_p90 = (q90_s.numpy() * scales[:, None]).astype(np.float64)

    svc = service_level.strip().lower()
    use_p90 = svc in ("p90", "0.90", "90", "conservador")
    demand_col_chosen = "pred_p90" if use_p90 else "pred_p50"

    rows_out: List[pd.DataFrame] = []
    for pos, (r_i, r_code) in enumerate(zip(r_indices, eval_routes)):
        rmeta = route_meta_map[r_code]
        y_real_vals: List[float] = []
        for dt_f in fut_dates:
            if dt_f in arrays.date_to_idx:
                y_real_vals.append(float(arrays.y[r_i, arrays.date_to_idx[dt_f]]))
            else:
                y_real_vals.append(float("nan"))

        rows_out.append(
            pd.DataFrame(
                {
                    "route_id": r_code,
                    "route_label": rmeta["route_label"],
                    "corridor_id": rmeta["corridor_id"],
                    "direction": rmeta["direction"],
                    "origin_date": origin_dt,
                    "target_date": fut_dates,
                    "horizon_step": np.arange(1, horizon + 1, dtype=np.int16),
                    "pred_p50": np.round(pred_p50[pos], 2),
                    "pred_p90": np.round(pred_p90[pos], 2),
                    "y_true": y_real_vals,
                }
            )
        )

    df_all_preds = pd.concat(rows_out, ignore_index=True)

    fleet_plan = compute_fleet_plan(
        demand_df=df_all_preds,
        config=cfg,
        demand_col=demand_col_chosen,
        date_col="target_date",
        target_occupancy_override=target_occupancy_ratio,
        demand_multiplier=demand_multiplier,
        absenteeism_rate_override=absenteeism_rate,
    )

    df_user_preds = (
        df_all_preds[df_all_preds["route_id"].isin(requested_routes)]
        .copy()
        .reset_index(drop=True)
    )
    df_user_preds["pred_p50_adj"] = np.round(
        df_user_preds["pred_p50"] * float(demand_multiplier), 2
    )
    df_user_preds["pred_p90_adj"] = np.round(
        df_user_preds["pred_p90"] * float(demand_multiplier), 2
    )
    df_user_preds["selected_demand"] = np.round(
        df_user_preds[demand_col_chosen] * float(demand_multiplier), 2
    )

    df_history = pd.concat(history_frames, ignore_index=True)

    return {
        "arch": chosen_arch,
        "origin_date": origin_dt.strftime("%Y-%m-%d"),
        "service_level": "p90" if use_p90 else "p50",
        "demand_multiplier": float(demand_multiplier),
        "target_occupancy_ratio": float(
            target_occupancy_ratio
            if target_occupancy_ratio is not None
            else cfg["fleet_business"]["target_occupancy_ratio"]
        ),
        "absenteeism_rate": float(
            absenteeism_rate
            if absenteeism_rate is not None
            else cfg["fleet_business"]["absenteeism_rate"]
        ),
        "requested_routes": requested_routes,
        "history_df": df_history,
        "forecast_df": df_user_preds,
        "fleet_plan": fleet_plan,
    }
