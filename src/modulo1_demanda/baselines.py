"""Modelos baseline y evaluador reutilizable de origen rodante (Rolling Origin) para M1.

Responsabilidad (100% offline sobre ``data/processed/m1/demanda_diaria.parquet``):
    1. Exponer un evaluador reutilizable de **origen rodante a horizonte de 30 días**
       (``generate_rolling_origins``, ``compute_mase_scales``, ``evaluate_forecast_records``)
       sobre ``val`` y ``test``, que será consumido de forma idéntica por ``evaluate.py``
       para comparar modelos neuronales bajo exactamente las mismas ventanas y presupuesto.
    2. Implementar y evaluar dos modelos baseline sin fuga de información:
         - ``seasonal_naive``: repite el patrón semanal más reciente en ``t_0``
           (``y[t_0 + ((h-1) mod 7) - 6]``) y suma para P90 el percentil 90 empírico de
           los residuales de ``train`` por ruta (truncado ``>= 0``).
         - ``ridge_calendar``: modelo global lineal ``Ridge`` (pronóstico puntual) +
           ``QuantileRegressor(quantile=0.90)`` (pronóstico P90) entrenado estrictamente en
           ``train`` sobre ``direct_tabular_h30_safe_features`` + ``future_known_calendar_exogenous``
           + codificación ``route_id`` (con interacciones ligeras ``direction x calendario``),
           seleccionando ``alpha`` sobre ``val``.
    3. Calcular MAE, RMSE, MASE (m=7 en ``train``), errores por bloques de horizonte
       (``1-7``, ``8-14``, ``15-30``), cobertura empírica del P90 y Pinball Loss (P90),
       exportando resultados completos a ``reports/metrics/m1_baselines_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import add_config_override_args, load_module_config, resolve_path
from src.utils.seed import set_seed

FUTURE_CALENDAR_COLS: List[str] = [
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_holiday_co",
    "is_puente_festivo",
    "is_salida_puente",
    "is_retorno_puente",
    "is_semana_santa",
    "is_temporada_mitad_anio",
    "is_temporada_fin_anio",
]

DIRECT_H30_SAFE_COLS: List[str] = [
    "lag_30",
    "roll_mean_7_h30",
    "roll_std_7_h30",
    "roll_mean_14_h30",
    "roll_std_14_h30",
    "roll_mean_28_h30",
    "roll_std_28_h30",
]


def pinball_loss_np(y_true: np.ndarray, y_pred_q: np.ndarray, quantile: float = 0.90) -> float:
    """Calcula la pérdida Pinball promedio para el cuantil ``quantile``."""
    diff = y_true - y_pred_q
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


def compute_mase_scales(df: pd.DataFrame, seasonality: int = 7) -> Dict[str, float]:
    """Calcula el denominador de escala estacional MASE (m=7) por ruta usando SOLO ``train``."""
    df_train = df[df["split"] == "train"].sort_values(["route_id", "fecha_despacho"])
    scales: Dict[str, float] = {}
    for rid, grp in df_train.groupby("route_id", sort=False):
        vals = grp["pasajeros"].to_numpy(dtype=np.float64)
        if len(vals) <= seasonality:
            raise ValueError(f"Serie train demasiado corta en ruta {rid} para m={seasonality}.")
        denom = float(np.mean(np.abs(vals[seasonality:] - vals[:-seasonality])))
        scales[str(rid)] = max(denom, 1e-8)
    return scales


def generate_rolling_origins(
    split_start: str | pd.Timestamp,
    split_end: str | pd.Timestamp,
    horizon: int = 30,
    step_days: int = 7,
) -> List[pd.Timestamp]:
    """Genera los orígenes ``t_0`` cuyas ventanas ``[t_0 + 1, ..., t_0 + H]`` caen 100% dentro del split.

    El primer origen es ``split_start - 1 día`` (pronosticando ``[split_start, split_start + H - 1]``)
    y avanza cada ``step_days`` hasta que ``t_0 + H <= split_end``.
    """
    s_start = pd.Timestamp(split_start).normalize()
    s_end = pd.Timestamp(split_end).normalize()
    first_origin = s_start - pd.Timedelta(days=1)
    last_origin = s_end - pd.Timedelta(days=int(horizon))

    origins: List[pd.Timestamp] = []
    cursor = first_origin
    while cursor <= last_origin:
        origins.append(cursor)
        cursor = cursor + pd.Timedelta(days=int(step_days))
    return origins


def _compute_slice_metrics(
    sub_df: pd.DataFrame,
    mase_scales: Mapping[str, float],
    quantile: float = 0.90,
) -> Dict[str, float]:
    """Calcula MAE, RMSE, MASE, Pinball(P90) y Cobertura(P90) sobre un DataFrame de predicciones."""
    y_true = sub_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = sub_df["y_pred_p50"].to_numpy(dtype=np.float64)
    y_p90 = sub_df["y_pred_p90"].to_numpy(dtype=np.float64)
    scales_arr = sub_df["route_id"].map(mase_scales).to_numpy(dtype=np.float64)

    abs_err = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mase = float(np.mean(abs_err / scales_arr))
    p_loss = pinball_loss_np(y_true, y_p90, quantile=quantile)
    coverage_p90 = float(np.mean(y_true <= y_p90))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mase": round(mase, 4),
        "pinball_p90": round(p_loss, 4),
        "coverage_p90": round(coverage_p90, 4),
        "coverage_p90_pct": round(100.0 * coverage_p90, 2),
        "n_forecasts": int(len(sub_df)),
    }


def evaluate_forecast_records(
    records_df: pd.DataFrame,
    mase_scales: Mapping[str, float],
    horizon_blocks: Mapping[str, Sequence[int]],
    quantile: float = 0.90,
) -> Dict[str, Any]:
    """Agrega métricas globales, por bloque de horizonte y por cada ruta (reutilizable en evaluate.py).

    Args:
        records_df: DataFrame con columnas ``['route_id', 'origin_date', 'target_date',
            'horizon_step', 'y_true', 'y_pred_p50', 'y_pred_p90']``.
        mase_scales: Diccionario ``{route_id: escala_mase_train}``.
        horizon_blocks: Mapeo ``{'days_1_7': [1, 7], 'days_8_14': [8, 14], 'days_15_30': [15, 30]}``.
        quantile: Cuantil superior evaluado (por defecto ``0.90``).
    """
    overall = _compute_slice_metrics(records_df, mase_scales=mase_scales, quantile=quantile)

    by_horizon: Dict[str, Dict[str, float]] = {}
    for block_name, (h_min, h_max) in horizon_blocks.items():
        mask = (records_df["horizon_step"] >= int(h_min)) & (records_df["horizon_step"] <= int(h_max))
        by_horizon[str(block_name)] = _compute_slice_metrics(
            records_df[mask], mase_scales=mase_scales, quantile=quantile
        )

    by_route: Dict[str, Dict[str, float]] = {}
    for rid, grp in records_df.groupby("route_id", sort=False):
        by_route[str(rid)] = _compute_slice_metrics(grp, mase_scales=mase_scales, quantile=quantile)

    return {
        "overall": overall,
        "by_horizon_block": by_horizon,
        "by_route": by_route,
    }


def run_rolling_origin_evaluation(
    df: pd.DataFrame,
    origins: Sequence[pd.Timestamp],
    horizon: int,
    predict_window_fn: Callable[[pd.DataFrame, pd.Timestamp, int], pd.DataFrame],
) -> pd.DataFrame:
    """Ejecuta ``predict_window_fn`` sobre cada origen ``t_0`` garantizando visibilidad ``<= t_0``."""
    all_windows: List[pd.DataFrame] = []
    for origin_dt in origins:
        win_preds = predict_window_fn(df, origin_dt, horizon)
        all_windows.append(win_preds)
    return pd.concat(all_windows, ignore_index=True)


class SeasonalNaiveBaseline:
    """Seasonal Naive semanal (m=7) con cuantil P90 basado en residuales empíricos de train."""

    def __init__(self, quantile: float = 0.90) -> None:
        self.quantile = float(quantile)
        self.route_q90_residual_: Dict[str, float] = {}
        self.n_parameters_: int = 0

    def fit(self, df_train: pd.DataFrame, horizon: int = 30, step_days: int = 7) -> "SeasonalNaiveBaseline":
        train_sorted = df_train.sort_values(["route_id", "fecha_despacho"])
        t_min = train_sorted["fecha_despacho"].min()
        t_max = train_sorted["fecha_despacho"].max()

        # Orígenes internos dentro de train con al menos 7 días de historia previa y H días hacia adelante
        first_origin = t_min + pd.Timedelta(days=6)
        last_origin = t_max - pd.Timedelta(days=int(horizon))

        residuals_by_route: Dict[str, List[np.ndarray]] = {}
        for rid, grp in train_sorted.groupby("route_id", sort=False):
            grp_idx = grp.set_index("fecha_despacho")["pasajeros"]
            res_list: List[np.ndarray] = []
            cursor = first_origin
            while cursor <= last_origin:
                last_week = np.array(
                    [float(grp_idx.loc[cursor - pd.Timedelta(days=6 - k)]) for k in range(7)],
                    dtype=np.float64,
                )
                idx_pattern = np.arange(horizon) % 7
                y_hat = last_week[idx_pattern]
                target_dates = pd.date_range(
                    cursor + pd.Timedelta(days=1), periods=horizon, freq="1D"
                )
                y_actual = grp_idx.reindex(target_dates).to_numpy(dtype=np.float64)
                res_list.append(y_actual - y_hat)
                cursor += pd.Timedelta(days=int(step_days))

            all_res = np.concatenate(res_list) if res_list else np.zeros(1, dtype=np.float64)
            self.route_q90_residual_[str(rid)] = float(np.quantile(all_res, self.quantile))

        self.n_parameters_ = len(self.route_q90_residual_)
        return self

    def predict_from_origin(
        self,
        df_all: pd.DataFrame,
        origin_dt: pd.Timestamp,
        horizon: int = 30,
    ) -> pd.DataFrame:
        # Restringir estrictamente la historia observada a fecha_despacho <= origin_dt
        hist = df_all[df_all["fecha_despacho"] <= origin_dt]
        target_dates = pd.date_range(origin_dt + pd.Timedelta(days=1), periods=horizon, freq="1D")
        future_truth = df_all[
            (df_all["fecha_despacho"] >= target_dates[0])
            & (df_all["fecha_despacho"] <= target_dates[-1])
        ]

        rows: List[pd.DataFrame] = []
        idx_pattern = np.arange(horizon) % 7

        for rid, grp_h in hist.groupby("route_id", sort=False):
            s_hist = grp_h.set_index("fecha_despacho")["pasajeros"]
            # Patrón semanal más reciente: [t_0 - 6, t_0 - 5, ..., t_0]
            last_week = np.array(
                [float(s_hist.loc[origin_dt - pd.Timedelta(days=6 - k)]) for k in range(7)],
                dtype=np.float64,
            )
            p50 = np.clip(last_week[idx_pattern], a_min=0.0, a_max=None)
            q_offset = self.route_q90_residual_[str(rid)]
            p90 = np.clip(p50 + q_offset, a_min=p50, a_max=None)

            s_true = (
                future_truth[future_truth["route_id"] == rid]
                .set_index("fecha_despacho")["pasajeros"]
                .reindex(target_dates)
                .to_numpy(dtype=np.float64)
            )

            rows.append(
                pd.DataFrame(
                    {
                        "route_id": str(rid),
                        "origin_date": origin_dt,
                        "target_date": target_dates,
                        "horizon_step": np.arange(1, horizon + 1, dtype=np.int16),
                        "y_true": s_true,
                        "y_pred_p50": p50,
                        "y_pred_p90": p90,
                    }
                )
            )

        return pd.concat(rows, ignore_index=True)


class RidgeCalendarBaseline:
    """Modelo global Ridge (puntual) + QuantileRegressor P90 con features seguras a 30 días."""

    def __init__(
        self,
        ridge_alpha: float = 10.0,
        quantile_alpha: float = 0.001,
        quantile: float = 0.90,
        include_interactions: bool = True,
    ) -> None:
        self.ridge_alpha = float(ridge_alpha)
        self.quantile_alpha = float(quantile_alpha)
        self.quantile = float(quantile)
        self.include_interactions = bool(include_interactions)

        self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler_ = StandardScaler()
        self.point_model_ = Ridge(alpha=self.ridge_alpha, random_state=42)
        self.p90_model_ = QuantileRegressor(
            quantile=self.quantile, alpha=self.quantile_alpha, solver="highs"
        )
        self.n_parameters_: int = 0
        self.feature_dim_: int = 0

    def _build_design_matrix(self, df_rows: pd.DataFrame, fit: bool = False) -> np.ndarray:
        num_arr = df_rows[DIRECT_H30_SAFE_COLS + FUTURE_CALENDAR_COLS].to_numpy(dtype=np.float64)
        if fit:
            num_scaled = self.scaler_.fit_transform(num_arr)
            route_ohe = self.ohe_.fit_transform(df_rows[["route_id"]])
        else:
            num_scaled = self.scaler_.transform(num_arr)
            route_ohe = self.ohe_.transform(df_rows[["route_id"]])

        parts = [num_scaled, route_ohe]

        if self.include_interactions:
            # Interacciones parsimoniosas sentido (ida=1/vuelta=0) x calendario de fin de semana/puentes
            is_vuelta = (df_rows["direction"].to_numpy() == "vuelta").astype(np.float64)[:, None]
            cal_key = df_rows[
                [
                    "dow_sin",
                    "dow_cos",
                    "is_weekend",
                    "is_puente_festivo",
                    "is_salida_puente",
                    "is_retorno_puente",
                ]
            ].to_numpy(dtype=np.float64)
            parts.append(is_vuelta * cal_key)

            # Interacción route_ohe x roll_mean_28_h30 estandarizada para capturar escala elástica por ruta
            roll28_col = num_scaled[:, [DIRECT_H30_SAFE_COLS.index("roll_mean_28_h30")]]
            parts.append(route_ohe * roll28_col)

        return np.hstack(parts)

    def fit(self, df_train: pd.DataFrame) -> "RidgeCalendarBaseline":
        # Descartar las primeras 30 filas por ruta donde lag_30 es NaN al inicio de 2021
        valid_train = df_train.dropna(subset=DIRECT_H30_SAFE_COLS).copy()
        X_train = self._build_design_matrix(valid_train, fit=True)
        y_train = valid_train["pasajeros"].to_numpy(dtype=np.float64)

        self.point_model_.fit(X_train, y_train)
        self.p90_model_.fit(X_train, y_train)

        self.feature_dim_ = int(X_train.shape[1])
        # Coeficientes + intercepto de ambos estimadores lineales (puntual + cuantil P90)
        self.n_parameters_ = int(2 * (self.feature_dim_ + 1))
        return self

    def predict_from_origin(
        self,
        df_all: pd.DataFrame,
        origin_dt: pd.Timestamp,
        horizon: int = 30,
    ) -> pd.DataFrame:
        """Predice [origin_dt + 1 .. origin_dt + H] usando únicamente features <= origin_dt.

        Nota de auditoría anti-fuga:
            Para todo ``t in [origin_dt + 1, ..., origin_dt + 30]``, ``t - 30 <= origin_dt``.
            Por tanto, ``DIRECT_H30_SAFE_COLS`` en ``t`` depende exclusivamente de
            observaciones ``<= origin_dt`` (cero información futura dentro de los 30 días).
        """
        t_start = origin_dt + pd.Timedelta(days=1)
        t_end = origin_dt + pd.Timedelta(days=int(horizon))

        window_df = (
            df_all[(df_all["fecha_despacho"] >= t_start) & (df_all["fecha_despacho"] <= t_end)]
            .sort_values(["route_idx", "fecha_despacho"])
            .copy()
        )

        X_win = self._build_design_matrix(window_df, fit=False)
        p50 = np.clip(self.point_model_.predict(X_win), a_min=0.0, a_max=None)
        p90_raw = np.clip(self.p90_model_.predict(X_win), a_min=0.0, a_max=None)
        p90 = np.maximum(p50, p90_raw)

        horizon_steps = ((window_df["fecha_despacho"] - origin_dt).dt.days).to_numpy(dtype=np.int16)

        return pd.DataFrame(
            {
                "route_id": window_df["route_id"].astype(str).to_numpy(),
                "origin_date": origin_dt,
                "target_date": window_df["fecha_despacho"].to_numpy(),
                "horizon_step": horizon_steps,
                "y_true": window_df["pasajeros"].to_numpy(dtype=np.float64),
                "y_pred_p50": p50,
                "y_pred_p90": p90,
            }
        )


def _print_comparison_tables(
    results: Mapping[str, Any],
    route_labels: Mapping[str, str],
) -> None:
    print("\n" + "=" * 108)
    print("COMPARATIVA GLOBAL DE BASELINES EN TEST (ROLLING ORIGIN H=30 DÍAS, PASO=7 DÍAS, 23 ORÍGENES)")
    print("=" * 108)
    summary_rows = []
    for model_name in ("seasonal_naive", "ridge_calendar"):
        m_info = results["models"][model_name]
        ov = m_info["test"]["overall"]
        hb = m_info["test"]["by_horizon_block"]
        summary_rows.append(
            {
                "Modelo": model_name,
                "Params": m_info["n_parameters"],
                "Tiempo(s)": f"{m_info['fit_and_eval_seconds']:.2f}",
                "MAE": f"{ov['mae']:.2f}",
                "RMSE": f"{ov['rmse']:.2f}",
                "MASE(m=7)": f"{ov['mase']:.4f}",
                "MAE 1-7d": f"{hb['days_1_7']['mae']:.2f}",
                "MAE 8-14d": f"{hb['days_8_14']['mae']:.2f}",
                "MAE 15-30d": f"{hb['days_15_30']['mae']:.2f}",
                "Pinball P90": f"{ov['pinball_p90']:.2f}",
                "Cob. P90(%)": f"{ov['coverage_p90_pct']:.2f}%",
            }
        )
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\n" + "-" * 108)
    print("DESGLOSE POR RUTA EN TEST (MAE | RMSE | MASE | Cobertura P90 | Pinball P90)")
    print("-" * 108)
    route_rows = []
    for rid, r_label in route_labels.items():
        sn_r = results["models"]["seasonal_naive"]["test"]["by_route"][rid]
        rc_r = results["models"]["ridge_calendar"]["test"]["by_route"][rid]
        route_rows.append(
            {
                "route_id": rid,
                "Ruta": r_label,
                "SN_MAE": f"{sn_r['mae']:.1f}",
                "Ridge_MAE": f"{rc_r['mae']:.1f}",
                "SN_MASE": f"{sn_r['mase']:.3f}",
                "Ridge_MASE": f"{rc_r['mase']:.3f}",
                "SN_RMSE": f"{sn_r['rmse']:.1f}",
                "Ridge_RMSE": f"{rc_r['rmse']:.1f}",
                "SN_CobP90": f"{sn_r['coverage_p90_pct']:.1f}%",
                "Ridge_CobP90": f"{rc_r['coverage_p90_pct']:.1f}%",
                "SN_PinP90": f"{sn_r['pinball_p90']:.1f}",
                "Ridge_PinP90": f"{rc_r['pinball_p90']:.1f}",
            }
        )
    print(pd.DataFrame(route_rows).to_string(index=False))


def evaluate_baselines(config: Dict[str, Any]) -> Dict[str, Any]:
    """Entrena en ``train`` y evalúa ``seasonal_naive`` y ``ridge_calendar`` en ``val`` y ``test``."""
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    ts_cfg = config["time_series"]
    eval_cfg = config["evaluation"]
    base_cfg = config.get("baselines", {})

    parquet_path = resolve_path(paths_cfg["processed_dataset_file"])
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset procesado no encontrado en {parquet_path}. Ejecuta preprocess.py primero."
        )

    df = pd.read_parquet(parquet_path)
    df["fecha_despacho"] = pd.to_datetime(df["fecha_despacho"])
    df = df.sort_values(["route_idx", "fecha_despacho"]).reset_index(drop=True)

    horizon = int(ts_cfg["forecast_horizon"])
    step_days = int(eval_cfg.get("rolling_origin_step_days", 7))
    seasonality = int(eval_cfg.get("mase_seasonality", 7))
    target_q = float(eval_cfg.get("target_quantile", 0.90))
    horizon_blocks = eval_cfg["horizon_blocks"]
    split_dates = data_cfg["split_dates"]

    mase_scales = compute_mase_scales(df, seasonality=seasonality)
    val_origins = generate_rolling_origins(
        split_dates["val_start"], split_dates["val_end"], horizon=horizon, step_days=step_days
    )
    test_origins = generate_rolling_origins(
        split_dates["test_start"], split_dates["test_end"], horizon=horizon, step_days=step_days
    )

    df_train = df[df["split"] == "train"].copy()

    # -------------------------------------------------------------------------
    # Baseline 1: Seasonal Naive
    # -------------------------------------------------------------------------
    t0_sn = time.perf_counter()
    sn_model = SeasonalNaiveBaseline(quantile=target_q).fit(
        df_train=df_train, horizon=horizon, step_days=step_days
    )
    sn_val_records = run_rolling_origin_evaluation(
        df, val_origins, horizon, sn_model.predict_from_origin
    )
    sn_test_records = run_rolling_origin_evaluation(
        df, test_origins, horizon, sn_model.predict_from_origin
    )
    sn_elapsed = time.perf_counter() - t0_sn

    sn_val_metrics = evaluate_forecast_records(
        sn_val_records, mase_scales=mase_scales, horizon_blocks=horizon_blocks, quantile=target_q
    )
    sn_test_metrics = evaluate_forecast_records(
        sn_test_records, mase_scales=mase_scales, horizon_blocks=horizon_blocks, quantile=target_q
    )

    # -------------------------------------------------------------------------
    # Baseline 2: Ridge + QuantileRegressor (selección de alpha en Val sin tocar Test)
    # -------------------------------------------------------------------------
    t0_rc = time.perf_counter()
    alpha_grid: List[float] = [float(a) for a in base_cfg.get("ridge_alpha_grid", [1.0, 10.0, 50.0])]
    q_alpha = float(base_cfg.get("quantile_alpha", 0.001))
    use_inter = bool(base_cfg.get("include_route_calendar_interactions", True))

    best_alpha = alpha_grid[0]
    best_val_mae = float("inf")
    best_rc_model: Optional[RidgeCalendarBaseline] = None
    best_rc_val_records: Optional[pd.DataFrame] = None

    for cand_alpha in alpha_grid:
        cand_model = RidgeCalendarBaseline(
            ridge_alpha=cand_alpha,
            quantile_alpha=q_alpha,
            quantile=target_q,
            include_interactions=use_inter,
        ).fit(df_train=df_train)
        cand_val_rec = run_rolling_origin_evaluation(
            df, val_origins, horizon, cand_model.predict_from_origin
        )
        cand_mae = float(np.mean(np.abs(cand_val_rec["y_true"] - cand_val_rec["y_pred_p50"])))
        if cand_mae < best_val_mae:
            best_val_mae = cand_mae
            best_alpha = cand_alpha
            best_rc_model = cand_model
            best_rc_val_records = cand_val_rec

    assert best_rc_model is not None and best_rc_val_records is not None
    rc_test_records = run_rolling_origin_evaluation(
        df, test_origins, horizon, best_rc_model.predict_from_origin
    )
    rc_elapsed = time.perf_counter() - t0_rc

    rc_val_metrics = evaluate_forecast_records(
        best_rc_val_records,
        mase_scales=mase_scales,
        horizon_blocks=horizon_blocks,
        quantile=target_q,
    )
    rc_test_metrics = evaluate_forecast_records(
        rc_test_records,
        mase_scales=mase_scales,
        horizon_blocks=horizon_blocks,
        quantile=target_q,
    )

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": {
            "method": "rolling_origin",
            "forecast_horizon_days": horizon,
            "step_days": step_days,
            "mase_seasonality": seasonality,
            "target_quantile": target_q,
            "val_n_origins": len(val_origins),
            "val_first_origin": val_origins[0].strftime("%Y-%m-%d"),
            "val_last_origin": val_origins[-1].strftime("%Y-%m-%d"),
            "test_n_origins": len(test_origins),
            "test_first_origin": test_origins[0].strftime("%Y-%m-%d"),
            "test_last_origin": test_origins[-1].strftime("%Y-%m-%d"),
            "mase_train_scales_by_route": {k: round(v, 4) for k, v in mase_scales.items()},
        },
        "models": {
            "seasonal_naive": {
                "description": (
                    "Repite los últimos 7 días observados en t_0 y suma para P90 el cuantil 0.90 "
                    "empírico de residuales multi-paso de train por ruta."
                ),
                "n_parameters": sn_model.n_parameters_,
                "fit_and_eval_seconds": round(sn_elapsed, 4),
                "train_p90_residual_offsets": {
                    k: round(v, 2) for k, v in sn_model.route_q90_residual_.items()
                },
                "val": sn_val_metrics,
                "test": sn_test_metrics,
            },
            "ridge_calendar": {
                "description": (
                    "Modelo global Ridge (puntual) + QuantileRegressor P90 (HiGHS) sobre "
                    "direct_tabular_h30_safe_features + calendario exógeno colombiano + route_id."
                ),
                "selected_ridge_alpha_on_val": best_alpha,
                "quantile_alpha": q_alpha,
                "feature_dim": best_rc_model.feature_dim_,
                "n_parameters": best_rc_model.n_parameters_,
                "fit_and_eval_seconds": round(rc_elapsed, 4),
                "val": rc_val_metrics,
                "test": rc_test_metrics,
            },
        },
    }

    out_path = resolve_path(paths_cfg["baselines_metrics_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    route_labels = {
        str(r["route_id"]): str(r["route_label"]) for r in data_cfg["selected_routes"]
    }
    _print_comparison_tables(payload, route_labels=route_labels)
    print(f"\n[OK] Métricas de baselines guardadas en: {out_path}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluación Rolling Origin (H=30d) de Baselines del Módulo 1."
    )
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(
        config["reproducibility"]["seed"],
        config["reproducibility"].get("deterministic_torch", True),
    )
    evaluate_baselines(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
