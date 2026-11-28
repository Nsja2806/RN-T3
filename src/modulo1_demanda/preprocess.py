"""Preprocesamiento reproducible de series de tiempo diarias para el Módulo 1 (Demanda).

Responsabilidad:
    1. Consultar vía SoQL (``datos.gov.co/resource/eh75-8ah6.csv``) los despachos en
       ``ORIGEN`` entre ``train_start`` y ``test_end`` para las 8 rutas seleccionadas en
       ``config/m1_demanda.yaml``, agregando directamente a resolución diaria y auditando
       despachos con ``pasajeros = 0`` (con soporte de caché crudo local para re-ejecución offline).
    2. Reindexar cada ruta sobre el calendario completo de 1,095 días (``2021-01-01`` a
       ``2023-12-31``), imputando ``0`` en días faltantes y marcando ``is_missing_day``.
    3. Generar y persistir ``data/processed/m1/calendario_exogenas.csv`` usando ``holidays``
       (``country='CO'``, Ley Emiliani), puentes festivos, Semana Santa, vacaciones escolares
       de mitad de año (junio-julio) y temporada de fin de año (diciembre-enero), además de
       codificaciones cíclicas (sin/cos) conocibles con >= 30 días de anticipación.
    4. Construir rezagos (``lag_1, lag_7, lag_14, lag_28, lag_30``) y medias móviles históricas
       sin fuga de información:
         - Features causales de encoder hasta el origen de pronóstico ``t_0`` (``shift(1)``:
           observables en ``<= t_0`` para el encoder secuencial GRU).
         - Features directas seguras a horizonte H=30 días (``shift(30)``: observables en
           ``t - 30`` sin usar ningún dato dentro de la ventana futura ``[t-29, ..., t]``).
    5. Exportar ``data/processed/m1/demanda_diaria.parquet`` (< 2 MB, 8,760 filas, 0 nulos en
       ``pasajeros``) y ``reports/m1_preprocess_summary.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import holidays
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import add_config_override_args, load_module_config, resolve_path
from src.utils.seed import set_seed


def _soql_get_csv(
    session: requests.Session,
    url: str,
    query: str,
    timeout: int,
    max_retries: int,
) -> pd.DataFrame:
    """Ejecuta una consulta SoQL contra Socrata y retorna un DataFrame en texto."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params={"$query": query}, timeout=timeout)
            if resp.status_code >= 500 and attempt < max_retries:
                time.sleep(2.0 * attempt)
                continue
            if not resp.ok:
                snippet = resp.text[:500].replace("\n", " ")
                raise RuntimeError(f"HTTP {resp.status_code} en SoQL: {snippet}")
            txt = resp.content.decode("utf-8-sig", errors="replace")
            if not txt.strip():
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(txt), dtype=str)
        except Exception:
            if attempt < max_retries:
                time.sleep(2.0 * attempt)
            else:
                raise
    return pd.DataFrame()


def _compute_easter_sunday(year: int) -> date:
    """Algoritmo gregoriano exacto de Meeus/Jones/Butcher para Domingo de Pascua."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l_val = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_val) // 451
    month = (h + l_val - 7 * m + 114) // 31
    day = ((h + l_val - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def build_colombian_calendar(
    start_date_str: str,
    end_date_str: str,
    country_code: str = "CO",
    extra_future_days: int = 35,
) -> pd.DataFrame:
    """Construye el calendario exógeno determinista de Colombia (100% libre de fuga futura).

    Todas las variables generadas dependen únicamente de la fecha calendario ``t``, por lo
    que son exactamente conocidas con 30 o más días de anticipación.
    """
    start_dt = pd.Timestamp(start_date_str).normalize()
    end_dt = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=int(extra_future_days))

    # Margen interno de 7 días a cada extremo para detectar puentes que cruzan fin de año
    pad_start = start_dt - pd.Timedelta(days=7)
    pad_end = end_dt + pd.Timedelta(days=7)
    years = sorted(set(range(pad_start.year, pad_end.year + 1)))

    co_holidays = holidays.country_holidays(country_code, years=years)
    holiday_dates = set(co_holidays.keys())

    # Identificar puentes festivos en Colombia (Ley Emiliani):
    # 1) Lunes festivo -> puente desde viernes de salida hasta lunes festivo de retorno [lun-3d .. lun]
    # 2) Viernes festivo fijo -> fin de semana largo [vie .. dom]
    puente_days: set[date] = set()
    viernes_salida_puente: set[date] = set()
    lunes_retorno_puente: set[date] = set()

    for h_date in holiday_dates:
        if h_date.weekday() == 0:  # Lunes festivo (Ley Emiliani)
            viernes_salida_puente.add(h_date - timedelta(days=3))
            lunes_retorno_puente.add(h_date)
            for offset in (-3, -2, -1, 0):
                puente_days.add(h_date + timedelta(days=offset))
        elif h_date.weekday() == 4:  # Viernes festivo
            # Si no es Viernes Santo (que ya pertenece a Semana Santa), marca puente vie-dom
            viernes_salida_puente.add(h_date - timedelta(days=1))
            for offset in (0, 1, 2):
                puente_days.add(h_date + timedelta(days=offset))

    # Semana Santa por año: desde Viernes previo a Domingo de Ramos (operación éxodo escolar)
    # hasta Domingo de Pascua (operación retorno), inclusive (10 días).
    semana_santa_days: set[date] = set()
    for yr in years:
        easter = _compute_easter_sunday(yr)
        for offset in range(-9, 1):  # Viernes de Dolores (easter-9) .. Domingo de Pascua (easter)
            semana_santa_days.add(easter + timedelta(days=offset))

    cal_index = pd.date_range(start_dt, end_dt, freq="1D")
    rows: List[Dict[str, Any]] = []

    for ts in cal_index:
        d_obj = ts.date()
        dow = int(ts.dayofweek)  # 0=Lunes .. 6=Domingo
        month = int(ts.month)
        doy = int(ts.dayofyear)
        dom = int(ts.day)

        is_hol = 1 if d_obj in holiday_dates else 0
        hol_name = str(co_holidays.get(d_obj, "")) if is_hol else ""

        rows.append(
            {
                "fecha_despacho": ts.strftime("%Y-%m-%d"),
                "day_of_week": dow,
                "dow_sin": round(math.sin(2.0 * math.pi * dow / 7.0), 6),
                "dow_cos": round(math.cos(2.0 * math.pi * dow / 7.0), 6),
                "month": month,
                "month_sin": round(math.sin(2.0 * math.pi * (month - 1) / 12.0), 6),
                "month_cos": round(math.cos(2.0 * math.pi * (month - 1) / 12.0), 6),
                "day_of_year": doy,
                "doy_sin": round(math.sin(2.0 * math.pi * (doy - 1) / 365.25), 6),
                "doy_cos": round(math.cos(2.0 * math.pi * (doy - 1) / 365.25), 6),
                "day_of_month": dom,
                "is_weekend": 1 if dow >= 5 else 0,
                "is_holiday_co": is_hol,
                "holiday_name": hol_name,
                "is_puente_festivo": 1 if d_obj in puente_days else 0,
                "is_salida_puente": 1 if d_obj in viernes_salida_puente else 0,
                "is_retorno_puente": 1 if d_obj in lunes_retorno_puente else 0,
                "is_semana_santa": 1 if d_obj in semana_santa_days else 0,
                "is_temporada_mitad_anio": 1 if month in (6, 7) else 0,
                "is_temporada_fin_anio": 1 if month in (12, 1) else 0,
            }
        )

    return pd.DataFrame(rows)


def fetch_or_load_raw_daily(
    config: Dict[str, Any],
    force_remote: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga la agregación diaria de las 8 rutas vía SoQL (o lee caché crudo si se solicita)."""
    paths_cfg = config["paths"]
    api_cfg = config["socrata_api"]
    data_cfg = config["data"]

    raw_cache_path = resolve_path(paths_cfg["raw_daily_cache_file"])
    raw_veh_cache_path = raw_cache_path.with_name(raw_cache_path.stem + "_vehiculos.csv")

    if raw_cache_path.exists() and raw_veh_cache_path.exists() and not force_remote:
        df_daily = pd.read_csv(raw_cache_path, dtype=str)
        df_veh = pd.read_csv(raw_veh_cache_path, dtype=str)
        return df_daily, df_veh

    st_url = api_cfg["supertransporte_url"]
    timeout = int(api_cfg.get("timeout_seconds", 120))
    max_retries = int(api_cfg.get("max_retries", 3))
    limit_rows = int(api_cfg.get("daily_query_limit", 50000))

    orig_col = data_cfg["origin_col"]
    dest_col = data_cfg["destination_col"]
    date_col = data_cfg["datetime_col"]
    pas_col = data_cfg["target_col"]
    desp_col = data_cfg["dispatches_col"]
    tipo_col = data_cfg["dispatch_type_col"]
    tipo_val = data_cfg["dispatch_type_origin_value"]
    veh_col = data_cfg["vehicle_class_col"]

    split_dates = data_cfg["split_dates"]
    t_start = f"{split_dates['train_start']}T00:00:00.000"
    t_end = f"{split_dates['test_end']}T23:59:59.999"

    routes = data_cfg["selected_routes"]
    route_or = " OR ".join(
        [
            f"({orig_col} = '{r['origin_code']}' AND {dest_col} = '{r['destination_code']}')"
            for r in routes
        ]
    )
    base_where = (
        f"{tipo_col} = '{tipo_val}' "
        f"AND {date_col} >= '{t_start}' AND {date_col} <= '{t_end}' "
        f"AND ({route_or})"
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "RN-T3-M1-Preprocess/1.0"})

    # 1) Agregación diaria principal (pasajeros, despachos, filas)
    q_main = (
        f"SELECT {orig_col}, {dest_col}, date_trunc_ymd({date_col}) AS fecha_dia, "
        f"sum({pas_col}) AS pasajeros, sum({desp_col}) AS despachos, count(*) AS registros_planilla "
        f"WHERE {base_where} "
        f"GROUP BY {orig_col}, {dest_col}, fecha_dia "
        f"ORDER BY {orig_col}, {dest_col}, fecha_dia "
        f"LIMIT {limit_rows}"
    )
    df_main = _soql_get_csv(session, st_url, q_main, timeout=timeout, max_retries=max_retries)

    # 2) Conteo diario de despachos con pasajeros = 0
    q_zero = (
        f"SELECT {orig_col}, {dest_col}, date_trunc_ymd({date_col}) AS fecha_dia, "
        f"sum({desp_col}) AS despachos_pasajeros_cero, count(*) AS registros_pasajeros_cero "
        f"WHERE {base_where} AND {pas_col} = 0 "
        f"GROUP BY {orig_col}, {dest_col}, fecha_dia "
        f"ORDER BY {orig_col}, {dest_col}, fecha_dia "
        f"LIMIT {limit_rows}"
    )
    df_zero = _soql_get_csv(session, st_url, q_zero, timeout=timeout, max_retries=max_retries)

    # 3) Distribución empírica por clase_vehiculo en el periodo 2021-2023
    q_veh = (
        f"SELECT {orig_col}, {dest_col}, {veh_col}, "
        f"sum({pas_col}) AS pasajeros, sum({desp_col}) AS despachos "
        f"WHERE {base_where} "
        f"GROUP BY {orig_col}, {dest_col}, {veh_col} "
        f"LIMIT 2000"
    )
    df_veh = _soql_get_csv(session, st_url, q_veh, timeout=timeout, max_retries=max_retries)

    if not df_zero.empty:
        df_daily = pd.merge(
            df_main,
            df_zero,
            on=[orig_col, dest_col, "fecha_dia"],
            how="left",
        )
    else:
        df_daily = df_main.copy()
        df_daily["despachos_pasajeros_cero"] = "0"
        df_daily["registros_pasajeros_cero"] = "0"

    df_daily["despachos_pasajeros_cero"] = df_daily["despachos_pasajeros_cero"].fillna("0")
    df_daily["registros_pasajeros_cero"] = df_daily["registros_pasajeros_cero"].fillna("0")

    raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(raw_cache_path, index=False, encoding="utf-8")
    df_veh.to_csv(raw_veh_cache_path, index=False, encoding="utf-8")

    return df_daily, df_veh


def add_autoregressive_features(
    df: pd.DataFrame,
    target_col: str,
    lag_list: Sequence[int],
    rolling_windows: Sequence[int],
    forecast_horizon: int,
) -> pd.DataFrame:
    """Construye rezagos y medias móviles históricas sin fuga de información.

    Convención estricta de causalidad:
        1. ``lag_{k}`` en la fila ``t`` corresponde exactamente a ``y[t - k]``.
           - En un encoder secuencial (ej. GRU con origen ``t_0``), el encoder SOLO lee
             filas ``tau <= t_0``, por lo que ningún ``lag_{k}`` toca ``[t_0 + 1, ..., t_0 + H]``.
           - En un modelo tabular directo a horizonte ``H = 30``, solo ``lag_30`` (y ``*_h30``)
             son observables desde ``t - 30`` sin recursión.
        2. ``roll_mean_{w}_past`` y ``roll_std_{w}_past`` usan ``shift(1).rolling(w)``:
           resumen únicamente ``[t - w, ..., t - 1]`` (observables en el paso ``t_0`` del encoder).
        3. ``roll_mean_{w}_h30`` y ``roll_std_{w}_h30`` usan ``shift(forecast_horizon).rolling(w)``:
           resumen únicamente ``[t - H - w + 1, ..., t - H]``, garantizando **cero uso de
           información dentro del horizonte de 30 días** incluso en regresores tabulares directos.
    """
    out = df.sort_values(["route_idx", "fecha_despacho"]).copy().reset_index(drop=True)
    grouped = out.groupby("route_id", sort=False)[target_col]

    for lag in lag_list:
        lag_int = int(lag)
        out[f"lag_{lag_int}"] = grouped.shift(lag_int).astype("float32")

    h_shift = int(forecast_horizon)
    for win in rolling_windows:
        w = int(win)
        # Estadísticos causales al cierre de t-1 (para ventana histórica del encoder <= t_0)
        past_shifted = grouped.shift(1)
        out[f"roll_mean_{w}_past"] = (
            past_shifted.groupby(out["route_id"], sort=False)
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
        out[f"roll_std_{w}_past"] = (
            past_shifted.groupby(out["route_id"], sort=False)
            .rolling(window=w, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
            .astype("float32")
        )

        # Estadísticos desplazados H=30 días (seguros para predicción directa a 30 días sin recursión)
        h_shifted = grouped.shift(h_shift)
        out[f"roll_mean_{w}_h{h_shift}"] = (
            h_shifted.groupby(out["route_id"], sort=False)
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
        out[f"roll_std_{w}_h{h_shift}"] = (
            h_shifted.groupby(out["route_id"], sort=False)
            .rolling(window=w, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
            .astype("float32")
        )

    return out


def _assign_split(dates: pd.Series, split_cfg: Dict[str, str]) -> pd.Series:
    train_start = pd.Timestamp(split_cfg["train_start"])
    train_end = pd.Timestamp(split_cfg["train_end"])
    val_start = pd.Timestamp(split_cfg["val_start"])
    val_end = pd.Timestamp(split_cfg["val_end"])
    test_start = pd.Timestamp(split_cfg["test_start"])
    test_end = pd.Timestamp(split_cfg["test_end"])

    split = pd.Series("out_of_range", index=dates.index, dtype="object")
    split.loc[(dates >= train_start) & (dates <= train_end)] = "train"
    split.loc[(dates >= val_start) & (dates <= val_end)] = "val"
    split.loc[(dates >= test_start) & (dates <= test_end)] = "test"
    return split


def _verify_no_horizon_leakage(df: pd.DataFrame, horizon: int) -> None:
    """Prueba interna determinista de ausencia de fuga futura en rezagos y medias móviles."""
    sample_route = df[df["route_id"] == df["route_id"].iloc[0]].sort_values("fecha_despacho").reset_index(drop=True)
    y = sample_route["pasajeros"].to_numpy(dtype=np.float64)

    # Verificar en t = 100 que lag_30 == y[70] y roll_mean_7_h30 == mean(y[64:71])
    t_idx = 100
    assert np.isclose(sample_route.loc[t_idx, "lag_30"], y[t_idx - 30]), "Fuga/error en lag_30"
    expected_h30_mean7 = np.mean(y[t_idx - horizon - 6 : t_idx - horizon + 1])
    assert np.isclose(
        sample_route.loc[t_idx, f"roll_mean_7_h{horizon}"], expected_h30_mean7, atol=1e-3
    ), "Fuga/error en roll_mean_7_h30"
    expected_past_mean7 = np.mean(y[t_idx - 7 : t_idx])
    assert np.isclose(
        sample_route.loc[t_idx, "roll_mean_7_past"], expected_past_mean7, atol=1e-3
    ), "Fuga/error en roll_mean_7_past (no debe incluir t_idx)"


def run_preprocess(config: Dict[str, Any], force_remote: bool = True) -> Dict[str, Any]:
    """Ejecuta el pipeline completo de preprocesamiento de M1 y exporta artefactos offline."""
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    ts_cfg = config["time_series"]

    split_cfg = data_cfg["split_dates"]
    start_date_str = split_cfg["train_start"]
    end_date_str = split_cfg["test_end"]

    # 1. Construir y guardar calendario exógeno colombiano (100% offline para etapas posteriores)
    cal_df = build_colombian_calendar(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        country_code=str(data_cfg.get("calendar_country", "CO")),
        extra_future_days=int(data_cfg.get("calendar_extra_future_days", 35)),
    )
    cal_out_path = resolve_path(paths_cfg["calendar_exog_file"])
    cal_out_path.parent.mkdir(parents=True, exist_ok=True)
    cal_df.to_csv(cal_out_path, index=False, encoding="utf-8")

    # 2. Descargar (o cargar de caché) los datos diarios agregados de las 8 rutas en ORIGEN
    df_raw_daily, df_veh = fetch_or_load_raw_daily(config=config, force_remote=force_remote)

    orig_col = data_cfg["origin_col"]
    dest_col = data_cfg["destination_col"]
    df_raw_daily["fecha_str"] = df_raw_daily["fecha_dia"].astype(str).str.slice(0, 10)
    df_raw_daily[orig_col] = df_raw_daily[orig_col].astype(str).str.strip().str.zfill(5)
    df_raw_daily[dest_col] = df_raw_daily[dest_col].astype(str).str.strip().str.zfill(5)

    for num_c in (
        "pasajeros",
        "despachos",
        "registros_planilla",
        "despachos_pasajeros_cero",
        "registros_pasajeros_cero",
    ):
        df_raw_daily[num_c] = (
            pd.to_numeric(df_raw_daily[num_c], errors="coerce").fillna(0.0).round().astype("int32")
        )

    # 3. Construir grilla completa de 8 rutas x 1,095 días = 8,760 filas
    full_dates = pd.date_range(start_date_str, end_date_str, freq="1D")
    n_days = len(full_dates)
    selected_routes = data_cfg["selected_routes"]

    grid_frames: List[pd.DataFrame] = []
    for route_idx, r_meta in enumerate(selected_routes):
        g = pd.DataFrame(
            {
                "fecha_str": full_dates.strftime("%Y-%m-%d"),
                "route_idx": np.int16(route_idx),
                "route_id": str(r_meta["route_id"]),
                "corridor_id": str(r_meta["corridor_id"]),
                "direction": str(r_meta["direction"]),
                "origin_code": str(r_meta["origin_code"]).zfill(5),
                "origin_name": str(r_meta["origin_name"]),
                "destination_code": str(r_meta["destination_code"]).zfill(5),
                "destination_name": str(r_meta["destination_name"]),
                "route_label": str(r_meta["route_label"]),
                "predominant_vehicle": str(r_meta["predominant_vehicle"]),
                "cycle_time_hours": np.float32(r_meta["cycle_time_hours"]),
            }
        )
        grid_frames.append(g)

    df_grid = pd.concat(grid_frames, ignore_index=True)

    # Cruzar con observaciones reales de SuperTransporte
    df_merged = pd.merge(
        df_grid,
        df_raw_daily[
            [
                orig_col,
                dest_col,
                "fecha_str",
                "pasajeros",
                "despachos",
                "registros_planilla",
                "despachos_pasajeros_cero",
                "registros_pasajeros_cero",
            ]
        ],
        left_on=["origin_code", "destination_code", "fecha_str"],
        right_on=[orig_col, dest_col, "fecha_str"],
        how="left",
    ).drop(columns=[orig_col, dest_col])

    df_merged["is_missing_day"] = df_merged["pasajeros"].isna().astype("int8")
    for c_imp in (
        "pasajeros",
        "despachos",
        "registros_planilla",
        "despachos_pasajeros_cero",
        "registros_pasajeros_cero",
    ):
        df_merged[c_imp] = df_merged[c_imp].fillna(0).astype("int32")

    # 4. Unir variables exógenas del calendario colombiano
    df_merged = pd.merge(
        df_merged,
        cal_df,
        left_on="fecha_str",
        right_on="fecha_despacho",
        how="left",
    ).drop(columns=["fecha_str"])

    df_merged["fecha_despacho"] = pd.to_datetime(df_merged["fecha_despacho"])
    df_merged["split"] = _assign_split(df_merged["fecha_despacho"], split_cfg)

    # Compactar tipos de calendario
    int8_cols = [
        "day_of_week",
        "month",
        "day_of_month",
        "is_weekend",
        "is_holiday_co",
        "is_puente_festivo",
        "is_salida_puente",
        "is_retorno_puente",
        "is_semana_santa",
        "is_temporada_mitad_anio",
        "is_temporada_fin_anio",
    ]
    for col in int8_cols:
        df_merged[col] = df_merged[col].astype("int8")
    df_merged["day_of_year"] = df_merged["day_of_year"].astype("int16")
    for fcol in ("dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"):
        df_merged[fcol] = df_merged[fcol].astype("float32")

    # 5. Generar rezagos y medias móviles respetando estrictamente la causalidad y el horizonte 30d
    horizon = int(ts_cfg["forecast_horizon"])
    df_final = add_autoregressive_features(
        df=df_merged,
        target_col=data_cfg["target_col"],
        lag_list=ts_cfg["lag_features"],
        rolling_windows=ts_cfg["rolling_windows"],
        forecast_horizon=horizon,
    )

    # Validaciones de integridad obligatorias
    expected_rows = len(selected_routes) * n_days
    if len(df_final) != expected_rows:
        raise AssertionError(f"Filas esperadas={expected_rows}, obtenidas={len(df_final)}")
    if int(df_final["pasajeros"].isna().sum()) != 0:
        raise AssertionError("Existen valores nulos en la variable objetivo 'pasajeros'.")
    if (df_final["split"] == "out_of_range").any():
        raise AssertionError("Existen fechas fuera de los rangos train/val/test.")
    _verify_no_horizon_leakage(df_final, horizon=horizon)

    # 6. Exportar Parquet y verificar peso < 2 MB
    parquet_path = resolve_path(paths_cfg["processed_dataset_file"])
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

    parquet_bytes = parquet_path.stat().st_size
    parquet_mb = parquet_bytes / (1024.0 * 1024.0)
    if parquet_mb >= 2.0:
        raise AssertionError(
            f"El archivo parquet ({parquet_mb:.3f} MB) excede el límite máximo de 2.0 MB."
        )

    # Mapear distribución empírica de clase_vehiculo por ruta (2021-2023)
    veh_share_map: Dict[str, Dict[str, float]] = {}
    if not df_veh.empty:
        df_veh_c = df_veh.copy()
        df_veh_c[orig_col] = df_veh_c[orig_col].astype(str).str.strip().str.zfill(5)
        df_veh_c[dest_col] = df_veh_c[dest_col].astype(str).str.strip().str.zfill(5)
        df_veh_c["pas_num"] = pd.to_numeric(df_veh_c["pasajeros"], errors="coerce").fillna(0.0)
        for (oc, dc), grp in df_veh_c.groupby([orig_col, dest_col]):
            rid = f"{oc}_{dc}"
            tot = float(grp["pas_num"].sum())
            if tot > 0:
                veh_share_map[rid] = {
                    str(r[data_cfg["vehicle_class_col"]]): round(100.0 * float(r["pas_num"]) / tot, 2)
                    for _, r in grp.sort_values("pas_num", ascending=False).iterrows()
                }

    # 7. Construir resumen estadístico por ruta y global en JSON
    route_summaries: List[Dict[str, Any]] = []
    for r_meta in selected_routes:
        rid = str(r_meta["route_id"])
        sub = df_final[df_final["route_id"] == rid]
        pas_s = sub["pasajeros"]
        desp_s = sub["despachos"]
        zero_desp = int(sub["despachos_pasajeros_cero"].sum())
        tot_desp = int(desp_s.sum())
        zero_desp_pct = (100.0 * zero_desp / tot_desp) if tot_desp > 0 else 0.0

        train_pas = sub.loc[sub["split"] == "train", "pasajeros"]
        val_pas = sub.loc[sub["split"] == "val", "pasajeros"]
        test_pas = sub.loc[sub["split"] == "test", "pasajeros"]

        route_summaries.append(
            {
                "route_id": rid,
                "route_label": r_meta["route_label"],
                "corridor_id": r_meta["corridor_id"],
                "direction": r_meta["direction"],
                "predominant_vehicle_cfg": r_meta["predominant_vehicle"],
                "empirical_vehicle_share_pct_2021_2023": veh_share_map.get(rid, {}),
                "cycle_time_hours": float(r_meta["cycle_time_hours"]),
                "days_total": int(len(sub)),
                "missing_days_imputed_zero": int(sub["is_missing_day"].sum()),
                "days_with_zero_total_passengers": int((pas_s == 0).sum()),
                "total_passengers": int(pas_s.sum()),
                "total_dispatches": tot_desp,
                "dispatches_with_zero_passengers": zero_desp,
                "dispatches_with_zero_passengers_pct": round(zero_desp_pct, 2),
                "daily_passengers_mean": round(float(pas_s.mean()), 2),
                "daily_passengers_std": round(float(pas_s.std(ddof=1)), 2),
                "daily_passengers_min": int(pas_s.min()),
                "daily_passengers_p50": round(float(pas_s.median()), 2),
                "daily_passengers_p90": round(float(pas_s.quantile(0.90)), 2),
                "daily_passengers_max": int(pas_s.max()),
                "split_means": {
                    "train_mean": round(float(train_pas.mean()), 2),
                    "val_mean": round(float(val_pas.mean()), 2),
                    "test_mean": round(float(test_pas.mean()), 2),
                },
            }
        )

    exog_cols = [
        "day_of_week",
        "dow_sin",
        "dow_cos",
        "month",
        "month_sin",
        "month_cos",
        "day_of_year",
        "doy_sin",
        "doy_cos",
        "day_of_month",
        "is_weekend",
        "is_holiday_co",
        "is_puente_festivo",
        "is_salida_puente",
        "is_retorno_puente",
        "is_semana_santa",
        "is_temporada_mitad_anio",
        "is_temporada_fin_anio",
    ]
    encoder_past_cols = [f"lag_{int(l)}" for l in ts_cfg["lag_features"]] + [
        col
        for w in ts_cfg["rolling_windows"]
        for col in (f"roll_mean_{int(w)}_past", f"roll_std_{int(w)}_past")
    ]
    direct_h30_cols = ["lag_30"] + [
        col
        for w in ts_cfg["rolling_windows"]
        for col in (f"roll_mean_{int(w)}_h{horizon}", f"roll_std_{int(w)}_h{horizon}")
    ]

    summary_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(paths_cfg["processed_dataset_file"]),
        "calendar_path": str(paths_cfg["calendar_exog_file"]),
        "parquet_size_bytes": parquet_bytes,
        "parquet_size_mb": round(parquet_mb, 4),
        "n_routes": len(selected_routes),
        "n_days_per_route": n_days,
        "total_rows": int(len(df_final)),
        "target_null_count": int(df_final["pasajeros"].isna().sum()),
        "date_range": {"start": start_date_str, "end": end_date_str},
        "splits": {
            sp: {
                "rows": int((df_final["split"] == sp).sum()),
                "days_per_route": int((df_final["split"] == sp).sum() // len(selected_routes)),
                "start": split_cfg[f"{sp}_start"],
                "end": split_cfg[f"{sp}_end"],
            }
            for sp in ("train", "val", "test")
        },
        "zero_passenger_dispatches_audit": {
            "total_dispatches_all_routes": int(df_final["despachos"].sum()),
            "total_dispatches_with_zero_passengers": int(
                df_final["despachos_pasajeros_cero"].sum()
            ),
            "pct_dispatches_with_zero_passengers": round(
                100.0
                * float(df_final["despachos_pasajeros_cero"].sum())
                / max(1.0, float(df_final["despachos"].sum())),
                2,
            ),
        },
        "feature_groups_anti_leakage": {
            "future_known_calendar_exogenous": exog_cols,
            "encoder_step_causal_features_up_to_t0": encoder_past_cols,
            "direct_tabular_h30_safe_features": direct_h30_cols,
        },
        "columns": list(df_final.columns),
        "routes_summary": route_summaries,
    }

    summary_path = resolve_path(paths_cfg["preprocess_summary_file"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Calendario exógeno guardado en: {cal_out_path} ({len(cal_df):,} días)")
    print(
        f"[OK] Dataset procesado guardado en:  {parquet_path} "
        f"({len(df_final):,} filas, {parquet_mb:.3f} MB < 2.0 MB)"
    )
    print(f"[OK] Resumen estadístico guardado en: {summary_path}")
    return summary_payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocesamiento diario M1 (8 rutas intermunicipales, 2021-2023)."
    )
    parser.add_argument(
        "--offline-cache",
        action="store_true",
        help="Usa el caché crudo local en data/raw/m1/ si ya fue descargado previamente.",
    )
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(
        config["reproducibility"]["seed"],
        config["reproducibility"].get("deterministic_torch", True),
    )
    run_preprocess(config=config, force_remote=not args.offline_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
