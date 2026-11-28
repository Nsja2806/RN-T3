"""Motor de dimensionamiento de flota, conductores y evaluación económica del Módulo 1.

Responsabilidad (100% offline, cero constantes quemadas, regido por ``config/m1_demanda.yaml``):
    1. Dimensionar despachos diarios ``D_{r,t} = ceil(y_hat / (C_r * rho))``, acoplar pares
       bidireccionales ``A <-> B`` según vueltas posibles por vehículo/día ``V_c`` (incluyendo
       el corredor de 24h ``Bogotá <-> Cali`` a razón de 1 despacho/día por vehículo asignado)
       y calcular flota operativa y flota total con reserva técnica.
    2. Calcular conductores activos en turno y plantilla total contratada incorporando la
       normativa colombiana vigente a 2026:
         - Ley 2101 de 2021: jornada ordinaria máxima de 42.0 h/semana (7.0 h/día en 6 días).
         - Decreto 1393 de 1970 Art. 56 y Ley 2466 de 2025 Art. 13: tope diario de 10.0 h
           con pausas activas PESV (Resolución 40595 de 2022).
         - Resolución 315 de 2013 Art. 6: doble conductor obligatorio en recorridos > 8.0 h
           por sentido (aplica automáticamente a ``Bogotá <-> Cali``).
         - Factor de descanso semanal obligatorio (7/6) y tasa de ausentismo configurable.
    3. Evaluar en ``test`` el valor operativo y económico de ``GRU`` vs ``ridge_calendar`` y
       ``Puntual`` vs ``P90`` (pasajeros sin atender, vehículos-día sobrantes y umbral crítico
       ``c_corto / c_sobrante``), además del análisis de temporada alta (diciembre 2023) y
       sensibilidad paramétrica.
    4. Exportar ``reports/metrics/m1_business_summary.json`` y las figuras a 300 dpi:
         - ``fig_m1_06_flota_conductores_30d.png``
         - ``fig_m1_07_sensibilidad_flota.png``
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulo1_demanda.baselines import RidgeCalendarBaseline
from src.modulo1_demanda.nn_data import NNPredictor, load_checkpoint, prepare_context
from src.utils.config import add_config_override_args, load_module_config, resolve_path
from src.utils.seed import set_seed


def _build_route_metadata_map(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    fb = config["fleet_business"]
    caps = fb["vehicle_capacities_passengers"]
    pred_veh = fb.get("route_predominant_vehicle", {})
    cyc_map = fb.get("route_cycle_time_hours", {})
    ow_map = fb.get("route_one_way_hours", {})
    default_veh = str(fb.get("default_vehicle_class", "BUS"))
    default_cyc = float(fb.get("default_cycle_time_hours", 8.0))

    meta_map: Dict[str, Dict[str, Any]] = {}
    for r in config["data"]["selected_routes"]:
        rid = str(r["route_id"])
        veh = str(pred_veh.get(rid, r.get("predominant_vehicle", default_veh)))
        cyc_h = float(cyc_map.get(rid, r.get("cycle_time_hours", default_cyc)))
        ow_h = float(ow_map.get(rid, cyc_h / 2.0))
        cap = float(caps[veh])
        meta_map[rid] = {
            "route_id": rid,
            "route_label": str(r["route_label"]),
            "corridor_id": str(r["corridor_id"]),
            "direction": str(r["direction"]),
            "vehicle_class": veh,
            "capacity_passengers": cap,
            "cycle_time_hours": cyc_h,
            "one_way_hours": ow_h,
        }
    return meta_map


def compute_fleet_plan(
    demand_df: pd.DataFrame,
    config: Mapping[str, Any],
    demand_col: str = "demand",
    date_col: str = "target_date",
    target_occupancy_override: Optional[float] = None,
    demand_multiplier: float = 1.0,
    capacity_multiplier: float = 1.0,
    cycle_time_multiplier: float = 1.0,
    operating_hours_multiplier: float = 1.0,
    reserve_ratio_override: Optional[float] = None,
    absenteeism_rate_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Traduce pronósticos diarios de demanda en despachos, flota acoplada y conductores.

    Args:
        demand_df: DataFrame con al menos ``['route_id', date_col, demand_col]`` (o serie/secuencia
            si se consulta una única ruta).
        config: Diccionario completo cargado con ``load_module_config('m1')``.
        demand_col: Nombre de la columna de demanda pronosticada o real a dimensionar.
        date_col: Nombre de la columna de fecha.
        target_occupancy_override: Sobreescritura opcional de ``target_occupancy_ratio``.
        demand_multiplier: Factor multiplicativo sobre la demanda (ej. ``1.10`` para +10%).
        capacity_multiplier: Factor multiplicativo sobre ``C_r`` (para sensibilidad).
        cycle_time_multiplier: Factor multiplicativo sobre ``cycle_time_hours`` (sensibilidad).
        operating_hours_multiplier: Factor sobre ``operating_hours_per_day`` (sensibilidad).
        reserve_ratio_override: Sobreescritura opcional de ``reserve_fleet_ratio``.

    Returns:
        Diccionario con tablas detalladas por ruta-día, corredor-día, sistema-día y resumen.
    """
    fb = config["fleet_business"]
    meta_map = _build_route_metadata_map(config)

    rho = float(
        target_occupancy_override
        if target_occupancy_override is not None
        else fb["target_occupancy_ratio"]
    )
    rho = min(max(rho, 0.10), 1.00)
    op_hours = float(fb["operating_hours_per_day"]) * float(operating_hours_multiplier)
    reserve_ratio = float(
        reserve_ratio_override if reserve_ratio_override is not None else fb["reserve_fleet_ratio"]
    )

    # Parámetros normativos laborales Colombia 2026
    weekly_ord_h = float(fb["max_weekly_ordinary_hours"])
    workdays_week = float(fb.get("ordinary_workdays_per_week", 6.0))
    daily_ord_h = weekly_ord_h / workdays_week  # 42 / 6 = 7.0 h/día ordinarias
    max_daily_shift_h = float(fb["max_daily_shift_hours"])  # 10.0 h/día tope legal con extras
    second_driver_thresh_h = float(fb["second_driver_threshold_hours"])  # 8.0 h Res. 315/2013
    break_hours = float(fb["mandatory_break_minutes"]) / 60.0
    rest_factor = float(fb["weekly_rest_factor"])  # 7/6
    absenteeism = float(
        absenteeism_rate_override
        if absenteeism_rate_override is not None
        else fb["absenteeism_rate"]
    )
    absenteeism = min(max(absenteeism, 0.0), 0.50)
    roster_multiplier = rest_factor / max(1.0 - absenteeism, 1e-6)

    costs_cfg = fb["costs_cop"]
    dispatch_cost_map = costs_cfg["cost_per_dispatch_by_vehicle_cop"]
    idle_veh_cost_map = costs_cfg["idle_vehicle_day_cost_by_class_cop"]
    driver_shift_cost = float(costs_cfg["driver_daily_shift_cost_cop"])

    df_in = demand_df.copy()
    df_in[date_col] = pd.to_datetime(df_in[date_col])
    df_in["route_id"] = df_in["route_id"].astype(str)

    route_records: List[Dict[str, Any]] = []
    for _, row in df_in.iterrows():
        rid = str(row["route_id"])
        rmeta = meta_map[rid]
        veh_cls = rmeta["vehicle_class"]
        cap_nom = max(1.0, round(rmeta["capacity_passengers"] * float(capacity_multiplier)))
        eff_cap = cap_nom * rho

        cyc_h = rmeta["cycle_time_hours"] * float(cycle_time_multiplier)
        ow_h = rmeta["one_way_hours"] * float(cycle_time_multiplier)

        dem_val = max(0.0, float(row[demand_col]) * float(demand_multiplier))
        dispatches = int(math.ceil(dem_val / eff_cap)) if dem_val > 0 else 0
        offered_seats = float(dispatches * cap_nom)
        effective_seats = float(dispatches * eff_cap)

        # Resolución 315 de 2013 Art. 6: doble conductor si recorrido por sentido > 8.0 h
        crew_per_dispatch = 2 if ow_h > second_driver_thresh_h else 1

        if dispatches == 0:
            active_drivers = 0
            payroll_drivers = 0
        elif crew_per_dispatch == 2:
            # En rutas > 8h por sentido (ej. Bogotá <-> Cali, 12h), los 2 conductores en relevo
            # cubren ese despacho y no pueden iniciar otro recorrido de 12h el mismo día.
            active_drivers = int(2 * dispatches)
            payroll_drivers = int(math.ceil(active_drivers * roster_multiplier))
        else:
            # Tope ordinario semanal Ley 2101/2021 (7.0 h/día promedio) y tope diario 10h con pausas PESV
            max_trips_per_driver_shift = max(1, int(math.floor((max_daily_shift_h - break_hours) / ow_h)))
            drivers_by_daily_cap = int(math.ceil(dispatches / max_trips_per_driver_shift))
            total_driver_hours = dispatches * (ow_h + break_hours * (ow_h / max_daily_shift_h))
            drivers_by_ordinary_42h = int(math.ceil(total_driver_hours / daily_ord_h))
            active_drivers = max(drivers_by_daily_cap, drivers_by_ordinary_42h)
            payroll_drivers = int(math.ceil(active_drivers * roster_multiplier))

        route_records.append(
            {
                date_col: row[date_col],
                "route_id": rid,
                "route_label": rmeta["route_label"],
                "corridor_id": rmeta["corridor_id"],
                "direction": rmeta["direction"],
                "vehicle_class": veh_cls,
                "nominal_capacity": int(cap_nom),
                "effective_capacity": round(eff_cap, 2),
                "cycle_time_hours": round(cyc_h, 3),
                "one_way_hours": round(ow_h, 3),
                "demand_planned": round(dem_val, 2),
                "dispatches": dispatches,
                "physical_seats_offered": offered_seats,
                "target_seats_offered": round(effective_seats, 2),
                "crew_per_dispatch": crew_per_dispatch,
                "active_drivers_day": active_drivers,
                "payroll_drivers_required": payroll_drivers,
                "dispatch_operating_cost_cop": float(dispatches * float(dispatch_cost_map[veh_cls])),
                "driver_cost_cop": float(active_drivers * driver_shift_cost),
                "idle_unit_cost_cop": float(idle_veh_cost_map[veh_cls]),
            }
        )

    df_route_plan = pd.DataFrame(route_records).sort_values([date_col, "corridor_id", "direction"]).reset_index(drop=True)

    # Acoplamiento bidireccional por corredor y fecha
    corridor_records: List[Dict[str, Any]] = []
    for (dt_val, corr_id), grp in df_route_plan.groupby([date_col, "corridor_id"], sort=False):
        cyc_h = float(grp["cycle_time_hours"].iloc[0])
        veh_cls = str(grp["vehicle_class"].iloc[0])
        idle_unit_cost = float(grp["idle_unit_cost_cop"].iloc[0])

        d_ida_s = grp.loc[grp["direction"] == "ida", "dispatches"]
        d_vta_s = grp.loc[grp["direction"] == "vuelta", "dispatches"]
        d_ida = int(d_ida_s.iloc[0]) if not d_ida_s.empty else 0
        d_vta = int(d_vta_s.iloc[0]) if not d_vta_s.empty else 0
        total_dir_dispatches = int(grp["dispatches"].sum())
        balanced_round_trips = max(d_ida, d_vta)

        # Regla física de acoplamiento de flota en corredor A <-> B:
        # - Si el ciclo es interdiario (>= 20h, ej. Bogotá <-> Cali 24h): los buses que salen hoy
        #   de A->B duermen/retornan al día siguiente desde B->A, compartiendo flota a razón de
        #   1 despacho/día por vehículo asignado (escalado por 24.0 / cyc_h ante perturbaciones).
        # - Si el ciclo cabe en el día (cyc_h <= op_hours): cada vehículo completa V_c = op_hours / cyc_h
        #   vueltas redondas por día cubriendo ambos sentidos (max(D_AB, D_BA) ciclos).
        if len(grp) == 1:
            # Inferencia aislada de una sola ruta dirigida en UI
            one_way_h = float(grp["one_way_hours"].iloc[0])
            trips_per_veh = (24.0 / cyc_h) if cyc_h > op_hours else max(1.0, op_hours / cyc_h)
            vueltas_veh_dia = round(trips_per_veh, 3)
            op_fleet = int(math.ceil(total_dir_dispatches / trips_per_veh)) if total_dir_dispatches > 0 else 0
            del one_way_h
        elif cyc_h > op_hours:
            dispatches_per_veh_day = 24.0 / cyc_h  # Nominal 24h -> 1.0 despacho/día por vehículo
            vueltas_veh_dia = round(0.5 * dispatches_per_veh_day, 3)
            op_fleet = int(math.ceil(total_dir_dispatches / dispatches_per_veh_day)) if total_dir_dispatches > 0 else 0
        else:
            vueltas_eff = op_hours / cyc_h  # Vueltas redondas posibles en la ventana operativa
            vueltas_veh_dia = round(vueltas_eff, 3)
            op_fleet = int(math.ceil(balanced_round_trips / vueltas_eff)) if balanced_round_trips > 0 else 0

        total_fleet_with_reserve = int(math.ceil(op_fleet * (1.0 + reserve_ratio))) if op_fleet > 0 else 0
        reserve_vehicles = total_fleet_with_reserve - op_fleet

        corridor_records.append(
            {
                date_col: dt_val,
                "corridor_id": str(corr_id),
                "vehicle_class": veh_cls,
                "cycle_time_hours": cyc_h,
                "vueltas_por_vehiculo_dia": vueltas_veh_dia,
                "dispatches_ida": d_ida,
                "dispatches_vuelta": d_vta,
                "dispatches_total": total_dir_dispatches,
                "operational_fleet": op_fleet,
                "reserve_vehicles": reserve_vehicles,
                "total_fleet_with_reserve": total_fleet_with_reserve,
                "active_drivers_day": int(grp["active_drivers_day"].sum()),
                "payroll_drivers_required": int(grp["payroll_drivers_required"].sum()),
                "demand_planned_total": round(float(grp["demand_planned"].sum()), 2),
                "idle_unit_cost_cop": idle_unit_cost,
            }
        )

    df_corridor_plan = pd.DataFrame(corridor_records).sort_values([date_col, "corridor_id"]).reset_index(drop=True)

    df_system_daily = (
        df_corridor_plan.groupby(date_col, as_index=False)
        .agg(
            demand_planned_total=("demand_planned_total", "sum"),
            dispatches_total=("dispatches_total", "sum"),
            operational_fleet_total=("operational_fleet", "sum"),
            reserve_vehicles_total=("reserve_vehicles", "sum"),
            total_fleet_with_reserve=("total_fleet_with_reserve", "sum"),
            active_drivers_day_total=("active_drivers_day", "sum"),
            payroll_drivers_total=("payroll_drivers_required", "sum"),
        )
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    # Resumen agregado por corredor (media diaria y pico diario en el horizonte evaluado)
    corridor_summary: Dict[str, Any] = {}
    for corr_id, grp in df_corridor_plan.groupby("corridor_id", sort=False):
        corridor_summary[str(corr_id)] = {
            "vehicle_class": str(grp["vehicle_class"].iloc[0]),
            "cycle_time_hours": float(grp["cycle_time_hours"].iloc[0]),
            "vueltas_por_vehiculo_dia": float(grp["vueltas_por_vehiculo_dia"].iloc[0]),
            "mean_daily_dispatches": round(float(grp["dispatches_total"].mean()), 2),
            "peak_daily_dispatches": int(grp["dispatches_total"].max()),
            "mean_operational_fleet": round(float(grp["operational_fleet"].mean()), 2),
            "peak_operational_fleet": int(grp["operational_fleet"].max()),
            "mean_total_fleet_with_reserve": round(float(grp["total_fleet_with_reserve"].mean()), 2),
            "peak_total_fleet_with_reserve": int(grp["total_fleet_with_reserve"].max()),
            "mean_active_drivers_day": round(float(grp["active_drivers_day"].mean()), 2),
            "peak_active_drivers_day": int(grp["active_drivers_day"].max()),
            "mean_payroll_drivers_required": round(float(grp["payroll_drivers_required"].mean()), 2),
            "peak_payroll_drivers_required": int(grp["payroll_drivers_required"].max()),
        }

    route_summary: Dict[str, Any] = {}
    for rid, grp in df_route_plan.groupby("route_id", sort=False):
        route_summary[str(rid)] = {
            "route_label": str(grp["route_label"].iloc[0]),
            "vehicle_class": str(grp["vehicle_class"].iloc[0]),
            "nominal_capacity": int(grp["nominal_capacity"].iloc[0]),
            "crew_per_dispatch": int(grp["crew_per_dispatch"].iloc[0]),
            "mean_daily_demand": round(float(grp["demand_planned"].mean()), 2),
            "mean_daily_dispatches": round(float(grp["dispatches"].mean()), 2),
            "peak_daily_dispatches": int(grp["dispatches"].max()),
            "mean_active_drivers_day": round(float(grp["active_drivers_day"].mean()), 2),
            "peak_active_drivers_day": int(grp["active_drivers_day"].max()),
            "mean_payroll_drivers_required": round(float(grp["payroll_drivers_required"].mean()), 2),
            "peak_payroll_drivers_required": int(grp["payroll_drivers_required"].max()),
        }

    system_summary = {
        "n_days": int(len(df_system_daily)),
        "target_occupancy_ratio": round(rho, 4),
        "demand_multiplier": round(float(demand_multiplier), 4),
        "reserve_fleet_ratio": round(reserve_ratio, 4),
        "absenteeism_rate": round(absenteeism, 4),
        "mean_daily_dispatches": round(float(df_system_daily["dispatches_total"].mean()), 2),
        "peak_daily_dispatches": int(df_system_daily["dispatches_total"].max()),
        "mean_operational_fleet": round(float(df_system_daily["operational_fleet_total"].mean()), 2),
        "p90_operational_fleet": round(float(df_system_daily["operational_fleet_total"].quantile(0.90)), 2),
        "peak_operational_fleet": int(df_system_daily["operational_fleet_total"].max()),
        "mean_total_fleet_with_reserve": round(float(df_system_daily["total_fleet_with_reserve"].mean()), 2),
        "p90_total_fleet_with_reserve": round(
            float(df_system_daily["total_fleet_with_reserve"].quantile(0.90)), 2
        ),
        "peak_total_fleet_with_reserve": int(df_system_daily["total_fleet_with_reserve"].max()),
        "mean_active_drivers_day": round(float(df_system_daily["active_drivers_day_total"].mean()), 2),
        "peak_active_drivers_day": int(df_system_daily["active_drivers_day_total"].max()),
        "mean_payroll_drivers_total": round(float(df_system_daily["payroll_drivers_total"].mean()), 2),
        "peak_payroll_drivers_total": int(df_system_daily["payroll_drivers_total"].max()),
    }

    return {
        "daily_route_plan": df_route_plan,
        "daily_corridor_plan": df_corridor_plan,
        "daily_system_totals": df_system_daily,
        "by_corridor_summary": corridor_summary,
        "by_route_summary": route_summary,
        "system_summary": system_summary,
    }


def _generate_chained_test_forecasts(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Genera pronósticos sin solapamiento cubriendo los 184 días únicos de test (2023-07-01..2023-12-31).

    Encadena orígenes cada 30 días (``2023-06-30``, ``2023-07-30``, ..., ``2023-11-27``) y completa
    los últimos 4 días (``2023-12-28..2023-12-31``) desde el origen ``2023-12-01`` para cubrir
    exactamente el semestre completo de prueba sin duplicar fechas calendario.
    """
    ctx = prepare_context(config, with_train_windows=False)
    arrays = ctx["arrays"]
    df_all = arrays.df
    horizon = int(ctx["horizon"])
    q = float(ctx["quantile"])

    # 1) Modelo GRU (checkpoint entrenado)
    gru_ckpt_path = resolve_path(config["paths"]["model_dir"]) / "gru_best.pt"
    gru_model, gru_ckpt = load_checkpoint(gru_ckpt_path)
    gru_pred = NNPredictor(
        gru_model,
        arrays,
        gru_ckpt["lookback"],
        gru_ckpt["horizon"],
        gru_ckpt["scale_floor_frac"],
        np.asarray(gru_ckpt["route_train_mean"], dtype=np.float32),
    )

    # 2) Modelo Ridge + QuantileRegressor (entrenado en train con el alpha seleccionado en val)
    bl_path = resolve_path(config["paths"]["baselines_metrics_file"])
    alpha = 1.0
    if bl_path.exists():
        alpha = float(
            json.loads(bl_path.read_text(encoding="utf-8"))["models"]["ridge_calendar"][
                "selected_ridge_alpha_on_val"
            ]
        )
    base_cfg = config.get("baselines", {})
    df_train = df_all[df_all["split"] == "train"].copy()
    ridge_pred = RidgeCalendarBaseline(
        ridge_alpha=alpha,
        quantile_alpha=float(base_cfg.get("quantile_alpha", 0.001)),
        quantile=q,
        include_interactions=bool(base_cfg.get("include_route_calendar_interactions", True)),
    ).fit(df_train)

    test_start = pd.Timestamp(config["data"]["split_dates"]["test_start"])
    test_end = pd.Timestamp(config["data"]["split_dates"]["test_end"])

    chained_origins: List[pd.Timestamp] = []
    cur = test_start - pd.Timedelta(days=1)
    while cur + pd.Timedelta(days=horizon) <= test_end:
        chained_origins.append(cur)
        cur += pd.Timedelta(days=horizon)
    tail_origin = test_end - pd.Timedelta(days=horizon)

    out: Dict[str, pd.DataFrame] = {}
    for name, predictor in (("gru", gru_pred), ("ridge", ridge_pred)):
        chunks = [predictor.predict_from_origin(df_all, orig, horizon) for orig in chained_origins]
        covered_max = chunks[-1]["target_date"].max()
        if covered_max < test_end:
            tail_df = predictor.predict_from_origin(df_all, tail_origin, horizon)
            chunks.append(tail_df[tail_df["target_date"] > covered_max])
        full_test_pred = (
            pd.concat(chunks, ignore_index=True)
            .sort_values(["target_date", "route_id"])
            .reset_index(drop=True)
        )
        out[name] = full_test_pred

    # También devolver la ventana pura de 30 días desde forecast_30d_origin_date (ej. 2023-12-01)
    dec_origin = pd.Timestamp(config["fleet_business"]["analysis"]["forecast_30d_origin_date"])
    out["gru_window_30d"] = gru_pred.predict_from_origin(df_all, dec_origin, horizon)
    return out


def _evaluate_forecast_economic_value(
    preds_by_model: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Cuantifica pasajeros sin atender, vehículos-día sobrantes y costos en los 4 escenarios."""
    fb = config["fleet_business"]
    costs_cfg = fb["costs_cop"]
    c_short_default = float(costs_cfg["unmet_passenger_penalty_cop"])
    c_surplus_ref = float(costs_cfg["idle_vehicle_day_cost_reference_cop"])
    ratio_grid: List[float] = [float(r) for r in costs_cfg["shortage_to_surplus_cost_ratios"]]

    # Plan óptimo ex-post (oráculo con demanda real observada y_true)
    ref_df = preds_by_model["gru"][["route_id", "target_date", "y_true"]].copy()
    oracle_plan = compute_fleet_plan(ref_df, config, demand_col="y_true", date_col="target_date")
    oracle_corr = oracle_plan["daily_corridor_plan"][
        ["target_date", "corridor_id", "operational_fleet"]
    ].rename(columns={"operational_fleet": "oracle_op_fleet"})
    total_real_passengers = float(ref_df["y_true"].sum())

    scenarios = [
        ("Ridge Puntual", "ridge", "y_pred_p50"),
        ("Ridge P90", "ridge", "y_pred_p90"),
        ("GRU Puntual", "gru", "y_pred_p50"),
        ("GRU P90", "gru", "y_pred_p90"),
    ]

    scenario_metrics: Dict[str, Any] = {}
    for label, m_key, col_pred in scenarios:
        df_m = preds_by_model[m_key].copy()
        plan = compute_fleet_plan(df_m, config, demand_col=col_pred, date_col="target_date")

        # 1) Pasajeros sin atender: demanda real por encima de las sillas físicas despachadas
        r_plan = pd.merge(
            plan["daily_route_plan"],
            df_m[["route_id", "target_date", "y_true"]],
            on=["route_id", "target_date"],
            how="left",
        )
        unmet_physical = np.maximum(0.0, r_plan["y_true"] - r_plan["physical_seats_offered"])
        unmet_target_occ = np.maximum(0.0, r_plan["y_true"] - r_plan["target_seats_offered"])

        # 2) Vehículos-día sobrantes frente a la flota operativa ex-post necesaria
        c_plan = pd.merge(
            plan["daily_corridor_plan"],
            oracle_corr,
            on=["target_date", "corridor_id"],
            how="left",
        )
        surplus_veh_days = np.maximum(0, c_plan["operational_fleet"] - c_plan["oracle_op_fleet"])
        deficit_veh_days = np.maximum(0, c_plan["oracle_op_fleet"] - c_plan["operational_fleet"])

        # Costo exacto de sobrantes valorado según clase vehicular de cada corredor + penalidad faltantes
        surplus_cost_cop = float(np.sum(surplus_veh_days * c_plan["idle_unit_cost_cop"]))
        shortage_cost_cop = float(np.sum(unmet_physical) * c_short_default)
        total_mismatch_cost_cop = surplus_cost_cop + shortage_cost_cop

        tot_unmet = int(round(float(np.sum(unmet_physical))))
        tot_surplus = int(np.sum(surplus_veh_days))

        # Costo relativo normalizado para barrido de ratios r = c_corto / c_sobrante
        cost_curve_in_surplus_units = {
            f"{r:.2f}": round(tot_surplus + r * tot_unmet, 2) for r in ratio_grid
        }

        scenario_metrics[label] = {
            "total_real_passengers_test": int(round(total_real_passengers)),
            "unmet_passengers_physical": tot_unmet,
            "unmet_passengers_pct_of_demand": round(100.0 * tot_unmet / total_real_passengers, 3),
            "passengers_above_target_85pct_occ": int(round(float(np.sum(unmet_target_occ)))),
            "surplus_vehicle_days": tot_surplus,
            "deficit_vehicle_days": int(np.sum(deficit_veh_days)),
            "mean_daily_operational_fleet": plan["system_summary"]["mean_operational_fleet"],
            "mean_daily_total_fleet_with_reserve": plan["system_summary"][
                "mean_total_fleet_with_reserve"
            ],
            "surplus_cost_cop": round(surplus_cost_cop, 0),
            "shortage_cost_cop": round(shortage_cost_cop, 0),
            "total_mismatch_cost_cop": round(total_mismatch_cost_cop, 0),
            "normalized_cost_by_ratio_c_short_over_c_surplus": cost_curve_in_surplus_units,
        }

    # Calcular umbral crítico exacto r* = (c_corto / c_sobrante)* donde P90 iguala al Puntual
    thresholds: Dict[str, Any] = {}
    for prefix in ("GRU", "Ridge"):
        u_pt = scenario_metrics[f"{prefix} Puntual"]["unmet_passengers_physical"]
        u_p90 = scenario_metrics[f"{prefix} P90"]["unmet_passengers_physical"]
        s_pt = scenario_metrics[f"{prefix} Puntual"]["surplus_vehicle_days"]
        s_p90 = scenario_metrics[f"{prefix} P90"]["surplus_vehicle_days"]
        delta_u = max(u_pt - u_p90, 1)
        delta_s = max(s_p90 - s_pt, 0)
        r_star = delta_s / delta_u
        thresholds[prefix] = {
            "unmet_passengers_avoided_by_p90": int(delta_u),
            "extra_surplus_vehicle_days_of_p90": int(delta_s),
            "critical_ratio_c_corto_over_c_sobrante": round(r_star, 4),
            "breakeven_c_corto_cop_at_ref_surplus_cost": round(r_star * c_surplus_ref, 0),
            "configured_ratio_at_yaml_costs": round(c_short_default / c_surplus_ref, 4),
            "p90_recommended_under_yaml_costs": bool((c_short_default / c_surplus_ref) > r_star),
        }

    # Ahorro neto GRU vs Ridge en COP sobre el semestre de prueba
    gru_vs_ridge = {
        "puntual_mismatch_savings_cop": round(
            scenario_metrics["Ridge Puntual"]["total_mismatch_cost_cop"]
            - scenario_metrics["GRU Puntual"]["total_mismatch_cost_cop"],
            0,
        ),
        "p90_mismatch_savings_cop": round(
            scenario_metrics["Ridge P90"]["total_mismatch_cost_cop"]
            - scenario_metrics["GRU P90"]["total_mismatch_cost_cop"],
            0,
        ),
        "p90_surplus_vehicle_days_saved_by_gru": int(
            scenario_metrics["Ridge P90"]["surplus_vehicle_days"]
            - scenario_metrics["GRU P90"]["surplus_vehicle_days"]
        ),
        "p90_unmet_passengers_diff_gru_minus_ridge": int(
            scenario_metrics["GRU P90"]["unmet_passengers_physical"]
            - scenario_metrics["Ridge P90"]["unmet_passengers_physical"]
        ),
    }

    return {
        "scenarios_test_184d": scenario_metrics,
        "p90_vs_point_thresholds": thresholds,
        "gru_vs_ridge_operational_justification": gru_vs_ridge,
    }


def _run_sensitivity_analysis(
    base_pred_df: pd.DataFrame,
    config: Mapping[str, Any],
    demand_col: str = "y_pred_p90",
) -> Dict[str, Any]:
    """Sensibilidad de flota ante +-10% de demanda y perturbaciones +-10% de parámetros operativos."""
    an_cfg = config["fleet_business"]["analysis"]
    dem_deltas = [float(d) for d in an_cfg["demand_sensitivity_deltas"]]
    p_delta = float(an_cfg["parameter_perturbation_delta"])
    fb = config["fleet_business"]

    base_plan = compute_fleet_plan(base_pred_df, config, demand_col=demand_col)
    base_mean_fleet = float(base_plan["system_summary"]["mean_total_fleet_with_reserve"])
    base_peak_fleet = int(base_plan["system_summary"]["peak_total_fleet_with_reserve"])

    demand_curve: List[Dict[str, Any]] = []
    for d in dem_deltas:
        mult = 1.0 + d
        p = compute_fleet_plan(
            base_pred_df, config, demand_col=demand_col, demand_multiplier=mult
        )
        m_fl = float(p["system_summary"]["mean_total_fleet_with_reserve"])
        pk_fl = int(p["system_summary"]["peak_total_fleet_with_reserve"])
        demand_curve.append(
            {
                "demand_delta_pct": round(100.0 * d, 1),
                "demand_multiplier": round(mult, 3),
                "mean_total_fleet": round(m_fl, 2),
                "mean_fleet_change_pct": round(100.0 * (m_fl - base_mean_fleet) / base_mean_fleet, 2),
                "peak_total_fleet": pk_fl,
                "mean_payroll_drivers": p["system_summary"]["mean_payroll_drivers_total"],
            }
        )

    # Perturbación individual +-10% de cada parámetro operativo clave (análisis Tornado)
    base_rho = float(fb["target_occupancy_ratio"])
    base_res = float(fb["reserve_fleet_ratio"])

    param_specs = [
        (
            "Capacidad vehicular (C_r)",
            dict(capacity_multiplier=1.0 - p_delta),
            dict(capacity_multiplier=1.0 + p_delta),
        ),
        (
            "Ocupación objetivo (rho)",
            dict(target_occupancy_override=base_rho * (1.0 - p_delta)),
            dict(target_occupancy_override=min(1.0, base_rho * (1.0 + p_delta))),
        ),
        (
            "Tiempo de ciclo (T_ciclo)",
            dict(cycle_time_multiplier=1.0 - p_delta),
            dict(cycle_time_multiplier=1.0 + p_delta),
        ),
        (
            "Horas operación terminal (H_op)",
            dict(operating_hours_multiplier=1.0 - p_delta),
            dict(operating_hours_multiplier=1.0 + p_delta),
        ),
        (
            "Fracción reserva técnica",
            dict(reserve_ratio_override=max(0.0, base_res * (1.0 - p_delta))),
            dict(reserve_ratio_override=base_res * (1.0 + p_delta)),
        ),
    ]

    tornado_rows: List[Dict[str, Any]] = []
    for p_name, kw_minus, kw_plus in param_specs:
        fl_minus = float(
            compute_fleet_plan(base_pred_df, config, demand_col=demand_col, **kw_minus)[
                "system_summary"
            ]["mean_total_fleet_with_reserve"]
        )
        fl_plus = float(
            compute_fleet_plan(base_pred_df, config, demand_col=demand_col, **kw_plus)[
                "system_summary"
            ]["mean_total_fleet_with_reserve"]
        )
        swing = abs(fl_plus - fl_minus)
        tornado_rows.append(
            {
                "parameter": p_name,
                "fleet_at_minus_10pct": round(fl_minus, 2),
                "fleet_at_plus_10pct": round(fl_plus, 2),
                "delta_minus_10pct_vehicles": round(fl_minus - base_mean_fleet, 2),
                "delta_plus_10pct_vehicles": round(fl_plus - base_mean_fleet, 2),
                "total_swing_vehicles": round(swing, 2),
                "total_swing_pct": round(100.0 * swing / base_mean_fleet, 2),
            }
        )

    tornado_rows.sort(key=lambda r: r["total_swing_vehicles"], reverse=True)
    most_influential = tornado_rows[0]["parameter"]

    # Curva fina para el parámetro más influyente (ej. barrido de ocupación objetivo rho: 0.70..0.95)
    occ_grid = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    occ_sweep = []
    for r_val in occ_grid:
        p = compute_fleet_plan(
            base_pred_df, config, demand_col=demand_col, target_occupancy_override=r_val
        )
        occ_sweep.append(
            {
                "target_occupancy_ratio": r_val,
                "mean_total_fleet": p["system_summary"]["mean_total_fleet_with_reserve"],
                "peak_total_fleet": p["system_summary"]["peak_total_fleet_with_reserve"],
            }
        )

    return {
        "baseline_mean_total_fleet": round(base_mean_fleet, 2),
        "baseline_peak_total_fleet": base_peak_fleet,
        "demand_sensitivity_curve": demand_curve,
        "parameter_tornado_pm10pct": tornado_rows,
        "most_influential_parameter": most_influential,
        "occupancy_ratio_sweep": occ_sweep,
    }


def _analyze_peak_season(
    gru_test_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compara semana pico de diciembre 2023 vs semana normal y evalúa esquemas de flota."""
    fb = config["fleet_business"]
    an_cfg = fb["analysis"]
    costs_cfg = fb["costs_cop"]
    sourcing_cfg = an_cfg["peak_sourcing_models"]

    pk_start = pd.Timestamp(an_cfg["peak_week_start"])
    pk_end = pd.Timestamp(an_cfg["peak_week_end"])
    nm_start = pd.Timestamp(an_cfg["normal_week_start"])
    nm_end = pd.Timestamp(an_cfg["normal_week_end"])

    df_pk = gru_test_df[
        (gru_test_df["target_date"] >= pk_start) & (gru_test_df["target_date"] <= pk_end)
    ]
    df_nm = gru_test_df[
        (gru_test_df["target_date"] >= nm_start) & (gru_test_df["target_date"] <= nm_end)
    ]

    # Dimensionar con GRU P90 (política de nivel de servicio seguro en temporada alta)
    plan_pk = compute_fleet_plan(df_pk, config, demand_col="y_pred_p90")
    plan_nm = compute_fleet_plan(df_nm, config, demand_col="y_pred_p90")

    by_corridor_delta: Dict[str, Any] = {}
    extra_buses = 0
    extra_microbuses = 0
    extra_peak_dispatches_per_day = 0.0

    for corr_id, pk_c in plan_pk["by_corridor_summary"].items():
        nm_c = plan_nm["by_corridor_summary"][corr_id]
        add_fleet_peak = int(pk_c["peak_total_fleet_with_reserve"] - nm_c["mean_total_fleet_with_reserve"])
        add_fleet_peak = max(0, add_fleet_peak)
        add_drivers_peak = max(
            0, int(pk_c["peak_payroll_drivers_required"] - nm_c["mean_payroll_drivers_required"])
        )
        veh_cls = pk_c["vehicle_class"]
        if veh_cls == "BUS":
            extra_buses += add_fleet_peak
        else:
            extra_microbuses += add_fleet_peak
        extra_peak_dispatches_per_day += max(
            0.0, pk_c["mean_daily_dispatches"] - nm_c["mean_daily_dispatches"]
        )

        by_corridor_delta[corr_id] = {
            "vehicle_class": veh_cls,
            "normal_week_mean_fleet": nm_c["mean_total_fleet_with_reserve"],
            "peak_week_mean_fleet": pk_c["mean_total_fleet_with_reserve"],
            "peak_week_max_fleet": pk_c["peak_total_fleet_with_reserve"],
            "additional_vehicles_needed_at_peak": add_fleet_peak,
            "normal_week_mean_payroll_drivers": nm_c["mean_payroll_drivers_required"],
            "peak_week_max_payroll_drivers": pk_c["peak_payroll_drivers_required"],
            "additional_drivers_needed_at_peak": add_drivers_peak,
        }

    # Evaluación económica anualizada de cubrir el bloque adicional de temporada alta (~52 días pico/año)
    peak_days_yr = int(an_cfg["estimated_peak_days_per_year"])
    own_fixed_bus = float(sourcing_cfg["own_vehicle_annual_fixed_cost_cop"]["BUS"])
    own_fixed_micro = float(sourcing_cfg["own_vehicle_annual_fixed_cost_cop"]["MICROBUS"])
    lease_mult = float(sourcing_cfg["leased_daily_cost_multiplier"])
    comm_rate = float(sourcing_cfg["intercompany_commission_rate"])
    avg_ticket = float(costs_cfg["average_ticket_cop"])

    idle_bus = float(costs_cfg["idle_vehicle_day_cost_by_class_cop"]["BUS"])
    idle_micro = float(costs_cfg["idle_vehicle_day_cost_by_class_cop"]["MICROBUS"])

    # Opción 1: Flota 100% propia mantenida todo el año para cubrir los picos
    annual_cost_own_peak_fleet = extra_buses * own_fixed_bus + extra_microbuses * own_fixed_micro

    # Opción 2: Flota arrendada/fletada únicamente durante los días pico del año
    annual_cost_leased_peak = peak_days_yr * lease_mult * (
        extra_buses * idle_bus + extra_microbuses * idle_micro
    )

    # Opción 3: Convenios de colaboración empresarial (comisión sobre pasajes excedentes cedidos)
    avg_pax_per_extra_disp = 28.0
    annual_cost_intercompany = (
        peak_days_yr * extra_peak_dispatches_per_day * avg_pax_per_extra_disp * avg_ticket * comm_rate
    )

    return {
        "normal_week_window": f"{an_cfg['normal_week_start']} a {an_cfg['normal_week_end']}",
        "peak_week_window": f"{an_cfg['peak_week_start']} a {an_cfg['peak_week_end']}",
        "system_normal_week": plan_nm["system_summary"],
        "system_peak_week": plan_pk["system_summary"],
        "additional_vehicles_at_december_peak": {
            "BUS": extra_buses,
            "MICROBUS": extra_microbuses,
            "total_extra_vehicles": extra_buses + extra_microbuses,
            "total_extra_payroll_drivers": int(
                plan_pk["system_summary"]["peak_payroll_drivers_total"]
                - round(plan_nm["system_summary"]["mean_payroll_drivers_total"])
            ),
        },
        "by_corridor_comparison": by_corridor_delta,
        "sourcing_alternatives_annual_cost_cop": {
            "assumed_peak_days_per_year": peak_days_yr,
            "1_own_fleet_for_peak_cop": round(annual_cost_own_peak_fleet, 0),
            "2_seasonal_leased_fleet_cop": round(annual_cost_leased_peak, 0),
            "3_intercompany_collaboration_agreement_cop": round(annual_cost_intercompany, 0),
            "recommended_hybrid_strategy": (
                "Cubrir la demanda base (P50 valle) con flota propia, el escalón estacional "
                "sostenido de diciembre/mitad de año con fletamento/arrendamiento temporal y "
                "los picos extremos de 3-5 días (22-24 y 29-30 dic) mediante convenios de "
                "colaboración empresarial (Decreto 1079 de 2015)."
            ),
        },
    }


def _plot_fleet_30d(
    window_30d_df: pd.DataFrame,
    config: Mapping[str, Any],
    out_path: Path,
) -> None:
    """Genera fig_m1_06_flota_conductores_30d.png (diciembre 2023, Puntual vs P90 vs Real)."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    dpi = int(config["eda"]["dpi"])

    plan_p50 = compute_fleet_plan(window_30d_df, config, demand_col="y_pred_p50")
    plan_p90 = compute_fleet_plan(window_30d_df, config, demand_col="y_pred_p90")
    plan_real = compute_fleet_plan(window_30d_df, config, demand_col="y_true")

    s_p50 = plan_p50["daily_system_totals"]
    s_p90 = plan_p90["daily_system_totals"]
    s_real = plan_real["daily_system_totals"]

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)

    # Panel superior: Flota total requerida (con reserva del 10%)
    ax0 = axes[0]
    ax0.fill_between(
        s_p50["target_date"],
        s_p50["total_fleet_with_reserve"],
        s_p90["total_fleet_with_reserve"],
        color="#1b9e77",
        alpha=0.22,
        label="Colchón de flota P50–P90",
    )
    ax0.plot(
        s_real["target_date"],
        s_real["total_fleet_with_reserve"],
        color="#222222",
        linestyle=":",
        linewidth=1.4,
        marker="o",
        markersize=3.5,
        label="Flota ex-post requerida (Demanda Real)",
    )
    ax0.plot(
        s_p50["target_date"],
        s_p50["total_fleet_with_reserve"],
        color="#2b8cbe",
        linewidth=1.8,
        marker="s",
        markersize=3.5,
        label="Plan GRU Puntual (P50)",
    )
    ax0.plot(
        s_p90["target_date"],
        s_p90["total_fleet_with_reserve"],
        color="#1b9e77",
        linewidth=2.0,
        marker="^",
        markersize=4.0,
        label="Plan GRU Conservador (P90)",
    )
    ax0.set_ylabel("Vehículos totales (operativos + 10% reserva)")
    ax0.set_title(
        "A. Flota Total Requerida Día a Día (4 Corredores Acoplados, Horizonte 30 Días — Dic 2023)",
        fontweight="bold",
    )
    ax0.legend(loc="upper left", ncol=2, frameon=True)

    # Panel inferior: Conductores en turno vs Plantilla contratada (Ley 2101/2021 + Res. 315/2013)
    ax1 = axes[1]
    ax1.plot(
        s_p50["target_date"],
        s_p50["active_drivers_day_total"],
        color="#2b8cbe",
        linestyle="--",
        linewidth=1.5,
        label="Conductores activos en turno/día (P50)",
    )
    ax1.plot(
        s_p90["target_date"],
        s_p90["active_drivers_day_total"],
        color="#1b9e77",
        linestyle="--",
        linewidth=1.6,
        label="Conductores activos en turno/día (P90)",
    )
    ax1.plot(
        s_p50["target_date"],
        s_p50["payroll_drivers_total"],
        color="#08519c",
        linewidth=1.9,
        label="Plantilla total requerida c/ descanso 7/6 + ausentismo (P50)",
    )
    ax1.plot(
        s_p90["target_date"],
        s_p90["payroll_drivers_total"],
        color="#d95f02",
        linewidth=2.1,
        label="Plantilla total requerida c/ descanso 7/6 + ausentismo (P90)",
    )
    ax1.set_ylabel("Número de conductores")
    ax1.set_xlabel("Fecha objetivo del horizonte de 30 días (origen $t_0$ = 2023-12-01)")
    ax1.set_title(
        "B. Conductores Requeridos (Jornada 42 h/sem Ley 2101/2021 + Relevo >8h Res. 315/2013)",
        fontweight="bold",
    )
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.tick_params(axis="x", rotation=25)
    ax1.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.5)

    fig.suptitle(
        "Planificación Operativa a 30 Días (Diciembre 2023): Flota y Conductores (Puntual vs. P90)",
        fontsize=13,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_fleet_sensitivity(
    sens: Mapping[str, Any],
    config: Mapping[str, Any],
    out_path: Path,
) -> None:
    """Genera fig_m1_07_sensibilidad_flota.png: curva +-10% demanda y Tornado + barrido ocupación."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    dpi = int(config["eda"]["dpi"])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    # Panel izquierdo: Sensibilidad ante variación de demanda (-10% a +10%) y barrido de ocupación
    ax0 = axes[0]
    dem_df = pd.DataFrame(sens["demand_sensitivity_curve"])
    ax0.plot(
        dem_df["demand_delta_pct"],
        dem_df["mean_total_fleet"],
        marker="o",
        linewidth=2.2,
        color="#1f4e78",
        label="Flota media diaria (con reserva)",
    )
    ax0.plot(
        dem_df["demand_delta_pct"],
        dem_df["peak_total_fleet"],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color="#d95f02",
        label="Flota pico diaria en Test",
    )
    for _, r in dem_df.iterrows():
        ax0.annotate(
            f"{r['mean_total_fleet']:.1f}",
            (r["demand_delta_pct"], r["mean_total_fleet"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
        )
    ax0.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax0.set_xlabel("Variación porcentual de la demanda pronosticada (%)")
    ax0.set_ylabel("Vehículos requeridos (sistema 4 corredores)")
    ax0.set_title("A. Sensibilidad de la Flota ante ±10% de Demanda (GRU P90)", fontweight="bold")
    ax0.legend(loc="upper left")

    # Panel derecho: Diagrama Tornado de perturbación +-10% de cada parámetro operativo
    ax1 = axes[1]
    tor_df = pd.DataFrame(sens["parameter_tornado_pm10pct"]).iloc[::-1]  # Mayor impacto arriba
    y_pos = np.arange(len(tor_df))

    ax1.barh(
        y_pos - 0.17,
        tor_df["delta_minus_10pct_vehicles"],
        height=0.34,
        color="#e08214",
        label="Parámetro -10%",
    )
    ax1.barh(
        y_pos + 0.17,
        tor_df["delta_plus_10pct_vehicles"],
        height=0.34,
        color="#2b8cbe",
        label="Parámetro +10%",
    )
    ax1.axvline(0.0, color="#222222", linewidth=1.1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tor_df["parameter"])
    ax1.set_xlabel(
        f"Cambio en flota media diaria respecto a base ({sens['baseline_mean_total_fleet']:.1f} veh)"
    )
    ax1.set_title("B. Impacto Relativo por Parámetro Operativo (Tornado ±10%)", fontweight="bold")
    ax1.legend(loc="lower right")

    fig.suptitle(
        "Análisis de Sensibilidad del Dimensionamiento de Flota (Test 2023-S2)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_business_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta el análisis integral de las 4 preguntas de negocio sobre Test y guarda artefactos."""
    preds = _generate_chained_test_forecasts(config)
    gru_test_df = preds["gru"]
    gru_30d_df = preds["gru_window_30d"]

    # P1 & P2: Dimensionamiento de flota y conductores en Test (GRU Puntual y GRU P90)
    plan_gru_p50 = compute_fleet_plan(gru_test_df, config, demand_col="y_pred_p50")
    plan_gru_p90 = compute_fleet_plan(gru_test_df, config, demand_col="y_pred_p90")

    # Sensibilidad +-10% demanda y parámetros operativos
    sensitivity = _run_sensitivity_analysis(gru_test_df, config, demand_col="y_pred_p90")

    # P3: Valor económico del pronóstico (Ridge vs GRU, Puntual vs P90)
    econ_value = _evaluate_forecast_economic_value(preds, config)

    # P4: Temporada alta (semana pico dic-2023 vs semana normal sep-2023)
    peak_season = _analyze_peak_season(gru_test_df, config)

    # Figuras 06 y 07
    fig6_path = resolve_path(config["paths"]["fig_fleet_30d"])
    fig7_path = resolve_path(config["paths"]["fig_fleet_sensitivity"])
    _plot_fleet_30d(gru_30d_df, config, fig6_path)
    _plot_fleet_sensitivity(sensitivity, config, fig7_path)

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "regulatory_parameters_used": {
            "max_weekly_ordinary_hours": config["fleet_business"]["max_weekly_ordinary_hours"],
            "max_daily_shift_hours": config["fleet_business"]["max_daily_shift_hours"],
            "second_driver_threshold_hours": config["fleet_business"][
                "second_driver_threshold_hours"
            ],
            "weekly_rest_factor": config["fleet_business"]["weekly_rest_factor"],
            "absenteeism_rate": config["fleet_business"]["absenteeism_rate"],
            "target_occupancy_ratio": config["fleet_business"]["target_occupancy_ratio"],
            "reserve_fleet_ratio": config["fleet_business"]["reserve_fleet_ratio"],
        },
        "q1_and_q2_fleet_and_drivers_test": {
            "gru_puntual_system": plan_gru_p50["system_summary"],
            "gru_p90_system": plan_gru_p90["system_summary"],
            "gru_p90_by_corridor": plan_gru_p90["by_corridor_summary"],
            "gru_p90_by_route": plan_gru_p90["by_route_summary"],
            "sensitivity": sensitivity,
        },
        "q3_forecast_economic_value": econ_value,
        "q4_high_season_analysis": peak_season,
        "figures_generated": [
            str(config["paths"]["fig_fleet_30d"]),
            str(config["paths"]["fig_fleet_sensitivity"]),
        ],
    }

    out_json = resolve_path(config["paths"]["business_summary_file"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Impresión ejecutiva en consola
    s50 = plan_gru_p50["system_summary"]
    s90 = plan_gru_p90["system_summary"]
    print("\n" + "=" * 105)
    print("RESUMEN EJECUTIVO DE NEGOCIO M1 (TEST 2023-07 A 2023-12, 184 DÍAS, 4 CORREDORES ACOPLADOS)")
    print("=" * 105)
    print(
        f"1) FLOTA REQUERIDA (con 10% reserva) | GRU Puntual: media={s50['mean_total_fleet_with_reserve']:.1f} veh "
        f"(pico={s50['peak_total_fleet_with_reserve']}) | GRU P90: media={s90['mean_total_fleet_with_reserve']:.1f} veh "
        f"(pico={s90['peak_total_fleet_with_reserve']})"
    )
    print(
        f"   Parámetro más influyente (±10%): {sensitivity['most_influential_parameter']} "
        f"(oscilación={sensitivity['parameter_tornado_pm10pct'][0]['total_swing_vehicles']:.1f} veh)"
    )
    print(
        f"2) CONDUCTORES (42h/sem Ley 2101/2021 + 2do cond >8h Res. 315/2013 + 7/6 descanso + 6% ausentismo):\n"
        f"   GRU Puntual: activos/día={s50['mean_active_drivers_day']:.1f}, plantilla={s50['mean_payroll_drivers_total']:.1f} "
        f"(pico={s50['peak_payroll_drivers_total']}) | "
        f"GRU P90: activos/día={s90['mean_active_drivers_day']:.1f}, plantilla={s90['mean_payroll_drivers_total']:.1f} "
        f"(pico={s90['peak_payroll_drivers_total']})"
    )
    print("\n3) VALOR DEL PRONÓSTICO EN TEST (PASAJEROS SIN ATENDER vs. VEHÍCULOS-DÍA SOBRANTES):")
    sc_rows = []
    for sc_name, sc_vals in econ_value["scenarios_test_184d"].items():
        sc_rows.append(
            {
                "Escenario": sc_name,
                "Pas. sin atender": f"{sc_vals['unmet_passengers_physical']:,} ({sc_vals['unmet_passengers_pct_of_demand']:.2f}%)",
                "Veh-día sobrantes": f"{sc_vals['surplus_vehicle_days']:,}",
                "Costo Sobrantes (MM COP)": f"{sc_vals['surplus_cost_cop']/1e6:,.1f}",
                "Costo Faltantes (MM COP)": f"{sc_vals['shortage_cost_cop']/1e6:,.1f}",
                "Costo Desajuste Total (MM COP)": f"{sc_vals['total_mismatch_cost_cop']/1e6:,.1f}",
            }
        )
    print(pd.DataFrame(sc_rows).to_string(index=False))
    th_gru = econ_value["p90_vs_point_thresholds"]["GRU"]
    print(
        f"   Umbral crítico GRU (c_corto / c_sobrante)* = {th_gru['critical_ratio_c_corto_over_c_sobrante']:.4f} "
        f"(equivale a c_corto >= ${th_gru['breakeven_c_corto_cop_at_ref_surplus_cost']:,.0f} COP/pasajero; "
        f"configurado=${config['fleet_business']['costs_cop']['unmet_passenger_penalty_cop']:,.0f} COP -> P90 óptimo={th_gru['p90_recommended_under_yaml_costs']})"
    )
    pk_add = peak_season["additional_vehicles_at_december_peak"]
    print(
        f"\n4) TEMPORADA ALTA (Dic 22-28 vs Sep 11-17): +{pk_add['total_extra_vehicles']} vehículos adicionales "
        f"({pk_add['BUS']} BUS, {pk_add['MICROBUS']} MICROBUS) y +{pk_add['total_extra_payroll_drivers']} conductores en plantilla."
    )
    print(f"\n[OK] Resumen JSON: {out_json}\n[OK] Figuras: {fig6_path} | {fig7_path}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Análisis de dimensionamiento de flota, conductores y valor económico (M1)."
    )
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(
        config["reproducibility"]["seed"],
        config["reproducibility"].get("deterministic_torch", True),
    )
    run_business_analysis(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
