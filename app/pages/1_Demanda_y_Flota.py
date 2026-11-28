"""Página interactiva en Streamlit del Módulo 1: Demanda, Flota y Conductores.

Contrato estricto de interfaz (AGENTS.md):
    - Importa exclusivamente funciones puras desde ``src/modulo1_demanda/predict.py``
      (``load_artifacts``, ``load_precomputed_reports``, ``predict_demand``).
    - Usa ``@st.cache_resource`` para mantener el modelo en memoria (< 0.15 s por cambio).
    - 100% offline: cero descargas y cero reentrenamiento.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulo1_demanda.predict import (
    load_artifacts,
    load_precomputed_reports,
    predict_demand,
)


@st.cache_resource(show_spinner="Cargando modelo neuronal M1 y catálogo offline...")
def get_cached_artifacts() -> Dict[str, Any]:
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_cached_reports() -> Dict[str, Any]:
    return load_precomputed_reports()


def build_forecast_figure(
    hist_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    service_level: str,
    demand_multiplier: float,
    title_label: str,
) -> plt.Figure:
    """Construye la figura limpia de historial (28d) + pronóstico (30d) + banda P90."""
    hist_agg = hist_df.groupby("fecha_despacho", as_index=False)["pasajeros_hist"].sum()
    fc_agg = (
        fc_df.groupby("target_date", as_index=False)
        .agg(
            pred_p50_adj=("pred_p50_adj", "sum"),
            pred_p90_adj=("pred_p90_adj", "sum"),
            selected_demand=("selected_demand", "sum"),
            y_true=("y_true", lambda s: s.sum(min_count=1)),
        )
        .sort_values("target_date")
    )

    fig, ax = plt.subplots(figsize=(11.5, 3.8), dpi=95)
    ax.plot(
        hist_agg["fecha_despacho"],
        hist_agg["pasajeros_hist"],
        color="#34495e",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Historial causal observado (28 días)",
    )

    origin_dt = hist_agg["fecha_despacho"].max()
    ax.axvline(origin_dt, color="#7f8c8d", linestyle="--", linewidth=1.2, label=f"Origen pronóstico ({origin_dt.strftime('%Y-%m-%d')})")

    ax.fill_between(
        fc_agg["target_date"],
        fc_agg["pred_p50_adj"],
        fc_agg["pred_p90_adj"],
        color="#1b9e77",
        alpha=0.22,
        label="Banda de incertidumbre P50–P90",
    )
    ax.plot(
        fc_agg["target_date"],
        fc_agg["pred_p50_adj"],
        color="#2b8cbe",
        linewidth=1.8 if service_level == "p90" else 2.4,
        linestyle="-" if service_level == "p50" else "--",
        label="Pronóstico Puntual (P50)",
    )
    ax.plot(
        fc_agg["target_date"],
        fc_agg["pred_p90_adj"],
        color="#1b9e77",
        linewidth=2.4 if service_level == "p90" else 1.5,
        label="Pronóstico Conservador (P90)",
    )

    if fc_agg["y_true"].notna().any():
        ax.plot(
            fc_agg["target_date"],
            fc_agg["y_true"],
            color="#111111",
            linestyle=":",
            linewidth=1.4,
            marker="x",
            markersize=3.5,
            label="Demanda real observada (backtest)",
        )

    mult_tag = f" | Escenario {(demand_multiplier - 1.0) * 100:+.0f}%" if abs(demand_multiplier - 1.0) > 1e-6 else ""
    ax.set_title(f"Pronóstico Diario a 30 Días — {title_label}{mult_tag}", fontweight="bold", fontsize=11)
    ax.set_ylabel("Pasajeros / día")
    ax.set_xlabel("Fecha")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper left", ncol=3, fontsize=8.5, frameon=True)
    fig.tight_layout()
    return fig


def build_fleet_drivers_figure(sys_daily: pd.DataFrame, title_label: str) -> plt.Figure:
    """Construye la figura dual día a día de vehículos y conductores requeridos (30 días)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.6), sharex=True, dpi=95)

    ax0 = axes[0]
    ax0.bar(
        sys_daily["target_date"],
        sys_daily["operational_fleet_total"],
        color="#2b8cbe",
        alpha=0.82,
        label="Flota operativa",
    )
    ax0.plot(
        sys_daily["target_date"],
        sys_daily["total_fleet_with_reserve"],
        color="#d95f02",
        linewidth=2.1,
        marker="o",
        markersize=3.5,
        label="Flota total (+10% reserva)",
    )
    ax0.set_title(f"Vehículos Requeridos / Día ({title_label})", fontweight="bold", fontsize=10.5)
    ax0.set_ylabel("Vehículos")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8.5)

    ax1 = axes[1]
    ax1.plot(
        sys_daily["target_date"],
        sys_daily["active_drivers_day_total"],
        color="#1b9e77",
        linestyle="--",
        linewidth=1.8,
        marker="s",
        markersize=3,
        label="Conductores activos en turno/día",
    )
    ax1.plot(
        sys_daily["target_date"],
        sys_daily["payroll_drivers_total"],
        color="#7570b3",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="Plantilla en nómina (descanso 7/6 + ausentismo)",
    )
    ax1.set_title("Conductores Requeridos / Día (Normativa 2026)", fontweight="bold", fontsize=10.5)
    ax1.set_ylabel("Conductores")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=20)
        ax.set_xlabel("Fecha objetivo (MM-DD)")

    fig.tight_layout()
    return fig


def _build_export_table(
    fc_df: pd.DataFrame,
    fleet_plan: Dict[str, Any],
    requested_routes: List[str],
) -> pd.DataFrame:
    """Une el pronóstico por ruta con el plan diario de despachos, flota de corredor y conductores."""
    r_plan = fleet_plan["daily_route_plan"]
    c_plan = fleet_plan["daily_corridor_plan"][
        [
            "target_date",
            "corridor_id",
            "operational_fleet",
            "reserve_vehicles",
            "total_fleet_with_reserve",
        ]
    ].rename(
        columns={
            "operational_fleet": "corridor_operational_fleet",
            "reserve_vehicles": "corridor_reserve_vehicles",
            "total_fleet_with_reserve": "corridor_total_fleet",
        }
    )

    merged = pd.merge(
        fc_df[
            [
                "target_date",
                "horizon_step",
                "route_id",
                "route_label",
                "corridor_id",
                "direction",
                "pred_p50_adj",
                "pred_p90_adj",
                "selected_demand",
                "y_true",
            ]
        ],
        r_plan[
            [
                "target_date",
                "route_id",
                "vehicle_class",
                "nominal_capacity",
                "effective_capacity",
                "dispatches",
                "crew_per_dispatch",
                "active_drivers_day",
                "payroll_drivers_required",
            ]
        ],
        on=["target_date", "route_id"],
        how="left",
    )
    merged = pd.merge(merged, c_plan, on=["target_date", "corridor_id"], how="left")
    merged = merged[merged["route_id"].isin(requested_routes)].copy()
    merged["target_date"] = pd.to_datetime(merged["target_date"]).dt.strftime("%Y-%m-%d")
    return merged.sort_values(["target_date", "route_id"]).reset_index(drop=True)


def render_page() -> None:
    st.set_page_config(page_title="M1: Demanda, Flota y Conductores", page_icon="📈", layout="wide")

    artifacts = get_cached_artifacts()
    reports = get_cached_reports()
    cfg = artifacts["config"]
    fb_cfg = cfg["fleet_business"]
    routes_cfg: List[Dict[str, Any]] = cfg["data"]["selected_routes"]

    st.title("📈 Módulo 1: Pronóstico de Demanda, Dimensionamiento de Flota y Conductores")
    st.caption(
        "Transporte Intermunicipal en Colombia (SuperTransporte + DANE DIVIPOLA) — "
        "Inferencia neuronal 100% offline acoplada a normativa laboral y operativa vigente (2026)."
    )

    # -------------------------------------------------------------------------
    # Controles interactivos en barra lateral
    # -------------------------------------------------------------------------
    route_options: Dict[str, Optional[str]] = {
        f"{r['route_id']} — {r['route_label']} ({r['predominant_vehicle']}, ciclo {r['cycle_time_hours']:g}h)": str(
            r["route_id"]
        )
        for r in routes_cfg
    }
    route_options["TODAS — Red Completa (4 Corredores / 8 Rutas Dirigidas)"] = None
    option_labels = list(route_options.keys())

    with st.sidebar:
        st.header("⚙️ Controles Operativos M1")
        selected_label = st.selectbox(
            "Ruta dirigida o red completa",
            options=option_labels,
            index=6,  # Default: Bogotá, D.C. -> Santiago de Cali (muestra doble conductor)
        )
        selected_route_id = route_options[selected_label]

        svc_choice = st.radio(
            "Nivel de servicio para dimensionar",
            options=["Conservador (P90)", "Puntual (P50)"],
            index=0,
            help="P90 cubre hasta el percentil 90 de incertidumbre evitando pasajeros desatendidos en picos.",
        )
        svc_code = "p90" if "P90" in svc_choice else "p50"

        arch_choice = st.selectbox(
            "Arquitectura neuronal",
            options=["gru", "mlp"],
            index=0,
            format_func=lambda a: "GRU Global (Ganador 30 semillas)" if a == "gru" else "MLP Global (Referencia neuronal)",
        )

        min_date = artifacts["arrays"].dates[int(cfg["time_series"]["lookback_window"]) - 1].date()
        max_date = artifacts["arrays"].dates[-1].date()
        default_origin = dt.date.fromisoformat(str(fb_cfg["analysis"]["forecast_30d_origin_date"]))
        origin_picked = st.date_input(
            "Fecha de origen del pronóstico (t₀)",
            value=default_origin,
            min_value=min_date,
            max_value=max_date,
            help="Usa 2023-12-01 para auditar diciembre 2023 contra demanda real, o 2023-12-31 para proyectar enero 2024.",
        )

        st.divider()
        st.subheader("Sliders de Sensibilidad en Tiempo Real")
        occ_ratio = st.slider(
            "Ocupación objetivo (ρ)",
            min_value=0.65,
            max_value=0.95,
            value=float(fb_cfg["target_occupancy_ratio"]),
            step=0.01,
        )
        demand_pct = st.slider(
            "Escenario de demanda (%)",
            min_value=-20,
            max_value=20,
            value=0,
            step=1,
            format="%+d%%",
        )
        absenteeism_pct = st.slider(
            "Tasa de ausentismo de conductores (%)",
            min_value=0,
            max_value=15,
            value=int(round(100.0 * float(fb_cfg["absenteeism_rate"]))),
            step=1,
            format="%d%%",
        )

    # -------------------------------------------------------------------------
    # Inferencia en tiempo real (< 0.15 s)
    # -------------------------------------------------------------------------
    pred_out = predict_demand(
        artifacts=artifacts,
        route_id=selected_route_id,
        origin_date=origin_picked.isoformat(),
        service_level=svc_code,
        target_occupancy_ratio=occ_ratio,
        demand_multiplier=1.0 + (demand_pct / 100.0),
        absenteeism_rate=absenteeism_pct / 100.0,
        arch=arch_choice,
        include_reverse_pair_for_fleet=True,
    )

    fc_df: pd.DataFrame = pred_out["forecast_df"]
    hist_df: pd.DataFrame = pred_out["history_df"]
    fleet_plan: Dict[str, Any] = pred_out["fleet_plan"]
    sys_summary = fleet_plan["system_summary"]
    sys_daily = fleet_plan["daily_system_totals"]

    # Métricas específicas de la selección actual
    route_plan_req = fleet_plan["daily_route_plan"][
        fleet_plan["daily_route_plan"]["route_id"].isin(pred_out["requested_routes"])
    ]
    total_pax_30d = float(fc_df["selected_demand"].sum())
    mean_pax_day = float(fc_df.groupby("target_date")["selected_demand"].sum().mean())
    mean_disp_day = float(route_plan_req.groupby("target_date")["dispatches"].sum().mean())
    peak_disp_day = int(route_plan_req.groupby("target_date")["dispatches"].sum().max())
    max_crew = int(route_plan_req["crew_per_dispatch"].max())

    if selected_route_id is not None:
        rmeta = artifacts["route_meta"][selected_route_id]
        scope_title = f"{rmeta['route_label']} [{selected_route_id}]"
        corridor_title = f"Corredor {rmeta['corridor_id']}"
    else:
        scope_title = "Red Nacional Seleccionada (8 Rutas / 4 Corredores)"
        corridor_title = "4 Corredores Acoplados"

    # -------------------------------------------------------------------------
    # Tarjetas KPI en tiempo real
    # -------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Pasajeros Proyectados (30 días)",
        f"{total_pax_30d:,.0f}",
        delta=f"{mean_pax_day:,.0f} pas/día promedio",
        delta_color="off",
    )
    k2.metric(
        "Despachos / Día Promedio",
        f"{mean_disp_day:.1f}",
        delta=f"Pico diario: {peak_disp_day} despachos",
        delta_color="off",
    )
    k3.metric(
        f"Flota Total c/ Reserva ({corridor_title})",
        f"{sys_summary['mean_total_fleet_with_reserve']:.1f} veh",
        delta=f"Pico: {sys_summary['peak_total_fleet_with_reserve']} veh (10% reserva)",
        delta_color="off",
    )
    crew_badge = (
        "2 cond/despacho (Res. 315/2013 >8h)"
        if max_crew == 2 and selected_route_id is not None
        else ("Incluye relevo >8h en Bogotá↔Cali" if max_crew == 2 else "1 cond/despacho (Ley 2101 42h/sem)")
    )
    k4.metric(
        f"Conductores en Plantilla ({corridor_title})",
        f"{sys_summary['mean_payroll_drivers_total']:.1f}",
        delta=f"Pico: {sys_summary['peak_payroll_drivers_total']} | {crew_badge}",
        delta_color="off",
    )

    if max_crew == 2:
        st.info(
            "⚖️ **Regla normativa activa (Resolución 315 de 2013 Art. 6 — MinTransporte):** "
            "El recorrido por sentido en `Bogotá ↔ Cali` (~12.0 h) supera las **8.0 horas**, por lo que "
            "el motor asigna automáticamente **doble conductor (relevo a bordo: `crew_per_dispatch = 2`)**, "
            "además de respetar la jornada ordinaria de **42 h/semana** (Ley 2101 de 2021) y descanso `7/6`."
        )

    tab_op, tab_eval = st.tabs(
        [
            "🚌 Pronóstico y Plan Operativo a 30 Días",
            "📊 Evaluación Estadística (30 Semillas) y Caso de Negocio",
        ]
    )

    with tab_op:
        fig_fc = build_forecast_figure(
            hist_df=hist_df,
            fc_df=fc_df,
            service_level=pred_out["service_level"],
            demand_multiplier=pred_out["demand_multiplier"],
            title_label=scope_title,
        )
        st.pyplot(fig_fc, width="stretch")
        plt.close(fig_fc)

        fig_fl = build_fleet_drivers_figure(sys_daily=sys_daily, title_label=corridor_title)
        st.pyplot(fig_fl, width="stretch")
        plt.close(fig_fl)

        export_df = _build_export_table(
            fc_df=fc_df,
            fleet_plan=fleet_plan,
            requested_routes=pred_out["requested_routes"],
        )
        col_dl, col_cap = st.columns([1, 3])
        with col_dl:
            csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
            fname_tag = selected_route_id or "red_completa"
            st.download_button(
                label="⬇️ Descargar Plan Operativo 30d (CSV)",
                data=csv_bytes,
                file_name=f"plan_operativo_m1_{fname_tag}_{pred_out['origin_date']}_{svc_code}.csv",
                mime="text/csv",
            )
        with col_cap:
            st.caption(
                f"Detalle diario para **{scope_title}** desde `t₀ = {pred_out['origin_date']}` "
                f"(`ρ = {occ_ratio:.2f}`, demanda `{demand_pct:+d}%`, ausentismo `{absenteeism_pct}%`)."
            )
        st.dataframe(export_df, width="stretch", height=320)

    with tab_eval:
        st.subheader("1. Comparación Estadística en Test (Rolling Origin H=30d, 30 Semillas vs. Baselines)")
        ev30 = reports.get("eval30")
        bl_rep = reports.get("baselines")
        if ev30 and bl_rep:
            comp_rows = []
            for b_key, b_name in (
                ("seasonal_naive", "Seasonal Naive (m=7)"),
                ("ridge_calendar", "Ridge + Calendario (HiGHS P90)"),
            ):
                b_ov = bl_rep["models"][b_key]["test"]["overall"]
                b_hb = bl_rep["models"][b_key]["test"]["by_horizon_block"]
                comp_rows.append(
                    {
                        "Modelo": b_name,
                        "Parámetros": bl_rep["models"][b_key]["n_parameters"],
                        "MAE Global": f"{b_ov['mae']:.2f}",
                        "RMSE Global": f"{b_ov['rmse']:.2f}",
                        "MASE (m=7)": f"{b_ov['mase']:.4f}",
                        "MAE 1–7d": f"{b_hb['days_1_7']['mae']:.2f}",
                        "MAE 15–30d": f"{b_hb['days_15_30']['mae']:.2f}",
                        "Pinball P90": f"{b_ov['pinball_p90']:.2f}",
                        "Cobertura P90": f"{b_ov['coverage_p90_pct']:.2f}%",
                    }
                )
            for arch_k, arch_title in (("mlp", "MLP Global (30 semillas)"), ("gru", "GRU Global (30 semillas)")):
                s = ev30["summary"][arch_k]
                n_par = artifacts["checkpoints"][arch_k]["n_parameters"] if arch_k in artifacts["checkpoints"] else "-"
                comp_rows.append(
                    {
                        "Modelo": arch_title,
                        "Parámetros": n_par,
                        "MAE Global": f"{s['mae']['mean']:.2f} ± {s['mae']['std']:.2f}",
                        "RMSE Global": f"{s['rmse']['mean']:.2f} ± {s['rmse']['std']:.2f}",
                        "MASE (m=7)": f"{s['mase']['mean']:.4f} ± {s['mase']['std']:.4f}",
                        "MAE 1–7d": f"{s['mae_h1_7']['mean']:.2f} ± {s['mae_h1_7']['std']:.2f}",
                        "MAE 15–30d": f"{s['mae_h15_30']['mean']:.2f} ± {s['mae_h15_30']['std']:.2f}",
                        "Pinball P90": f"{s['pinball_p90']['mean']:.2f} ± {s['pinball_p90']['std']:.2f}",
                        "Cobertura P90": f"{100.0 * s['coverage_p90']['mean']:.2f}% ± {100.0 * s['coverage_p90']['std']:.2f}%",
                    }
                )
            st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

        biz = reports.get("business")
        if biz:
            st.subheader("2. Valor Operativo y Costos de Desajuste en Test (Semestre 2023-S2, COP)")
            econ = biz["q3_forecast_economic_value"]
            biz_rows = []
            for sc_name, sc_d in econ["scenarios_test_184d"].items():
                biz_rows.append(
                    {
                        "Política / Modelo": sc_name,
                        "Pasajeros sin atender": f"{sc_d['unmet_passengers_physical']:,} ({sc_d['unmet_passengers_pct_of_demand']:.2f}%)",
                        "Vehículos-día sobrantes": f"{sc_d['surplus_vehicle_days']:,}",
                        "Flota total media (veh)": f"{sc_d['mean_daily_total_fleet_with_reserve']:.1f}",
                        "Costo sobrantes (MM COP)": f"${sc_d['surplus_cost_cop']/1e6:,.1f}",
                        "Costo faltantes (MM COP)": f"${sc_d['shortage_cost_cop']/1e6:,.1f}",
                        "Costo total desajuste (MM COP)": f"${sc_d['total_mismatch_cost_cop']/1e6:,.1f}",
                    }
                )
            st.dataframe(pd.DataFrame(biz_rows), width="stretch", hide_index=True)

            th_g = econ["p90_vs_point_thresholds"]["GRU"]
            pk_s = biz["q4_high_season_analysis"]
            c_left, c_right = st.columns(2)
            with c_left:
                st.success(
                    f"**Umbral económico P90 vs Puntual (`GRU`):** "
                    f"`(c_corto / c_sobrante)* = {th_g['critical_ratio_c_corto_over_c_sobrante']:.4f}` "
                    f"(punto de equilibrio `${th_g['breakeven_c_corto_cop_at_ref_surplus_cost']:,.0f} COP` por pasajero; "
                    f"evita **{th_g['unmet_passengers_avoided_by_p90']:,}** pasajeros desatendidos en el semestre)."
                )
            with c_right:
                add_v = pk_s["additional_vehicles_at_december_peak"]
                src_c = pk_s["sourcing_alternatives_annual_cost_cop"]
                st.warning(
                    f"**Temporada Alta (Navidad Dic 22–28 vs Normal):** requiere **+{add_v['total_extra_vehicles']} vehículos** "
                    f"(`+{add_v['BUS']} BUS`, `+{add_v['MICROBUS']} MICROBUS`) y **+{add_v['total_extra_payroll_drivers']} conductores**. "
                    f"Cubrir el pico con flota arrendada (`${src_c['2_seasonal_leased_fleet_cop']/1e6:,.0f} MM/año`) o convenios "
                    f"(`${src_c['3_intercompany_collaboration_agreement_cop']/1e6:,.0f} MM/año`) ahorra >78% frente a flota propia "
                    f"ociosa (`${src_c['1_own_fleet_for_peak_cop']/1e6:,.0f} MM/año`)."
                )


if __name__ == "__main__":
    render_page()
