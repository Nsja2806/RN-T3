"""Análisis Exploratorio de Datos (EDA) temporal y figuras de alta resolución para M1.

Responsabilidad (100% offline sobre ``data/processed/m1/demanda_diaria.parquet``):
    1. Verificar cuantitativamente sobre ``train + val`` (y documentar en ``test``) las 4
       hipótesis de dominio del transporte intermunicipal colombiano:
         - H1: Efecto día de la semana (viernes/domingo vs. martes/miércoles).
         - H2: Efecto festivos y puentes festivos (Ley Emiliani).
         - H3: Efecto temporadas altas (Semana Santa, mitad de año junio-julio, fin de año).
         - H4: Recuperación post-pandemia (2021 vs. 2022 vs. 2023) y choque del Paro Nacional
               (mayo 2021).
    2. Exportar resumen numérico estructurado en ``reports/metrics/m1_eda_summary.json``.
    3. Generar figuras listas para publicación (300 dpi) en ``reports/figures/m1/``:
         - ``fig_m1_01_series_demanda.png``: Panel 4x2 de las 8 rutas con franjas Train/Val/Test.
         - ``fig_m1_02_descomposicion.png``: Descomposición multi-estacional MSTL (semanal P=7
           y anual P=365) sobre rutas representativas.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import MSTL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import add_config_override_args, load_module_config, resolve_path
from src.utils.seed import set_seed

DOW_NAMES_ES = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}


def _pct_uplift(high_val: float, base_val: float) -> float:
    if base_val <= 0:
        return 0.0
    return round(100.0 * (high_val - base_val) / base_val, 2)


def _mannwhitney_pvalue(group_a: pd.Series, group_b: pd.Series) -> float:
    a = group_a.dropna().to_numpy(dtype=np.float64)
    b = group_b.dropna().to_numpy(dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 1.0
    _, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p_val)


def _evaluate_hypotheses_on_subset(
    df_sub: pd.DataFrame,
    paro_start: str,
    paro_end: str,
) -> Dict[str, Any]:
    """Evalúa las hipótesis H1-H4 sobre un subconjunto temporal (ej. train+val o test)."""
    # H1: Viernes (4) y Domingo (6) vs Martes (1) y Miércoles (2)
    dow_means = {
        DOW_NAMES_ES[int(k)]: round(float(v), 2)
        for k, v in df_sub.groupby("day_of_week")["pasajeros"].mean().items()
    }
    fri_sun = df_sub.loc[df_sub["day_of_week"].isin([4, 6]), "pasajeros"]
    tue_wed = df_sub.loc[df_sub["day_of_week"].isin([1, 2]), "pasajeros"]
    h1_by_route: Dict[str, Dict[str, float]] = {}
    for rid, grp in df_sub.groupby("route_id", sort=False):
        fs_m = float(grp.loc[grp["day_of_week"].isin([4, 6]), "pasajeros"].mean())
        tw_m = float(grp.loc[grp["day_of_week"].isin([1, 2]), "pasajeros"].mean())
        h1_by_route[str(rid)] = {
            "viernes_domingo_mean": round(fs_m, 2),
            "martes_miercoles_mean": round(tw_m, 2),
            "uplift_pct": _pct_uplift(fs_m, tw_m),
        }

    h1 = {
        "mean_by_day_of_week": dow_means,
        "viernes_domingo_mean": round(float(fri_sun.mean()), 2),
        "martes_miercoles_mean": round(float(tue_wed.mean()), 2),
        "uplift_viernes_domingo_vs_martes_miercoles_pct": _pct_uplift(
            float(fri_sun.mean()), float(tue_wed.mean())
        ),
        "mannwhitney_p_value": _mannwhitney_pvalue(fri_sun, tue_wed),
        "confirmed": bool(float(fri_sun.mean()) > float(tue_wed.mean())),
        "by_route": h1_by_route,
    }

    # H2: Puentes festivos y festivos Ley Emiliani vs días ordinarios sin puente
    puente_pas = df_sub.loc[df_sub["is_puente_festivo"] == 1, "pasajeros"]
    non_puente_pas = df_sub.loc[df_sub["is_puente_festivo"] == 0, "pasajeros"]
    salida_puente_pas = df_sub.loc[df_sub["is_salida_puente"] == 1, "pasajeros"]
    retorno_puente_pas = df_sub.loc[df_sub["is_retorno_puente"] == 1, "pasajeros"]
    hol_pas = df_sub.loc[df_sub["is_holiday_co"] == 1, "pasajeros"]

    h2_by_route: Dict[str, Dict[str, float]] = {}
    for rid, grp in df_sub.groupby("route_id", sort=False):
        p_m = float(grp.loc[grp["is_puente_festivo"] == 1, "pasajeros"].mean())
        np_m = float(grp.loc[grp["is_puente_festivo"] == 0, "pasajeros"].mean())
        sal_m = float(grp.loc[grp["is_salida_puente"] == 1, "pasajeros"].mean())
        ret_m = float(grp.loc[grp["is_retorno_puente"] == 1, "pasajeros"].mean())
        h2_by_route[str(rid)] = {
            "puente_mean": round(p_m, 2),
            "no_puente_mean": round(np_m, 2),
            "salida_puente_mean": round(sal_m, 2),
            "retorno_puente_mean": round(ret_m, 2),
            "uplift_puente_pct": _pct_uplift(p_m, np_m),
        }

    h2 = {
        "puente_festivo_mean": round(float(puente_pas.mean()), 2) if len(puente_pas) else 0.0,
        "no_puente_mean": round(float(non_puente_pas.mean()), 2) if len(non_puente_pas) else 0.0,
        "viernes_salida_puente_mean": (
            round(float(salida_puente_pas.mean()), 2) if len(salida_puente_pas) else 0.0
        ),
        "lunes_retorno_puente_mean": (
            round(float(retorno_puente_pas.mean()), 2) if len(retorno_puente_pas) else 0.0
        ),
        "holiday_day_mean": round(float(hol_pas.mean()), 2) if len(hol_pas) else 0.0,
        "uplift_puente_vs_no_puente_pct": _pct_uplift(
            float(puente_pas.mean()), float(non_puente_pas.mean())
        ),
        "uplift_viernes_salida_vs_no_puente_pct": _pct_uplift(
            float(salida_puente_pas.mean()), float(non_puente_pas.mean())
        ),
        "mannwhitney_p_value": _mannwhitney_pvalue(puente_pas, non_puente_pas),
        "confirmed": bool(float(puente_pas.mean()) > float(non_puente_pas.mean())),
        "by_route": h2_by_route,
    }

    # H3: Temporadas altas (Semana Santa, mitad de año jun-jul, fin de año dic-ene) vs temporada valle
    off_mask = (
        (df_sub["is_semana_santa"] == 0)
        & (df_sub["is_temporada_mitad_anio"] == 0)
        & (df_sub["is_temporada_fin_anio"] == 0)
    )
    base_season = df_sub.loc[off_mask, "pasajeros"]
    sem_santa = df_sub.loc[df_sub["is_semana_santa"] == 1, "pasajeros"]
    mitad_anio = df_sub.loc[df_sub["is_temporada_mitad_anio"] == 1, "pasajeros"]
    fin_anio = df_sub.loc[df_sub["is_temporada_fin_anio"] == 1, "pasajeros"]
    dec_only = df_sub.loc[df_sub["month"] == 12, "pasajeros"]

    base_mean = float(base_season.mean()) if len(base_season) else 0.0
    h3 = {
        "off_peak_baseline_mean": round(base_mean, 2),
        "semana_santa_mean": round(float(sem_santa.mean()), 2) if len(sem_santa) else None,
        "uplift_semana_santa_pct": (
            _pct_uplift(float(sem_santa.mean()), base_mean) if len(sem_santa) else None
        ),
        "mitad_anio_jun_jul_mean": round(float(mitad_anio.mean()), 2) if len(mitad_anio) else None,
        "uplift_mitad_anio_pct": (
            _pct_uplift(float(mitad_anio.mean()), base_mean) if len(mitad_anio) else None
        ),
        "fin_anio_dic_ene_mean": round(float(fin_anio.mean()), 2) if len(fin_anio) else None,
        "uplift_fin_anio_pct": (
            _pct_uplift(float(fin_anio.mean()), base_mean) if len(fin_anio) else None
        ),
        "diciembre_mean": round(float(dec_only.mean()), 2) if len(dec_only) else None,
        "uplift_diciembre_pct": (
            _pct_uplift(float(dec_only.mean()), base_mean) if len(dec_only) else None
        ),
    }

    # H4: Evolución interanual y choque del Paro Nacional (mayo 2021)
    df_sub_yr = df_sub.assign(year=df_sub["fecha_despacho"].dt.year)
    yearly_means = {
        str(int(yr)): round(float(val), 2)
        for yr, val in df_sub_yr.groupby("year")["pasajeros"].mean().items()
    }

    p_start = pd.Timestamp(paro_start)
    p_end = pd.Timestamp(paro_end)
    may_2021 = df_sub[
        (df_sub["fecha_despacho"] >= pd.Timestamp("2021-05-01"))
        & (df_sub["fecha_despacho"] <= pd.Timestamp("2021-05-31"))
    ]
    paro_window = df_sub[(df_sub["fecha_despacho"] >= p_start) & (df_sub["fecha_despacho"] <= p_end)]
    ref_2021_non_paro = df_sub[
        (df_sub["fecha_despacho"].dt.year == 2021)
        & ~((df_sub["fecha_despacho"] >= p_start) & (df_sub["fecha_despacho"] <= p_end))
    ]

    paro_by_route: Dict[str, Any] = {}
    if not may_2021.empty and not ref_2021_non_paro.empty:
        for rid in sorted(df_sub["route_id"].unique()):
            m_may = float(may_2021.loc[may_2021["route_id"] == rid, "pasajeros"].mean())
            m_ref = float(
                ref_2021_non_paro.loc[ref_2021_non_paro["route_id"] == rid, "pasajeros"].mean()
            )
            paro_by_route[str(rid)] = {
                "may_2021_mean": round(m_may, 2),
                "rest_of_2021_mean": round(m_ref, 2),
                "drop_pct_vs_rest_of_2021": _pct_uplift(m_may, m_ref),
            }

    h4: Dict[str, Any] = {
        "yearly_daily_mean_passengers": yearly_means,
    }
    if "2021" in yearly_means and "2022" in yearly_means:
        h4["growth_2022_vs_2021_pct"] = _pct_uplift(yearly_means["2022"], yearly_means["2021"])
    if "2022" in yearly_means and "2023" in yearly_means:
        h4["growth_2023_vs_2022_pct"] = _pct_uplift(yearly_means["2023"], yearly_means["2022"])
    if not may_2021.empty and not ref_2021_non_paro.empty:
        h4["paro_nacional_may_2021"] = {
            "window": f"{paro_start} a {paro_end}",
            "may_2021_mean": round(float(may_2021["pasajeros"].mean()), 2),
            "paro_window_mean": round(float(paro_window["pasajeros"].mean()), 2),
            "rest_of_2021_mean": round(float(ref_2021_non_paro["pasajeros"].mean()), 2),
            "may_2021_drop_pct_vs_rest_2021": _pct_uplift(
                float(may_2021["pasajeros"].mean()),
                float(ref_2021_non_paro["pasajeros"].mean()),
            ),
            "by_route": paro_by_route,
        }

    return {
        "h1_day_of_week_effect": h1,
        "h2_holidays_and_puentes_emiliani": h2,
        "h3_high_seasons_effect": h3,
        "h4_post_pandemic_and_paro_2021": h4,
    }


def plot_series_panel(df: pd.DataFrame, config: Dict[str, Any], out_path: Path) -> None:
    """Genera fig_m1_01_series_demanda.png: panel 4x2 de las 8 rutas con franjas Train/Val/Test."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    eda_cfg = config["eda"]
    split_cfg = config["data"]["split_dates"]
    routes = config["data"]["selected_routes"]
    smooth_w = int(eda_cfg.get("rolling_smooth_days", 14))
    dpi = int(eda_cfg.get("dpi", 300))

    train_start = pd.Timestamp(split_cfg["train_start"])
    train_end = pd.Timestamp(split_cfg["train_end"])
    val_start = pd.Timestamp(split_cfg["val_start"])
    val_end = pd.Timestamp(split_cfg["val_end"])
    test_start = pd.Timestamp(split_cfg["test_start"])
    test_end = pd.Timestamp(split_cfg["test_end"])
    paro_start = pd.Timestamp(eda_cfg["paro_nacional_start"])
    paro_end = pd.Timestamp(eda_cfg["paro_nacional_end"])

    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    axes_flat = axes.flatten()

    for idx, r_meta in enumerate(routes):
        ax = axes_flat[idx]
        rid = str(r_meta["route_id"])
        sub = df[df["route_id"] == rid].sort_values("fecha_despacho")
        dates = sub["fecha_despacho"]
        vals = sub["pasajeros"]
        roll_vals = vals.rolling(window=smooth_w, min_periods=1).mean()

        # Franjas de partición temporal
        ax.axvspan(train_start, train_end, color="#e8f1fa", alpha=0.55, label="Train (2021-2022)")
        ax.axvspan(val_start, val_end, color="#fff4db", alpha=0.70, label="Val (2023-S1)")
        ax.axvspan(test_start, test_end, color="#fde8e8", alpha=0.65, label="Test (2023-S2)")
        ax.axvspan(paro_start, paro_end, color="#9e9e9e", alpha=0.25, label="Paro Nacional (May-2021)")

        ax.axvline(val_start, color="#d9822b", linestyle="--", linewidth=1.1)
        ax.axvline(test_start, color="#c0392b", linestyle="--", linewidth=1.1)

        ax.plot(dates, vals, color="#5c88b5", alpha=0.42, linewidth=0.7, label="Pasajeros diarios")
        ax.plot(
            dates,
            roll_vals,
            color="#10375c",
            linewidth=1.5,
            label=f"Media móvil ({smooth_w}d)",
        )

        veh = r_meta["predominant_vehicle"]
        ax.set_title(f"{r_meta['route_label']}  [{rid} | {veh}]", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Pasajeros / día", fontsize=9)
        ax.set_xlim(train_start, test_end)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        if idx == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Fecha de despacho", fontsize=10)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Series Diarias de Demanda Intermunicipal en Origen (8 Rutas, 2021–2023) y Partición Temporal",
        fontsize=13.5,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.978])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mstl_decomposition(df: pd.DataFrame, config: Dict[str, Any], out_path: Path) -> None:
    """Genera fig_m1_02_descomposicion.png usando MSTL (P=7 y P=365) sobre train+val."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    eda_cfg = config["eda"]
    dpi = int(eda_cfg.get("dpi", 300))
    periods = tuple(int(p) for p in eda_cfg.get("mstl_periods", [7, 365]))
    target_routes: List[str] = list(eda_cfg.get("decomposition_routes", ["11001_50001", "11001_76001"]))

    route_labels = {str(r["route_id"]): str(r["route_label"]) for r in config["data"]["selected_routes"]}

    # Ajustar MSTL estrictamente sobre train + val (911 días >= 2*365) para evitar fuga del bloque Test
    df_tv = df[df["split"].isin(["train", "val"])].copy()

    n_cols = len(target_routes)
    fig, axes = plt.subplots(5, n_cols, figsize=(8 * n_cols, 12.5), sharex="col")
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    panel_titles = [
        "Serie Observada (Pasajeros/día)",
        "Tendencia Suavizada",
        f"Estacionalidad Semanal (P = {periods[0]}d)",
        f"Estacionalidad Anual (P = {periods[1]}d)",
        "Componente Residual",
    ]
    colors = ["#1f4e78", "#d95f02", "#2b8cbe", "#7570b3", "#636363"]

    for col_idx, rid in enumerate(target_routes):
        sub = df_tv[df_tv["route_id"] == rid].sort_values("fecha_despacho").reset_index(drop=True)
        series = pd.Series(
            sub["pasajeros"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(sub["fecha_despacho"], freq="D"),
            name="pasajeros",
        )
        mstl_res = MSTL(series, periods=periods).fit()

        seasonal_df = mstl_res.seasonal
        if isinstance(seasonal_df, pd.Series):
            seas_w = seasonal_df
            seas_y = pd.Series(0.0, index=series.index)
        else:
            seas_w = seasonal_df.iloc[:, 0]
            seas_y = seasonal_df.iloc[:, 1]

        components = [
            mstl_res.observed,
            mstl_res.trend,
            seas_w,
            seas_y,
            mstl_res.resid,
        ]

        label_title = route_labels.get(rid, rid)
        for row_idx, (comp, p_title, col_hex) in enumerate(zip(components, panel_titles, colors)):
            ax = axes[row_idx, col_idx]
            lw = 0.85 if row_idx in (0, 2, 4) else 1.8
            ax.plot(series.index, comp.values, color=col_hex, linewidth=lw)
            if row_idx in (2, 3, 4):
                ax.axhline(0.0, color="#333333", linestyle=":", linewidth=0.8)

            if row_idx == 0:
                ax.set_title(
                    f"{label_title} ({rid})\n{p_title}",
                    fontsize=11,
                    fontweight="bold",
                )
            else:
                ax.set_title(p_title, fontsize=9.5, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel("Pasajeros", fontsize=9)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        axes[-1, col_idx].set_xlabel("Fecha (Train + Val: 2021-01 a 2023-06)", fontsize=9.5)
        axes[-1, col_idx].tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Descomposición Multi-Estacional MSTL (Semanal P=7d + Anual P=365d) sobre Train + Val",
        fontsize=13,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_eda(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta la verificación cuantitativa de hipótesis y genera las figuras exploratorias."""
    paths_cfg = config["paths"]
    eda_cfg = config["eda"]

    parquet_path = resolve_path(paths_cfg["processed_dataset_file"])
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset procesado offline: {parquet_path}. "
            "Ejecuta primero src/modulo1_demanda/preprocess.py."
        )

    df = pd.read_parquet(parquet_path)
    df["fecha_despacho"] = pd.to_datetime(df["fecha_despacho"])

    paro_start = str(eda_cfg["paro_nacional_start"])
    paro_end = str(eda_cfg["paro_nacional_end"])

    df_train_val = df[df["split"].isin(["train", "val"])].copy()
    df_test = df[df["split"] == "test"].copy()
    df_all = df.copy()

    summary_train_val = _evaluate_hypotheses_on_subset(df_train_val, paro_start, paro_end)
    summary_test = _evaluate_hypotheses_on_subset(df_test, paro_start, paro_end)
    full_years = {
        str(int(yr)): round(float(val), 2)
        for yr, val in df_all.assign(year=df_all["fecha_despacho"].dt.year)
        .groupby("year")["pasajeros"]
        .mean()
        .items()
    }

    fig_dir = resolve_path(paths_cfg["figures_dir"])
    fig1_path = fig_dir / str(eda_cfg["fig_series_filename"])
    fig2_path = fig_dir / str(eda_cfg["fig_decomposition_filename"])

    plot_series_panel(df=df_all, config=config, out_path=fig1_path)
    plot_mstl_decomposition(df=df_all, config=config, out_path=fig2_path)

    eda_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(paths_cfg["processed_dataset_file"]),
        "figures_generated": [
            str(Path(paths_cfg["figures_dir"]) / str(eda_cfg["fig_series_filename"])),
            str(Path(paths_cfg["figures_dir"]) / str(eda_cfg["fig_decomposition_filename"])),
        ],
        "full_3year_daily_means_2021_2023": {
            "by_year": full_years,
            "growth_2022_vs_2021_pct": _pct_uplift(full_years["2022"], full_years["2021"]),
            "growth_2023_vs_2022_pct": _pct_uplift(full_years["2023"], full_years["2022"]),
            "growth_2023_vs_2021_pct": _pct_uplift(full_years["2023"], full_years["2021"]),
        },
        "train_val_hypotheses_verification": summary_train_val,
        "test_behavior_documentation": summary_test,
    }

    out_json = resolve_path(paths_cfg["eda_summary_file"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(eda_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    tv_h1 = summary_train_val["h1_day_of_week_effect"]
    tv_h2 = summary_train_val["h2_holidays_and_puentes_emiliani"]
    tv_h3 = summary_train_val["h3_high_seasons_effect"]
    tv_h4 = summary_train_val["h4_post_pandemic_and_paro_2021"]

    print("\n=== RESUMEN CUANTITATIVO DE HIPÓTESIS M1 (TRAIN + VAL: 2021-01 a 2023-06) ===")
    print(
        f"  H1 Día semana (Vie/Dom vs Mar/Mié): {tv_h1['viernes_domingo_mean']:.1f} vs "
        f"{tv_h1['martes_miercoles_mean']:.1f} pas/día "
        f"({tv_h1['uplift_viernes_domingo_vs_martes_miercoles_pct']:+.2f}%, p={tv_h1['mannwhitney_p_value']:.2e})"
    )
    print(
        f"  H2 Puentes Emiliani vs No-Puente:   {tv_h2['puente_festivo_mean']:.1f} vs "
        f"{tv_h2['no_puente_mean']:.1f} pas/día "
        f"({tv_h2['uplift_puente_vs_no_puente_pct']:+.2f}%, Viernes salida: {tv_h2['uplift_viernes_salida_vs_no_puente_pct']:+.2f}%)"
    )
    print(
        f"  H3 Temporadas vs Valle ({tv_h3['off_peak_baseline_mean']:.1f} pas/día): "
        f"Semana Santa {tv_h3['uplift_semana_santa_pct']:+.2f}% | "
        f"Mitad de año {tv_h3['uplift_mitad_anio_pct']:+.2f}% | "
        f"Diciembre {tv_h3['uplift_diciembre_pct']:+.2f}%"
    )
    print(
        f"  H4 Post-pandemia (2021 -> 2022 -> 2023 completo): "
        f"{full_years['2021']:.1f} -> {full_years['2022']:.1f} ({eda_payload['full_3year_daily_means_2021_2023']['growth_2022_vs_2021_pct']:+.2f}%) "
        f"-> {full_years['2023']:.1f} ({eda_payload['full_3year_daily_means_2021_2023']['growth_2023_vs_2022_pct']:+.2f}%) | "
        f"Choque Mayo 2021: {tv_h4['paro_nacional_may_2021']['may_2021_drop_pct_vs_rest_2021']:+.2f}%"
    )
    print(f"[OK] Resumen EDA guardado en: {out_json}")
    print(f"[OK] Figuras 300 dpi guardadas en:\n  - {fig1_path}\n  - {fig2_path}")
    return eda_payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Análisis Exploratorio de Datos (EDA) offline del Módulo 1.")
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(
        config["reproducibility"]["seed"],
        config["reproducibility"].get("deterministic_torch", True),
    )
    run_eda(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
