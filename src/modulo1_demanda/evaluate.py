"""Evaluación offline del Módulo 1 con las redes guardadas (sin reentrenar).

Responsabilidad:
    - Cargar ``models/m1/{mlp,gru}_best.pt`` y evaluar en ``test`` con el MISMO
      ``run_rolling_origin_evaluation`` / ``evaluate_forecast_records`` de ``baselines.py``
      (23 orígenes, H=30, paso 7 días, MASE con escala train m=7).
    - Recalcular los baselines (``seasonal_naive``, ``ridge_calendar`` con el alpha elegido en val) solo para
      comparar por día de horizonte; no se entrena ninguna red.
    - Guardar ``reports/metrics/m1_nn_metrics.json`` y las figuras (300 dpi):
        fig_m1_03_pronostico_vs_real.png, fig_m1_04_error_por_horizonte.png,
        fig_m1_05_comparacion_30_semillas.png (requiere haber corrido ``--mode eval30``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulo1_demanda.baselines import (
    RidgeCalendarBaseline,
    SeasonalNaiveBaseline,
    evaluate_forecast_records,
    run_rolling_origin_evaluation,
)
from src.modulo1_demanda.nn_data import NNPredictor, load_checkpoint, prepare_context
from src.utils.config import add_config_override_args, load_module_config, resolve_path
from src.utils.seed import set_seed

ARCHS = ("mlp", "gru")
LABEL = {"mlp": "MLP", "gru": "GRU", "seasonal_naive": "Seasonal Naive", "ridge_calendar": "Ridge + calendario"}
COLOR = {"mlp": "#d95f02", "gru": "#1b9e77", "seasonal_naive": "#7570b3", "ridge_calendar": "#555555"}


def _mae_by_horizon_day(records: pd.DataFrame) -> List[float]:
    ae = (records["y_true"] - records["y_pred_p50"]).abs()
    return [round(float(v), 4) for v in ae.groupby(records["horizon_step"]).mean().sort_index().to_numpy()]


def _baseline_records(config: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Recalcula las predicciones de los baselines en test (rápido, sin redes)."""
    df = ctx["arrays"].df
    df_train = df[df["split"] == "train"].copy()
    horizon, q = ctx["horizon"], ctx["quantile"]
    step = int(config["evaluation"].get("rolling_origin_step_days", 7))
    base_cfg = config.get("baselines", {})
    alpha = 10.0
    bl_path = resolve_path(config["paths"]["baselines_metrics_file"])
    if bl_path.exists():
        alpha = float(json.loads(bl_path.read_text(encoding="utf-8"))["models"]["ridge_calendar"]["selected_ridge_alpha_on_val"])
    sn = SeasonalNaiveBaseline(quantile=q).fit(df_train, horizon=horizon, step_days=step)
    rc = RidgeCalendarBaseline(
        ridge_alpha=alpha, quantile_alpha=float(base_cfg.get("quantile_alpha", 0.001)), quantile=q,
        include_interactions=bool(base_cfg.get("include_route_calendar_interactions", True)),
    ).fit(df_train)
    return {
        "seasonal_naive": run_rolling_origin_evaluation(df, ctx["test_origins"], horizon, sn.predict_from_origin),
        "ridge_calendar": run_rolling_origin_evaluation(df, ctx["test_origins"], horizon, rc.predict_from_origin),
    }


def _plot_forecast_vs_real(config, ctx, predictors, route_labels, out_path: Path) -> None:
    """Pronóstico P50 + banda P90 en test: ventanas de 30 días encadenadas (sin solapamiento)."""
    arrays, horizon = ctx["arrays"], ctx["horizon"]
    first = arrays.date_to_idx[pd.Timestamp(config["data"]["split_dates"]["test_start"])] - 1
    last_ok = arrays.date_to_idx[pd.Timestamp(config["data"]["split_dates"]["test_end"])] - horizon
    origin_pos = list(range(first, last_ok + 1, horizon))
    chains: Dict[str, pd.DataFrame] = {}
    for arch, predictor in predictors.items():
        chains[arch] = pd.concat(
            [predictor.predict_from_origin(arrays.df, arrays.dates[i], horizon) for i in origin_pos], ignore_index=True
        )
    test_df = arrays.df[arrays.df["split"] == "test"]
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    for ax, rid in zip(axes.flatten(), arrays.route_ids):
        real = test_df[test_df["route_id"] == rid].sort_values("fecha_despacho")
        ax.plot(real["fecha_despacho"], real["pasajeros"], color="#222222", linewidth=1.0, label="Real")
        if "gru" in chains:
            g = chains["gru"][chains["gru"]["route_id"] == rid].sort_values("target_date")
            ax.fill_between(g["target_date"], g["y_pred_p50"], g["y_pred_p90"], color=COLOR["gru"], alpha=0.22, label="GRU banda P50–P90")
            ax.plot(g["target_date"], g["y_pred_p50"], color=COLOR["gru"], linewidth=1.5, label="GRU P50")
            ax.plot(g["target_date"], g["y_pred_p90"], color=COLOR["gru"], linewidth=0.8, linestyle=":")
        if "mlp" in chains:
            m = chains["mlp"][chains["mlp"]["route_id"] == rid].sort_values("target_date")
            ax.plot(m["target_date"], m["y_pred_p50"], color=COLOR["mlp"], linewidth=1.3, linestyle="--", label="MLP P50")
        for i in origin_pos[1:]:
            ax.axvline(arrays.dates[i + 1], color="#bbbbbb", linewidth=0.6)
        ax.set_title(route_labels[rid], fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Pasajeros / día")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0, 0].legend(fontsize=8, ncol=2, loc="upper left")
    for ax in axes[-1, :]:
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Test (2023-07 a 2023-12): pronóstico a 30 días encadenado (P50 y banda hasta P90) vs. demanda real",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(config["eda"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _plot_error_by_horizon(config, mae_day: Dict[str, List[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    days = np.arange(1, len(next(iter(mae_day.values()))) + 1)
    for name in ("seasonal_naive", "ridge_calendar", "mlp", "gru"):
        if name in mae_day:
            ax.plot(days, mae_day[name], marker="o", markersize=3, linewidth=1.8, color=COLOR[name], label=LABEL[name])
    for edge in (7.5, 14.5):
        ax.axvline(edge, color="#bbbbbb", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Día del horizonte (h)")
    ax.set_ylabel("MAE en test (pasajeros/día, promedio 8 rutas)")
    ax.set_title("Error absoluto medio por día de horizonte (rolling origin, test)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(config["eda"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _plot_30_seeds(config, baselines_test: Dict[str, Dict[str, float]], out_path: Path) -> bool:
    csv_path = resolve_path(config["paths"]["eval30_csv"])
    if not csv_path.exists():
        print(f"[evaluate] {csv_path} no existe: se omite fig_m1_05 (ejecuta --mode eval30).")
        return False
    df = pd.read_csv(csv_path)
    metrics = [("mae", "MAE"), ("mase", "MASE (m=7)"), ("pinball_p90", "Pinball P90")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rng = np.random.default_rng(int(config["reproducibility"]["seed"]))
    for ax, (col, title) in zip(axes, metrics):
        data = [df.loc[df["arch"] == a, col].to_numpy() for a in ARCHS if (df["arch"] == a).any()]
        names = [LABEL[a] for a in ARCHS if (df["arch"] == a).any()]
        ax.boxplot(data, tick_labels=names, widths=0.5, showmeans=True)
        for i, vals in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, alpha=0.6, color=COLOR[ARCHS[i - 1]] if len(data) == 2 else "#555555")
        for name, style in (("seasonal_naive", "--"), ("ridge_calendar", ":")):
            if name in baselines_test and col in baselines_test[name]:
                ax.axhline(baselines_test[name][col], color=COLOR[name], linestyle=style, linewidth=1.6,
                           label=f"{LABEL[name]} ({baselines_test[name][col]:.2f})")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
    n = int(df.groupby("arch").size().min())
    fig.suptitle(f"Variabilidad entre {n} semillas en test (rolling origin) vs. baselines", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(config["eda"]["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return True


def run_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evalúa las redes guardadas en ``test`` y genera métricas y figuras."""
    ctx = prepare_context(config, with_train_windows=False)
    model_dir = resolve_path(config["paths"]["model_dir"])
    arrays = ctx["arrays"]
    route_labels = {str(r["route_id"]): str(r["route_label"]) for r in config["data"]["selected_routes"]}

    predictors: Dict[str, NNPredictor] = {}
    payload: Dict[str, Any] = {"models": {}, "mae_by_horizon_day": {}}
    records: Dict[str, pd.DataFrame] = {}
    for arch in ARCHS:
        ckpt_path = model_dir / f"{arch}_best.pt"
        if not ckpt_path.exists():
            print(f"[evaluate] Falta {ckpt_path}: se omite {arch}.")
            continue
        model, ckpt = load_checkpoint(ckpt_path)
        predictor = NNPredictor(model, arrays, ckpt["lookback"], ckpt["horizon"], ckpt["scale_floor_frac"],
                                np.asarray(ckpt["route_train_mean"], dtype=np.float32))
        rec = run_rolling_origin_evaluation(arrays.df, ctx["test_origins"], ctx["horizon"], predictor.predict_from_origin)
        metrics = evaluate_forecast_records(rec, ctx["mase_scales"], ctx["horizon_blocks"], ctx["quantile"])
        predictors[arch], records[arch] = predictor, rec
        payload["models"][arch] = {"n_parameters": ckpt["n_parameters"], "hparams": ckpt["hparams"], "checkpoint_meta": ckpt["meta"], "test": metrics}
        payload["mae_by_horizon_day"][arch] = _mae_by_horizon_day(rec)
        o = metrics["overall"]
        print(f"[evaluate:{arch}] params={ckpt['n_parameters']:,} | TEST MAE={o['mae']:.1f} RMSE={o['rmse']:.1f} MASE={o['mase']:.3f} "
              f"PinballP90={o['pinball_p90']:.1f} CobP90={o['coverage_p90_pct']:.1f}%")
    if not predictors:
        raise FileNotFoundError("No hay checkpoints en models/m1/. Ejecuta: python run.py train --module m1 --mode single")

    for name, rec in _baseline_records(config, ctx).items():
        payload["mae_by_horizon_day"][name] = _mae_by_horizon_day(rec)
    baselines_test: Dict[str, Dict[str, float]] = {}
    bl_path = resolve_path(config["paths"]["baselines_metrics_file"])
    if bl_path.exists():
        bl = json.loads(bl_path.read_text(encoding="utf-8"))
        baselines_test = {k: v["test"]["overall"] for k, v in bl["models"].items()}
        payload["baselines_test_overall"] = baselines_test

    fig_dir = config["paths"]
    _plot_forecast_vs_real(config, ctx, predictors, route_labels, resolve_path(fig_dir["fig_forecast_vs_real"]))
    _plot_error_by_horizon(config, payload["mae_by_horizon_day"], resolve_path(fig_dir["fig_error_by_horizon"]))
    made_30 = _plot_30_seeds(config, baselines_test, resolve_path(fig_dir["fig_30_seeds"]))

    out_json = resolve_path(config["paths"]["nn_metrics_file"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {out_json} | figuras: 03, 04{', 05' if made_30 else ''} en {config['paths']['figures_dir']}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluación offline de las redes del Módulo 1.")
    add_config_override_args(parser)
    args = parser.parse_args(argv)
    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(config["reproducibility"]["seed"], config["reproducibility"].get("deterministic_torch", True))
    run_evaluate(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
