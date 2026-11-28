"""Suite automática de pruebas unitarias y de integridad MLOps para el Módulo 1.

Verifica en < 10 segundos y 100% offline:
    1. Integridad estructural de ``data/processed/m1/demanda_diaria.parquet`` (8,760 filas,
       0 nulos en ``pasajeros``, 8 rutas oficiales y partición exacta train/val/test).
    2. Cero fuga temporal (*no data leakage*): prueba adversarial mutando el futuro ``> t_0``
       con valores extremos (``999,999``) para demostrar invarianza exacta de ``build_windows``,
       ``NNPredictor`` y ``predict_demand``, más verificación de rezagos a 30 días (``*_h30``).
    3. Propiedades matemáticas de ``gru_best.pt``: positividad estricta (``mu > 0``) y
       ausencia total de cruce cuantílico (``q90 >= mu``).
    4. Reglas de negocio y normativa colombiana 2026: doble conductor obligatorio en
       ``Bogotá <-> Cali`` (Res. 315/2013 Art. 6), 1 conductor en ``Armenia <-> Pereira`` y
       monotonicidad decreciente de la flota respecto a la ocupación objetivo ``rho``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulo1_demanda.business_fleet import compute_fleet_plan
from src.modulo1_demanda.nn_data import NNPredictor, build_windows, load_arrays, load_checkpoint
from src.modulo1_demanda.predict import load_artifacts, predict_demand
from src.utils.config import load_module_config, resolve_path


@pytest.fixture(scope="module")
def m1_config():
    return load_module_config("m1")


@pytest.fixture(scope="module")
def m1_artifacts(m1_config):
    return load_artifacts(config=m1_config, arch="gru")


def test_parquet_integrity_and_splits(m1_config):
    """Verifica 8,760 filas exactas, 0 nulos en pasajeros y las 8 rutas configuradas."""
    parquet_path = resolve_path(m1_config["paths"]["processed_dataset_file"])
    assert parquet_path.exists(), f"Falta archivo procesado: {parquet_path}"
    assert parquet_path.stat().st_size < 2 * 1024 * 1024, "El parquet supera 2 MB"

    df = pd.read_parquet(parquet_path)
    expected_routes = {str(r["route_id"]) for r in m1_config["data"]["selected_routes"]}

    assert len(expected_routes) == 8
    assert len(df) == 8 * 1095 == 8760
    assert int(df["pasajeros"].isna().sum()) == 0
    assert (df["pasajeros"] >= 0).all()
    assert set(df["route_id"].unique()) == expected_routes

    split_counts = df["split"].value_counts().to_dict()
    assert split_counts == {"train": 5840, "val": 1448, "test": 1472}


def test_zero_temporal_leakage_adversarial_and_h30_features(m1_config, m1_artifacts):
    """Demuestra que mutar valores futuros (> t_0) jamás altera las entradas ni predicciones."""
    arrays_clean = load_arrays(m1_config)
    lookback = int(m1_config["time_series"]["lookback_window"])
    horizon = int(m1_config["time_series"]["forecast_horizon"])
    floor_frac = float(m1_config["model"]["scale_floor_frac"])

    origin_dt = pd.Timestamp("2023-09-15")
    i0 = arrays_clean.date_to_idx[origin_dt]

    # Clonar y contaminar TODO el futuro t > t_0 con valores extremos
    arrays_corrupted = copy.deepcopy(arrays_clean)
    arrays_corrupted.y[:, i0 + 1 :] = 999_999.0

    w_clean = build_windows(
        arrays_clean,
        np.array([i0]),
        lookback=lookback,
        horizon=horizon,
        scale_floor_frac=floor_frac,
        route_mean=arrays_clean.route_train_mean,
    )
    w_corrupted = build_windows(
        arrays_corrupted,
        np.array([i0]),
        lookback=lookback,
        horizon=horizon,
        scale_floor_frac=floor_frac,
        route_mean=arrays_corrupted.route_train_mean,
    )

    # Todos los tensores de entrada al modelo deben ser idénticos bit a bit
    for key in ("hist_y", "hist_cal", "fut_cal", "log_level", "scale", "route"):
        assert np.array_equal(
            w_clean[key].numpy(), w_corrupted[key].numpy()
        ), f"Fuga temporal detectada en tensor de entrada '{key}'"

    # Verificar que las predicciones de GRU no cambian en absoluto ante el futuro contaminado
    gru_model = m1_artifacts["models"]["gru"]
    pred_clean = NNPredictor(gru_model, arrays_clean, lookback, horizon, floor_frac).predict_from_origin(
        arrays_clean.df, origin_dt, horizon
    )
    pred_corrupted = NNPredictor(
        gru_model, arrays_corrupted, lookback, horizon, floor_frac
    ).predict_from_origin(arrays_corrupted.df, origin_dt, horizon)

    np.testing.assert_allclose(pred_clean["y_pred_p50"], pred_corrupted["y_pred_p50"], rtol=0, atol=0)
    np.testing.assert_allclose(pred_clean["y_pred_p90"], pred_corrupted["y_pred_p90"], rtol=0, atol=0)

    # Verificar además que las features tabulares *_h30 en la fila t solo usan <= t - 30
    sub = arrays_clean.df[arrays_clean.df["route_id"] == "11001_50001"].sort_values("fecha_despacho").reset_index(drop=True)
    y_vals = sub["pasajeros"].to_numpy(dtype=np.float64)
    for t_check in (60, 250, 800, 1050):
        assert np.isclose(sub.loc[t_check, "lag_30"], y_vals[t_check - 30])
        assert np.isclose(
            sub.loc[t_check, "roll_mean_7_h30"],
            np.mean(y_vals[t_check - 36 : t_check - 29]),
            atol=1e-3,
        )
        assert np.isclose(
            sub.loc[t_check, "roll_mean_28_h30"],
            np.mean(y_vals[t_check - 57 : t_check - 29]),
            atol=1e-3,
        )


def test_gru_positivity_and_no_quantile_crossing(m1_config, m1_artifacts):
    """Verifica mu > 0 estricto y q90 >= mu en todas las rutas y los 30 pasos del horizonte."""
    ckpt_path = resolve_path(m1_config["paths"]["model_dir"]) / "gru_best.pt"
    assert ckpt_path.exists()
    _, ckpt = load_checkpoint(ckpt_path)
    assert ckpt["n_parameters"] > 0

    for origin_str in ("2023-07-15", "2023-12-01", "2023-12-31"):
        res = predict_demand(
            artifacts=m1_artifacts,
            route_id=None,  # las 8 rutas simultáneamente
            origin_date=origin_str,
            service_level="p90",
        )
        fc = res["forecast_df"]
        assert len(fc) == 8 * 30 == 240
        assert np.isfinite(fc["pred_p50"]).all()
        assert np.isfinite(fc["pred_p90"]).all()
        assert (fc["pred_p50"] > 0.0).all(), "El pronóstico puntual debe ser estrictamente > 0"
        assert (fc["pred_p90"] >= fc["pred_p50"]).all(), "Violación de cruce cuantílico: P90 < P50"


def test_colombian_regulations_and_occupancy_monotonicity(m1_config, m1_artifacts):
    """Verifica doble conductor en Bogotá<->Cali, 1 en Armenia<->Pereira y monotonicidad en rho."""
    # 1) Regla Res. 315 de 2013 Art. 6: > 8h por sentido exige 2 conductores
    res_cali = predict_demand(
        artifacts=m1_artifacts,
        route_id="11001_76001",
        origin_date="2023-12-01",
        service_level="p90",
    )
    rp_cali = res_cali["fleet_plan"]["daily_route_plan"]
    assert set(rp_cali["route_id"].unique()) == {"11001_76001", "76001_11001"}
    assert (rp_cali["crew_per_dispatch"] == 2).all()

    # 2) Armenia <-> Pereira (1.5h por sentido <= 8h) exige 1 conductor por despacho
    res_arm = predict_demand(
        artifacts=m1_artifacts,
        route_id="63001_66001",
        origin_date="2023-12-01",
        service_level="p90",
    )
    rp_arm = res_arm["fleet_plan"]["daily_route_plan"]
    assert set(rp_arm["route_id"].unique()) == {"63001_66001", "66001_63001"}
    assert (rp_arm["crew_per_dispatch"] == 1).all()

    # 3) Monotonicidad: aumentar la ocupación objetivo rho nunca puede incrementar la flota requerida
    base_fc = predict_demand(
        artifacts=m1_artifacts,
        route_id=None,
        origin_date="2023-12-01",
        service_level="p90",
    )["forecast_df"]

    rhos = [0.65, 0.75, 0.85, 0.95]
    daily_fleets = []
    for rho in rhos:
        plan = compute_fleet_plan(
            demand_df=base_fc,
            config=m1_config,
            demand_col="pred_p90",
            date_col="target_date",
            target_occupancy_override=rho,
        )
        daily_fleets.append(plan["daily_system_totals"]["total_fleet_with_reserve"].to_numpy())

    for prev_fl, next_fl in zip(daily_fleets[:-1], daily_fleets[1:]):
        assert (next_fl <= prev_fl).all(), (
            "Violación de monotonicidad: aumentar ocupación objetivo incrementó la flota requerida."
        )
