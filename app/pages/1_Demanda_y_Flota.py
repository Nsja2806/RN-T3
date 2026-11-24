"""Página Streamlit del Módulo 1: Pronóstico de Demanda y Planificación de Flota.

Contrato de interfaz:
    - Importa únicamente funciones puras desde ``src/modulo1_demanda/predict.py`` y
      ``src/modulo1_demanda/business_fleet.py``.
    - Lee parámetros por defecto desde ``config/m1_demanda.yaml`` vía ``load_module_config('m1')``.
    - Nunca entrena modelos ni descarga datos externos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_module_config


def render_page() -> None:
    st.set_page_config(page_title="M1: Demanda y Flota", page_icon="📈", layout="wide")
    cfg = load_module_config("m1")
    fleet_cfg = cfg["fleet_business"]

    st.title("📈 Módulo 1: Pronóstico de Demanda y Planificación de Flota")
    st.caption(cfg["module"]["description"])

    with st.sidebar:
        st.header("Parámetros Operativos de Flota")
        st.number_input(
            "Capacidad por bus (pasajeros)",
            min_value=10,
            max_value=200,
            value=int(fleet_cfg["bus_capacity_passengers"]),
        )
        st.slider(
            "Ocupación objetivo",
            min_value=0.50,
            max_value=1.00,
            value=float(fleet_cfg["target_occupancy_ratio"]),
            step=0.05,
        )
        st.number_input(
            "Jornada máxima conductor (horas)",
            min_value=4.0,
            max_value=12.0,
            value=float(fleet_cfg["max_driver_shift_hours"]),
            step=0.5,
        )

    st.info(
        "Esqueleto listo. Conectar `src.modulo1_demanda.predict.predict_demand` y "
        "`src.modulo1_demanda.business_fleet.compute_fleet_plan` una vez entrenados los pesos en `models/m1/`."
    )


if __name__ == "__main__":
    render_page()
