"""Página Streamlit del Módulo 2: Seguridad Vial y Detección de Distracción.

Contrato de interfaz:
    - Importa únicamente funciones puras desde ``src/modulo2_vision/predict.py``.
    - Lee parámetros por defecto (umbral de alerta, clases) desde ``config/m2_vision.yaml``.
    - Opera 100% offline leyendo pesos de ``models/m2/``.
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
    st.set_page_config(page_title="M2: Seguridad Vial", page_icon="🛡️", layout="wide")
    cfg = load_module_config("m2")
    alert_cfg = cfg["alerting"]

    st.title("🛡️ Módulo 2: Clasificación de Conducción Distractiva (Grad-CAM)")
    st.caption(cfg["module"]["description"])

    with st.sidebar:
        st.header("Configuración de Alerta")
        st.slider(
            "Umbral de probabilidad de distracción",
            min_value=0.10,
            max_value=0.95,
            value=float(alert_cfg["distraction_prob_threshold"]),
            step=0.05,
        )
        st.checkbox("Mostrar mapa de calor Grad-CAM", value=True)

    st.file_uploader("Cargar imagen de cabina (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.info(
        "Esqueleto listo. Conectar `src.modulo2_vision.predict.predict_driver_state` "
        "una vez entrenados y guardados los pesos en `models/m2/`."
    )


if __name__ == "__main__":
    render_page()
