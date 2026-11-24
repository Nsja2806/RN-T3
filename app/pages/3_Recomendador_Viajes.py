"""Página Streamlit del Módulo 3: Recomendador de Destinos y Campañas.

Contrato de interfaz:
    - Importa únicamente funciones puras desde ``src/modulo3_recsys/predict.py`` y
      ``src/modulo3_recsys/business_campaign.py``.
    - Lee parámetros por defecto desde ``config/m3_recsys.yaml``.
    - Opera 100% offline leyendo pesos y embeddings de ``models/m3/``.
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
    st.set_page_config(page_title="M3: Recomendador de Viajes", page_icon="🧭", layout="wide")
    cfg = load_module_config("m3")
    rec_cfg = cfg["recommendation"]
    camp_cfg = cfg["campaign_business"]

    st.title("🧭 Módulo 3: Recomendación de Destinos y Simulador de Campañas")
    st.caption(cfg["module"]["description"])

    with st.sidebar:
        st.header("Parámetros de Recomendación y Campaña")
        st.slider(
            "Top-K recomendaciones",
            min_value=1,
            max_value=20,
            value=int(rec_cfg["top_k"]),
        )
        st.number_input(
            "Presupuesto de campaña (USD)",
            min_value=100.0,
            max_value=100000.0,
            value=float(camp_cfg["campaign_budget_usd"]),
            step=250.0,
        )
        st.slider(
            "Tasa de descuento promocional",
            min_value=0.0,
            max_value=0.50,
            value=float(camp_cfg["discount_rate"]),
            step=0.05,
        )

    st.info(
        "Esqueleto listo. Conectar `src.modulo3_recsys.predict.recommend_destinations` y "
        "`src.modulo3_recsys.business_campaign.optimize_campaign` una vez entrenados los pesos en `models/m3/`."
    )


if __name__ == "__main__":
    render_page()
