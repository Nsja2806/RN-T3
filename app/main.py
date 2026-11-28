"""Aplicación principal de Streamlit — Sistema Integrado de Transporte (RN-T3).

Consume de forma 100% offline los módulos entrenados en ``models/`` a través de sus
contratos de inferencia (``src/modulo*/predict.py``) y centraliza los accesos al
reporte técnico, registro de alucinaciones y videos demostrativos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml


def main() -> None:
    base_cfg = load_yaml(PROJECT_ROOT / "config" / "base.yaml")
    st_cfg = base_cfg.get("streamlit", {})

    st.set_page_config(
        page_title=st_cfg.get("page_title", "Sistema Integrado de Transporte - RN-T3"),
        page_icon="🚌",
        layout=st_cfg.get("layout", "wide"),
    )

    st.title("🚌 Sistema Integrado de Transporte — Plataforma de Inteligencia Artificial")
    st.markdown(
        """
        Bienvenido a la aplicación interactiva del proyecto **RN-T3** (modo **100% offline**).
        Utiliza el menú lateral para navegar por los tres módulos independientes:

        1. **📈 Módulo 1 — Demanda y Flota (`1_Demanda_y_Flota`)**:
           Pronóstico probabilístico de demanda (P10 / P50 / P90) y dimensionamiento de flota y turnos de conductores.
        2. **🛡️ Módulo 2 — Seguridad Vial (`2_Seguridad_Vial`)**:
           Detección de conducción distractiva con CNN / Transfer Learning y explicabilidad visual Grad-CAM.
        3. **🧭 Módulo 3 — Recomendador de Viajes (`3_Recomendador_Viajes`)**:
           Recomendación personalizada de destinos (NCF / Two-Towers) y simulación de ROI de campañas.
        """
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Documentación y Reportes")
        st.markdown(
            """
            - **Reporte Técnico (Blog):** `reports/reporte_tecnico.md` *(reemplazar por enlace público si aplica)*
            - **Registro de Alucinaciones y Auditoría IA:** `reports/hallucinations.md`
            - **Reglas de Reproducibilidad MLOps:** `AGENTS.md`
            """
        )

    with col2:
        st.subheader("🎬 Videos Demostrativos")
        st.markdown(
            """
            - **Video Pitch / Demo General:** *(Pendiente de insertar enlace)*
            - **Demo Técnica M1 (Demanda y Flota):** *(Pendiente de insertar enlace)*
            - **Demo Técnica M2 (Seguridad Vial):** *(Pendiente de insertar enlace)*
            - **Demo Técnica M3 (Recomendador de Viajes):** *(Pendiente de insertar enlace)*
            """
        )

    st.success(
        "✅ **Módulo 1 (Demanda, Flota y Conductores)** activo y verificado 100% offline "
        "(`models/m1/gru_best.pt` + evaluación en 30 semillas). Selecciona **1 Demanda y Flota** "
        "en la barra lateral para explorar pronósticos y escenarios operativos en tiempo real."
    )


if __name__ == "__main__":
    main()
