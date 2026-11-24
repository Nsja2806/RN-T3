"""Contrato de inferencia pura para el Módulo 1 (consumido por Streamlit).

Responsabilidad:
    - Cargar exclusivamente pesos entrenados y artefactos de escalado desde ``models/m1/``.
    - Exponer funciones puras de predicción de demanda (P10/P50/P90) e integración con
      ``business_fleet.py`` sin disparar jamás entrenamientos ni descargas externas.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def load_artifacts(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Carga modelo entrenado y escaladores de M1 desde ``models/m1/``."""
    raise NotImplementedError(
        "M1 predict.load_artifacts pendiente de implementación (carga offline desde models/m1)."
    )


def predict_demand(
    history_window: Any,
    config: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Genera pronóstico cuantílico de demanda para el horizonte configurado."""
    raise NotImplementedError(
        "M1 predict.predict_demand pendiente de implementación."
    )
