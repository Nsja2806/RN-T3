"""Contrato de inferencia pura para el Módulo 2 (consumido por Streamlit).

Responsabilidad:
    - Cargar pesos entrenados exclusivamente desde ``models/m2/`` sin reentrenar ni
      descargar artefactos de Internet.
    - Clasificar una imagen de cabina, devolver probabilidades por clase, estado de alerta
      (usando ``config['alerting']['distraction_prob_threshold']``) y superposición Grad-CAM.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def load_artifacts(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Carga el modelo entrenado de M2 y metadatos desde ``models/m2/``."""
    raise NotImplementedError(
        "M2 predict.load_artifacts pendiente de implementación (carga offline desde models/m2)."
    )


def predict_driver_state(
    image: Any,
    include_gradcam: bool = True,
    threshold: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Clasifica el estado del conductor y evalúa si dispara alerta de distracción."""
    raise NotImplementedError(
        "M2 predict.predict_driver_state pendiente de implementación."
    )
