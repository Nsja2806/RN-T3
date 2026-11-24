"""Contrato de inferencia pura para el Módulo 3 (consumido por Streamlit).

Responsabilidad:
    - Cargar exclusivamente pesos entrenados, embeddings y catálogos desde ``models/m3/``
      y ``data/processed/m3/`` sin reentrenar ni conectarse a Internet.
    - Devolver las Top-K recomendaciones para un usuario (con manejo de cold-start según
      ``recommendation.cold_start_fallback``) y estimación de impacto de campaña.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def load_artifacts(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Carga el modelo entrenado de M3, embeddings e índices desde ``models/m3/``."""
    raise NotImplementedError(
        "M3 predict.load_artifacts pendiente de implementación (carga offline desde models/m3)."
    )


def recommend_destinations(
    user_id: Any,
    top_k: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Genera las Top-K recomendaciones de destinos para un usuario."""
    raise NotImplementedError(
        "M3 predict.recommend_destinations pendiente de implementación."
    )
