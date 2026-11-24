"""Modelos baseline clásicos/estadísticos para el Módulo 1 (Demanda).

Responsabilidad:
    - Implementar baselines reproducibles (Naive estacional, Media móvil, Regresión lineal /
      Ridge con rezagos) bajo la misma partición temporal y métricas que las redes neuronales.
    - Proveer una referencia justa de comparación (presupuesto justo).
"""

from __future__ import annotations

from typing import Any, Dict


def evaluate_baselines(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Entrena y evalúa los modelos baseline del Módulo 1."""
    raise NotImplementedError(
        "M1 baselines pendiente de implementación. "
        "Debe evaluar baselines sobre config['paths']['processed_dir']."
    )
