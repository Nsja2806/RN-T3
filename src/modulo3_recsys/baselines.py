"""Modelos baseline de recomendación para el Módulo 3.

Responsabilidad:
    - Implementar baselines reproducibles: Popularidad global, Filtrado Colaborativo
      basado en ítems/usuarios (k-NN / coseno) y SVD/Factorización Matricial clásica.
    - Evaluar bajo los mismos valores de ``recommendation.eval_k_values`` y el mismo split.
"""

from __future__ import annotations

from typing import Any, Dict


def evaluate_baselines(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Entrena y evalúa los baselines de recomendación de M3."""
    raise NotImplementedError(
        "M3 baselines pendiente de implementación."
    )
