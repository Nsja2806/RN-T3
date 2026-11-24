"""Modelos baseline clásicos de visión para el Módulo 2.

Responsabilidad:
    - Implementar baselines reproducibles (p. ej. histogramas de color/HOG + Regresión Logística
      o Random Forest, y CNN ligera desde cero) sobre exactamente las mismas particiones por
      conductor que el modelo principal de Transfer Learning.
"""

from __future__ import annotations

from typing import Any, Dict


def evaluate_baselines(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Entrena y evalúa los baselines de clasificación de conducción distractiva."""
    raise NotImplementedError(
        "M2 baselines pendiente de implementación."
    )
