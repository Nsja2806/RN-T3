"""Análisis Exploratorio de Datos (EDA) de imágenes para el Módulo 2.

Responsabilidad:
    - Analizar distribución de clases, balance por conductor, estadísticas de brillo/resolución
      y reporte de duplicados eliminados leyendo desde ``paths.processed_dir``.
    - Guardar figuras en ``paths.figures_dir``.
"""

from __future__ import annotations

from typing import Any, Dict


def run_eda(config: Dict[str, Any]) -> None:
    """Genera estadísticas y visualizaciones exploratorias del dataset de conducción."""
    raise NotImplementedError(
        "M2 EDA pendiente de implementación. "
        "Debe guardar figuras en config['paths']['figures_dir']."
    )
