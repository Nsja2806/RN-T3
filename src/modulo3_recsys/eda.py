"""Análisis Exploratorio de Datos (EDA) de interacciones para el Módulo 3.

Responsabilidad:
    - Medir dispersión (sparsity) de la matriz usuario-destino, distribución de cola larga
      (long-tail), tasa de cold-start y concentración por categoría/región.
    - Guardar gráficos diagnósticos en ``config['paths']['figures_dir']``.
"""

from __future__ import annotations

from typing import Any, Dict


def run_eda(config: Dict[str, Any]) -> None:
    """Genera estadísticas y visualizaciones exploratorias de interacciones en M3."""
    raise NotImplementedError(
        "M3 EDA pendiente de implementación. "
        "Debe leer de config['paths']['processed_dir'] y exportar figuras a config['paths']['figures_dir']."
    )
