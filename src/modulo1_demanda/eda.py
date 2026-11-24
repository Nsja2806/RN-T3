"""Análisis Exploratorio de Datos (EDA) temporal para el Módulo 1.

Responsabilidad:
    - Analizar estacionalidad horaria/diaria/semanal, tendencia, autocorrelación (ACF/PACF)
      y detección de anomalías en la demanda procesada (``paths.processed_dir``).
    - Exportar figuras listas para el reporte técnico en ``paths.figures_dir``.
"""

from __future__ import annotations

from typing import Any, Dict


def run_eda(config: Dict[str, Any]) -> None:
    """Genera estadísticas y gráficos exploratorios de la serie de demanda."""
    raise NotImplementedError(
        "M1 EDA pendiente de implementación. "
        "Debe leer de config['paths']['processed_dir'] y guardar figuras en config['paths']['figures_dir']."
    )
