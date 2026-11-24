"""Preprocesamiento de series de tiempo para el Módulo 1 (Demanda y Flota).

Responsabilidad:
    - Leer datos crudos desde ``paths.raw_dir`` (uso exclusivo durante preparación inicial).
    - Limpiar valores faltantes, estandarizar frecuencia temporal (``data.frequency``),
      construir rezagos (``time_series.lag_features``), ventanas móviles y variables de calendario.
    - Realizar partición estrictamente cronológica (train/val/test) sin fuga temporal y
      ajustar el escalador únicamente con train.
    - Guardar los conjuntos procesados y el escalador en ``paths.processed_dir`` para
      garantizar ejecución 100% offline.
"""

from __future__ import annotations

from typing import Any, Dict


def run_preprocess(config: Dict[str, Any]) -> None:
    """Ejecuta el pipeline de preprocesamiento temporal para M1."""
    raise NotImplementedError(
        "M1 preprocess pendiente de implementación. "
        "Debe leer de config['paths']['raw_dir'] y exportar a config['paths']['processed_dir']."
    )
