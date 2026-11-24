"""Preprocesamiento y deduplicación de imágenes para el Módulo 2 (Seguridad Vial).

Responsabilidad:
    - Leer imágenes crudas desde ``config['paths']['raw_dir']`` (fase de preparación).
    - Detectar y eliminar imágenes duplicadas o casi-duplicadas (pHash / distancia Hamming
      configurada en ``data.deduplication``).
    - Garantizar split estratificado agrupado por conductor (``group_by_driver: true``) para
      evitar fuga de datos entre train, val y test.
    - Exportar metadatos/manifests y muestras livianas procesadas a ``paths.processed_dir``
      para ejecución 100% offline.
"""

from __future__ import annotations

from typing import Any, Dict


def run_preprocess_dedup(config: Dict[str, Any]) -> None:
    """Ejecuta limpieza, deduplicación y partición sin fuga por conductor para M2."""
    raise NotImplementedError(
        "M2 preprocess_dedup pendiente de implementación. "
        "Debe aplicar deduplicación y split por conductor guardando resultados en data/processed/m2."
    )
