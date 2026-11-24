"""Preprocesamiento de interacciones usuario-destino para el Módulo 3 (RecSys).

Responsabilidad:
    - Leer interacciones crudas desde ``config['paths']['raw_dir']`` (preparación inicial).
    - Filtrar usuarios/destinos por interacciones mínimas (``min_user_interactions``,
      ``min_item_interactions``), indexar IDs contiguos y construir atributos de contenido.
    - Aplicar estrategia de split (``leave_one_out`` o ``temporal``) evitando fuga futura y
      generar muestreo negativo determinista (``negative_sampling_ratio``).
    - Exportar tablas procesadas y diccionarios de mapeo a ``paths.processed_dir`` (offline).
"""

from __future__ import annotations

from typing import Any, Dict


def run_preprocess(config: Dict[str, Any]) -> None:
    """Ejecuta el pipeline de preprocesamiento de interacciones para M3."""
    raise NotImplementedError(
        "M3 preprocess pendiente de implementación. "
        "Debe exportar datasets y diccionarios de IDs a config['paths']['processed_dir']."
    )
