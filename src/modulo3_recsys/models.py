"""Arquitecturas neuronales de recomendación (NCF y Two-Towers) para el Módulo 3.

Responsabilidad:
    - Implementar en PyTorch modelos de embeddings y recomendación: Matrix Factorization
      neuronal (MF), Neural Collaborative Filtering (NCF) y arquitectura Two-Towers con
      similitud coseno escalada por temperatura (``model.temperature``).
    - Permitir extracción limpia de embeddings de usuarios y destinos para visualización UMAP.
"""

from __future__ import annotations

from typing import Any, Dict


def build_model(config: Dict[str, Any], num_users: int, num_items: int) -> Any:
    """Instancia la arquitectura RecSys definida en ``config['model']['architecture']``."""
    raise NotImplementedError(
        "M3 models pendiente de implementación."
    )
