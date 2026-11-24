"""Arquitecturas neuronales de pronóstico cuantílico para el Módulo 1.

Responsabilidad:
    - Definir las arquitecturas en PyTorch (MLP, LSTM, GRU cuantílica, TCN) y la función
      de pérdida Pinball Loss multi-cuantil (``time_series.quantiles``).
    - Construir modelos exclusivamente a partir de los hiperparámetros de ``config['model']``.
"""

from __future__ import annotations

from typing import Any, Dict


def build_model(config: Dict[str, Any], input_dim: int) -> Any:
    """Instancia la arquitectura neuronal configurada para M1."""
    raise NotImplementedError(
        "M1 models pendiente de implementación. "
        "Debe construir la red según config['model']['architecture']."
    )
