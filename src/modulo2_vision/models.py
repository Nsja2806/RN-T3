"""Arquitecturas CNN y Transfer Learning para el Módulo 2 (Seguridad Vial).

Responsabilidad:
    - Construir arquitecturas PyTorch (CNN personalizada, MobileNetV3, ResNet18, EfficientNet)
      cargando pesos locales únicamente cuando ``model.pretrained_local_weights`` esté definido
      (estrictamente offline, sin descargas automáticas de torchvision en tiempo de ejecución).
"""

from __future__ import annotations

from typing import Any, Dict


def build_model(config: Dict[str, Any]) -> Any:
    """Instancia la arquitectura de visión configurada en ``config['model']``."""
    raise NotImplementedError(
        "M2 models pendiente de implementación. "
        "Debe respetar ejecución offline (weights=None o carga desde ruta local)."
    )
