"""Explicabilidad visual mediante Grad-CAM para el Módulo 2 (Seguridad Vial).

Responsabilidad:
    - Registrar hooks de activaciones y gradientes sobre la capa objetivo definida en
      ``config['model']['gradcam_target_layer']``.
    - Generar mapas de calor superpuestos sobre la imagen original para auditar si el modelo
      atiende al conductor/teléfono/manos y no a atajos espurios del fondo.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def generate_gradcam(
    model: Any,
    image_tensor: Any,
    target_class: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Calcula el mapa de activación Grad-CAM para una imagen de entrada."""
    raise NotImplementedError(
        "M2 gradcam.generate_gradcam pendiente de implementación."
    )
