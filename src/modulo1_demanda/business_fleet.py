"""Lógica de negocio de dimensionamiento de flota y turnos de conductores (M1).

Responsabilidad:
    - Traducir los pronósticos cuantílicos de demanda (P10, P50, P90) a requerimientos
      operativos de buses por hora, flota de reserva, número de conductores necesarios
      respetando la jornada laboral máxima y descansos (``config['fleet_business']``),
      y estimación de costos operativos vs. demanda insatisfecha.
    - Cero constantes quemadas: todos los parámetros operativos se leen de ``config``.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence


def compute_fleet_plan(
    demand_forecast: Sequence[float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Calcula el plan óptimo de flota, turnos de conductores y costos estimados."""
    raise NotImplementedError(
        "M1 business_fleet pendiente de implementación. "
        "Usar parámetros de config['fleet_business'] para dimensionar buses y conductores."
    )
