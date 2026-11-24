"""Evaluación offline del Módulo 2 (métricas multiclase y análisis de alertas).

Responsabilidad:
    - Cargar pesos guardados en ``config['paths']['model_dir']`` sin reentrenar.
    - Calcular Accuracy, Macro-F1, Recall de clases de alto riesgo, matriz de confusión y
      tasa de falsas alarmas según ``config['alerting']['distraction_prob_threshold']``.
    - Exportar gráficos y métricas a ``reports/figures/m2/`` y ``reports/metrics/``.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from src.utils.config import add_config_override_args, load_module_config
from src.utils.seed import set_seed


def run_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evalúa el modelo entrenado de M2 sobre el conjunto de prueba offline."""
    raise NotImplementedError(
        "Evaluación de M2 aún no implementada. "
        "Debe cargar pesos desde config['paths']['model_dir'] y evaluar en data/processed/m2."
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluación offline del Módulo 2 (Seguridad Vial).")
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m2", overrides=args.override, seed_override=args.seed)
    set_seed(config["reproducibility"]["seed"], config["reproducibility"].get("deterministic_torch", True))
    run_evaluate(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
