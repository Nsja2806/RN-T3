"""Evaluación offline del Módulo 1 (métricas puntuales, cuantílicas y de negocio).

Responsabilidad:
    - Cargar pesos guardados en ``config['paths']['model_dir']`` sin reentrenar.
    - Calcular MAE, RMSE, MAPE/sMAPE, Pinball Loss y cobertura empírica de intervalos
      (P10-P90) sobre el conjunto de prueba en ``config['paths']['processed_dir']``.
    - Generar gráficas diagnósticas en ``config['paths']['figures_dir']`` y exportar métricas.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from src.utils.config import add_config_override_args, load_module_config
from src.utils.seed import set_seed


def run_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evalúa el modelo entrenado de M1 sobre el conjunto de prueba offline."""
    raise NotImplementedError(
        "Evaluación de M1 aún no implementada. "
        "Debe cargar pesos desde config['paths']['model_dir'] y evaluar en data/processed/m1."
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluación offline del Módulo 1 (Demanda y Flota).")
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(config["reproducibility"]["seed"], config["reproducibility"].get("deterministic_torch", True))
    run_evaluate(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
