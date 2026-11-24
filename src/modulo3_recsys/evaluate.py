"""Evaluación offline del Módulo 3 (ranking Top-K, cobertura, diversidad y UMAP).

Responsabilidad:
    - Cargar pesos y embeddings guardados en ``config['paths']['model_dir']`` sin reentrenar.
    - Calcular HitRate@K, Precision@K, Recall@K, NDCG@K, cobertura de catálogo y diversidad
      intra-lista para ``recommendation.eval_k_values``.
    - Generar proyecciones UMAP de embeddings y gráficos en ``reports/figures/m3/``.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from src.utils.config import add_config_override_args, load_module_config
from src.utils.seed import set_seed


def run_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Evalúa el sistema de recomendación M3 sobre el split de prueba offline."""
    raise NotImplementedError(
        "Evaluación de M3 aún no implementada. "
        "Debe cargar pesos desde config['paths']['model_dir'] y evaluar en data/processed/m3."
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluación offline del Módulo 3 (Recomendador).")
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m3", overrides=args.override, seed_override=args.seed)
    set_seed(config["reproducibility"]["seed"], config["reproducibility"].get("deterministic_torch", True))
    run_evaluate(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
