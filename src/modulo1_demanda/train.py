"""Entrenamiento reproducible del Módulo 1 (modos demo, single y eval30).

Responsabilidad:
    - Leer datos procesados offline desde ``config['paths']['processed_dir']``.
    - Soportar ``--mode demo`` (< 60s), ``--mode single`` (1 semilla) y ``--mode eval30``
      (30 semillas guardando media, std, min y max en JSON y CSV con ``metrics_logger``).
    - Guardar pesos entrenados y metadatos en ``config['paths']['model_dir']``.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from src.utils.config import add_config_override_args, load_module_config
from src.utils.seed import set_seed


def run_train(config: Dict[str, Any], mode: str = "demo") -> Dict[str, Any]:
    """Ejecuta el entrenamiento de M1 según el modo solicitado."""
    raise NotImplementedError(
        f"Entrenamiento de M1 (mode='{mode}') aún no implementado. "
        "Implementar lectura offline de data/processed/m1, entrenamiento PyTorch y guardado en models/m1."
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento del Módulo 1 (Demanda y Flota).")
    parser.add_argument("--mode", choices=["demo", "single", "eval30"], default="demo")
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(config["reproducibility"]["seed"], config["reproducibility"].get("deterministic_torch", True))
    run_train(config=config, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
