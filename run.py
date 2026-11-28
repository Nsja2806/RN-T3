#!/usr/bin/env python3
"""Punto de entrada unificado (CLI) para el Sistema Integrado de Transporte (RN-T3).

Compatible con Windows y Linux mediante comandos de una sola línea:
    - python run.py train --module m1 --mode demo
    - python run.py train --module m2 --mode single
    - python run.py train --module m3 --mode eval30
    - python run.py evaluate --module m1
    - python run.py app
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent

# Asegura que el directorio raíz esté en sys.path al invocar desde cualquier carpeta
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PACKAGE_MAP: Dict[str, str] = {
    "m1": "src.modulo1_demanda",
    "m2": "src.modulo2_vision",
    "m3": "src.modulo3_recsys",
}


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser principal con los subcomandos train, evaluate y app."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "CLI reproducible para entrenar, evaluar y desplegar los 3 módulos del "
            "Sistema Integrado de Transporte (M1: Demanda/Flota, M2: Visión/Seguridad, "
            "M3: Recomendación de Destinos)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos de uso:\n"
            "  python run.py train --module m1 --mode demo\n"
            "  python run.py train --module m2 --mode single --seed 42\n"
            "  python run.py train --module m3 --mode eval30 --override training.epochs=15\n"
            "  python run.py evaluate --module m1\n"
            "  python run.py app\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMANDO")

    # ------------------------------------------------------------------ train
    train_parser = subparsers.add_parser(
        "train",
        help="Entrena un módulo (m1, m2 o m3) en modo demo, single o eval30.",
    )
    train_parser.add_argument(
        "--module",
        "-m",
        required=True,
        choices=sorted(MODULE_PACKAGE_MAP.keys()),
        help="Módulo a entrenar: m1 (demanda), m2 (visión) o m3 (recsys).",
    )
    train_parser.add_argument(
        "--mode",
        default="demo",
        choices=["demo", "single", "eval30", "hpo"],
        help=(
            "Modo de ejecución: 'demo' (<60s), 'single' (1 semilla completa) "
            "o 'eval30' (30 semillas con reporte media/std/min/max). "
            "'hpo' (búsqueda de hiperparámetros acotada) está disponible solo en m1."
        ),
    )
    train_parser.add_argument(
        "--arch",
        default="all",
        choices=["mlp", "gru", "all"],
        help="Arquitectura neuronal a entrenar (solo m1): mlp, gru o all (por defecto).",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional para sobreescribir reproducibility.seed de config/base.yaml.",
    )
    train_parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Sobreescribe parámetros de config YAML (repetible, ej. training.epochs=5).",
    )

    # --------------------------------------------------------------- evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evalúa los pesos guardados de un módulo (m1, m2 o m3) de forma offline.",
    )
    eval_parser.add_argument(
        "--module",
        "-m",
        required=True,
        choices=sorted(MODULE_PACKAGE_MAP.keys()),
        help="Módulo a evaluar: m1 (demanda), m2 (visión) o m3 (recsys).",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional para sobreescribir reproducibility.seed de config/base.yaml.",
    )
    eval_parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Sobreescribe parámetros de config YAML (repetible).",
    )

    # -------------------------------------------------------------------- app
    app_parser = subparsers.add_parser(
        "app",
        help="Inicia la aplicación web interactiva en Streamlit (modo offline).",
    )
    app_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Puerto HTTP para el servidor de Streamlit (por defecto: 8501).",
    )
    app_parser.add_argument(
        "--headless",
        action="store_true",
        help="Ejecuta Streamlit en modo headless (sin abrir navegador automáticamente).",
    )

    return parser


def _handle_train(args: argparse.Namespace) -> int:
    from src.utils.config import load_module_config
    from src.utils.seed import set_seed

    config = load_module_config(
        module=args.module,
        overrides=args.override,
        seed_override=args.seed,
    )
    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"].get("deterministic_torch", True))
    set_seed(seed, deterministic_torch=deterministic)

    pkg = MODULE_PACKAGE_MAP[args.module]
    train_mod = importlib.import_module(f"{pkg}.train")
    extra_kwargs = {"arch": args.arch} if args.module == "m1" else {}
    if args.mode == "hpo" and args.module != "m1":
        raise SystemExit("El modo 'hpo' solo está implementado para el módulo m1.")
    train_mod.run_train(config=config, mode=args.mode, **extra_kwargs)
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    from src.utils.config import load_module_config
    from src.utils.seed import set_seed

    config = load_module_config(
        module=args.module,
        overrides=args.override,
        seed_override=args.seed,
    )
    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"].get("deterministic_torch", True))
    set_seed(seed, deterministic_torch=deterministic)

    pkg = MODULE_PACKAGE_MAP[args.module]
    eval_mod = importlib.import_module(f"{pkg}.evaluate")
    eval_mod.run_evaluate(config=config)
    return 0


def _handle_app(args: argparse.Namespace) -> int:
    app_entrypoint = PROJECT_ROOT / "app" / "main.py"
    if not app_entrypoint.exists():
        raise FileNotFoundError(f"No se encontró el entrypoint de Streamlit: {app_entrypoint}")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_entrypoint),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


def main(argv: Optional[List[str]] = None) -> int:
    """Función principal de la CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return _handle_train(args)
    if args.command == "evaluate":
        return _handle_evaluate(args)
    if args.command == "app":
        return _handle_app(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
