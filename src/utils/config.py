"""Carga jerárquica de configuraciones YAML y soporte para overrides por CLI.

Fusiona ``config/base.yaml`` con el archivo específico de cada módulo
(``m1_demanda.yaml``, ``m2_vision.yaml``, ``m3_recsys.yaml``) y aplica
sobreescrituras tipo ``clave.subclave=valor`` recibidas desde ``argparse``.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"

MODULE_CONFIG_MAP: Dict[str, Path] = {
    "m1": PROJECT_ROOT / "config" / "m1_demanda.yaml",
    "m2": PROJECT_ROOT / "config" / "m2_vision.yaml",
    "m3": PROJECT_ROOT / "config" / "m3_recsys.yaml",
}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Carga un archivo YAML y devuelve un diccionario Python."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML es requerido para cargar archivos de configuración. "
            "Instala las dependencias con: pip install -r requirements.txt"
        ) from exc

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"El archivo YAML debe definir un mapeo en la raíz: {file_path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Fusiona recursivamente dos diccionarios sin mutar los originales.

    Las claves de ``override`` prevalecen sobre ``base``. Cuando ambos valores
    son diccionarios, se fusionan de manera recursiva.
    """
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override_value(raw_value: str) -> Any:
    """Convierte el valor textual de un override CLI a su tipo nativo usando YAML."""
    try:
        import yaml

        return yaml.safe_load(raw_value)
    except Exception:
        return raw_value


def apply_overrides(
    config: Mapping[str, Any],
    overrides: Optional[Sequence[str] | Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aplica sobreescrituras por clave punteada sobre un diccionario de configuración.

    Soporta:
        - Secuencia de strings tipo ``["training.epochs=5", "model.dropout=0.1"]``
        - Diccionario tipo ``{"training.epochs": 5, "reproducibility.seed": 123}``

    Args:
        config: Diccionario de configuración base/fusionado.
        overrides: Overrides provenientes de CLI o llamadas programáticas.

    Returns:
        Nuevo diccionario con los overrides aplicados.
    """
    updated: Dict[str, Any] = copy.deepcopy(dict(config))
    if not overrides:
        return updated

    if isinstance(overrides, Mapping):
        items = list(overrides.items())
    else:
        items = []
        for entry in overrides:
            if "=" not in entry:
                raise ValueError(
                    f"Override inválido '{entry}'. Usa el formato 'clave.subclave=valor'."
                )
            key_path, raw_val = entry.split("=", 1)
            key_path = key_path.strip()
            if not key_path:
                raise ValueError(f"Clave vacía en override '{entry}'.")
            items.append((key_path, parse_override_value(raw_val.strip())))

    for dotted_key, value in items:
        parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not parts:
            raise ValueError(f"Clave de override inválida: '{dotted_key}'")

        cursor: Dict[str, Any] = updated
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value

    return updated


def load_module_config(
    module: str,
    overrides: Optional[Sequence[str] | Mapping[str, Any]] = None,
    base_config_path: Optional[str | Path] = None,
    module_config_path: Optional[str | Path] = None,
    seed_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Carga ``base.yaml``, lo fusiona con el config del módulo y aplica overrides CLI.

    Args:
        module: Identificador del módulo (``"m1"``, ``"m2"`` o ``"m3"``).
        overrides: Lista de overrides ``clave.subclave=valor`` o diccionario.
        base_config_path: Ruta alternativa a ``config/base.yaml`` (opcional).
        module_config_path: Ruta alternativa al YAML del módulo (opcional).
        seed_override: Entero opcional para sobreescribir ``reproducibility.seed``.

    Returns:
        Diccionario de configuración completo y listo para consumo.
    """
    module_key = module.lower().strip()
    if module_config_path is None:
        if module_key not in MODULE_CONFIG_MAP:
            valid = ", ".join(sorted(MODULE_CONFIG_MAP.keys()))
            raise ValueError(f"Módulo desconocido '{module}'. Opciones válidas: {valid}")
        resolved_module_cfg = MODULE_CONFIG_MAP[module_key]
    else:
        resolved_module_cfg = Path(module_config_path)

    resolved_base_cfg = Path(base_config_path) if base_config_path else BASE_CONFIG_PATH

    base_cfg = load_yaml(resolved_base_cfg)
    mod_cfg = load_yaml(resolved_module_cfg)
    merged = deep_merge(base_cfg, mod_cfg)

    if overrides:
        merged = apply_overrides(merged, overrides)

    if seed_override is not None:
        merged = apply_overrides(merged, {"reproducibility.seed": int(seed_override)})

    return merged


def resolve_path(path_value: str | Path) -> Path:
    """Resuelve rutas relativas respecto a la raíz del repositorio (Windows y Linux)."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def add_config_override_args(parser: argparse.ArgumentParser) -> None:
    """Agrega argumentos estándar de configuración y override a un parser argparse."""
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sobreescribe reproducibility.seed definido en config/base.yaml.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Sobreescribe parámetros YAML usando notación punteada "
            "(ej. --override training.epochs=10 --override model.dropout=0.25)."
        ),
    )
