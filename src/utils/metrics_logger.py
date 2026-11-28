"""Registro estandarizado de métricas para ejecuciones individuales y de 30 semillas.

Guarda resultados reproducibles en JSON y CSV, incluyendo estadísticas agregadas
(media, desviación estándar, mínimo y máximo) exigidas por el modo ``eval30``.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate_seed_metrics(
    per_seed_records: Sequence[Mapping[str, Any]],
    exclude_keys: Sequence[str] = ("seed", "run_index"),
) -> Dict[str, Dict[str, float]]:
    """Calcula media, desviación estándar muestral, mínimo y máximo por métrica.

    Args:
        per_seed_records: Lista de diccionarios (uno por semilla) con métricas numéricas.
        exclude_keys: Claves numéricas que no deben agregarse como métricas.

    Returns:
        Diccionario ``{metrica: {"mean": ..., "std": ..., "min": ..., "max": ..., "n": ...}}``.
    """
    if not per_seed_records:
        raise ValueError("per_seed_records no puede estar vacío.")

    excluded = set(exclude_keys)
    metric_values: Dict[str, List[float]] = {}

    for record in per_seed_records:
        for key, value in record.items():
            if key in excluded or not _is_numeric(value):
                continue
            metric_values.setdefault(key, []).append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for metric, values in sorted(metric_values.items()):
        n = len(values)
        mean_val = sum(values) / n
        if n > 1:
            variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
            std_val = math.sqrt(variance)
        else:
            std_val = 0.0
        summary[metric] = {
            "mean": mean_val,
            "std": std_val,
            "min": min(values),
            "max": max(values),
            "n": float(n),
        }

    return summary


def save_single_run_metrics(
    metrics: Mapping[str, Any],
    output_json: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Guarda métricas de una ejecución individual (``demo`` o ``single``) en JSON."""
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
        "metrics": dict(metrics),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return output_path


def save_eval30_results_grouped(
    per_seed_records: Sequence[Mapping[str, Any]],
    group_key: str,
    output_csv: str | Path,
    output_json: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Como ``save_eval30_results`` pero agregando por grupo (p. ej. arquitectura).

    El CSV conserva una fila por (grupo, semilla); el JSON contiene el resumen
    (mean, std, min, max, n) de cada métrica numérica dentro de cada grupo.
    ``min``/``max`` son mejor/peor valor según la métrica (menor es mejor salvo cobertura).
    """
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for record in per_seed_records:
        groups.setdefault(str(record[group_key]), []).append(record)

    summary = {name: aggregate_seed_metrics(recs) for name, recs in groups.items()}

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for record in per_seed_records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_seed_records:
            writer.writerow(dict(record))

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "group_key": group_key,
        "n_seeds_per_group": {name: len(recs) for name, recs in groups.items()},
        "metadata": dict(metadata or {}),
        "summary": summary,
        "per_seed": [dict(r) for r in per_seed_records],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def save_eval30_results(
    per_seed_records: Sequence[Mapping[str, Any]],
    output_csv: str | Path,
    output_json: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Exporta resultados de evaluación multi-semilla (30 semillas) en CSV y JSON.

    Args:
        per_seed_records: Lista de registros por semilla (cada uno con ``seed`` y métricas).
        output_csv: Ruta destino del archivo CSV con detalle por semilla.
        output_json: Ruta destino del archivo JSON con resumen (mean, std, min, max) y detalle.
        metadata: Metadatos del experimento (módulo, arquitectura, modo, etc.).

    Returns:
        Diccionario completo serializado en ``output_json``.
    """
    summary = aggregate_seed_metrics(per_seed_records)

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for record in per_seed_records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_seed_records:
            writer.writerow(dict(record))

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_seeds": len(per_seed_records),
        "metadata": dict(metadata or {}),
        "summary": summary,
        "per_seed": [dict(r) for r in per_seed_records],
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return payload
