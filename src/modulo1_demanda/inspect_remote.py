"""Inspección remota vía API Socrata (SoQL) de datos.gov.co para el Módulo 1.

Responsabilidad (uso exclusivo en fase de exploración/ingesta previa al corte offline):
    1. Descargar el catálogo oficial DIVIPOLA (``gdxc-w37w``) en ``data/raw/m1/divipola.csv``.
    2. Consultar 5 filas de muestra de SuperTransporte (``eh75-8ah6``) para verificar los
       nombres exactos de las 10 columnas y sus tipos en Socrata.
    3. Validar mediante ``GROUP BY`` en SoQL las categorías reales de ``tipo_despacho``,
       ``clase_vehiculo`` y ``nivel_servicio``.
    4. Auditar volumen total, rango de fechas (``MIN``/``MAX``) y anomalías (``pasajeros <= 0``,
       ``despachos <= 0``, bucles origen==destino y códigos DIVIPOLA distintos de 5 dígitos).
    5. Extraer el Top 25 de rutas nacionales en origen (``SUM(pasajeros) DESC``) cruzado con
       DIVIPOLA, calculando días activos, clase de vehículo principal e indicador booleano de
       ruta inversa de alto volumen, además de la serie mensual histórica y una propuesta de
       fechas de corte (``train`` / ``val`` / ``test``).
    6. Guardar el reporte completo en ``reports/m1_raw_inspection.txt``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import add_config_override_args, load_module_config, resolve_path
from src.utils.seed import set_seed


def _request_with_retries(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]],
    timeout: int,
    max_retries: int,
) -> requests.Response:
    """Ejecuta una petición HTTP GET con reintentos exponenciales ante fallos transitorios."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code >= 500 and attempt < max_retries:
                time.sleep(2.0 * attempt)
                continue
            if not resp.ok:
                snippet = resp.text[:600].replace("\n", " ")
                raise RuntimeError(
                    f"HTTP {resp.status_code} consultando {url} con params={params}: {snippet}"
                )
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2.0 * attempt)
            else:
                raise
    raise RuntimeError(f"Fallo inesperado en _request_with_retries: {last_exc}")


def fetch_soql_df(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Ejecuta una consulta SoQL contra un endpoint CSV de Socrata y devuelve DataFrame + headers."""
    resp = _request_with_retries(session, url, params=params, timeout=timeout, max_retries=max_retries)
    text = resp.content.decode("utf-8-sig", errors="replace")
    if not text.strip():
        return pd.DataFrame(), dict(resp.headers)
    df = pd.read_csv(io.StringIO(text), dtype=str)
    return df, dict(resp.headers)


def download_divipola(
    session: requests.Session,
    url: str,
    limit: int,
    dest_path: Path,
    timeout: int,
    max_retries: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Descarga DIVIPOLA a disco y construye un diccionario {codigo_5_digitos: info}."""
    resp = _request_with_retries(
        session,
        url,
        params={"$limit": limit},
        timeout=timeout,
        max_retries=max_retries,
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    content = resp.content.decode("utf-8-sig", errors="replace")
    dest_path.write_text(content, encoding="utf-8")

    df_div = pd.read_csv(dest_path, dtype=str)
    cols_lower = {c.lower(): c for c in df_div.columns}

    # Identificar columnas de código/nombre municipio y departamento de forma robusta
    code_col = next(
        (cols_lower[k] for k in ("cod_mpio", "codigo_municipio", "c_digo_municipio") if k in cols_lower),
        None,
    )
    if code_col is None:
        code_col = [c for c in df_div.columns if "mpio" in c.lower() or "municipio" in c.lower()][0]

    mun_name_col = next(
        (cols_lower[k] for k in ("nom_mpio", "nombre_municipio") if k in cols_lower),
        None,
    )
    if mun_name_col is None:
        mun_cands = [c for c in df_div.columns if ("nom" in c.lower() or "mpio" in c.lower()) and c != code_col]
        mun_name_col = mun_cands[0]

    dpto_name_col = next(
        (cols_lower[k] for k in ("dpto", "nom_dpto", "nombre_departamento", "departamento") if k in cols_lower),
        None,
    )
    if dpto_name_col is None:
        dpto_cands = [c for c in df_div.columns if "dpto" in c.lower() or "depart" in c.lower()]
        dpto_name_col = dpto_cands[-1] if dpto_cands else mun_name_col

    lookup: Dict[str, Dict[str, str]] = {}
    for _, row in df_div.iterrows():
        raw_code = str(row.get(code_col, "")).strip()
        if not raw_code:
            continue
        code5 = raw_code.zfill(5) if raw_code.isdigit() and len(raw_code) <= 5 else raw_code
        lookup[code5] = {
            "raw_code": raw_code,
            "municipio": str(row.get(mun_name_col, "")).strip(),
            "departamento": str(row.get(dpto_name_col, "")).strip(),
        }
    return df_div, lookup


def _resolve_api_col(actual_cols: Sequence[str], candidates: Sequence[str]) -> str:
    lower_map = {c.lower(): c for c in actual_cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col in actual_cols:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    raise KeyError(f"No se pudo mapear ninguna columna candidata {candidates} en {list(actual_cols)}")


def _format_mun(code: str, div_lookup: Dict[str, Dict[str, str]]) -> str:
    c5 = code.zfill(5) if code.isdigit() and len(code) <= 5 else code
    info = div_lookup.get(c5)
    if info:
        return f"{info['municipio']}, {info['departamento']} ({c5})"
    return f"NO_EN_DIVIPOLA ({code})"


def _propose_temporal_splits(df_monthly: pd.DataFrame, horizon_days: int, lookback_days: int) -> List[str]:
    """Genera una propuesta razonada de cortes train/val/test a partir de la serie mensual."""
    lines: List[str] = []
    if df_monthly.empty:
        lines.append("No se obtuvieron registros mensuales para proponer cortes.")
        return lines

    df_m = df_monthly.copy()
    df_m["pasajeros_num"] = pd.to_numeric(df_m["total_pasajeros"], errors="coerce").fillna(0.0)
    df_m["registros_num"] = pd.to_numeric(df_m["total_registros"], errors="coerce").fillna(0.0)
    df_m["ym_dt"] = pd.to_datetime(df_m["anio_mes"], errors="coerce")
    df_m = df_m.dropna(subset=["ym_dt"]).sort_values("ym_dt").reset_index(drop=True)
    df_m["ym_str"] = df_m["ym_dt"].dt.strftime("%Y-%m")

    # Filtrar meses con operación sustancial (>= 5% de la mediana de meses activos) para ignorar colas espurias
    active_median = df_m.loc[df_m["registros_num"] > 100, "registros_num"].median()
    threshold_rows = max(1000.0, 0.05 * float(active_median if pd.notna(active_median) else 0.0))
    df_m["is_substantial"] = df_m["registros_num"] >= threshold_rows
    df_m["period_m"] = df_m["ym_dt"].dt.to_period("M")

    # Detectar bloques de meses estrictamente consecutivos sin huecos de cargue
    substantial = df_m[df_m["is_substantial"]].copy().reset_index(drop=True)
    if substantial.empty:
        substantial = df_m.copy().reset_index(drop=True)

    month_ints = substantial["period_m"].map(lambda p: p.year * 12 + p.month)
    substantial["block_id"] = (month_ints.diff().fillna(1) != 1).cumsum()

    blocks_summary: List[Tuple[int, pd.DataFrame]] = []
    for bid, b_df in substantial.groupby("block_id"):
        blocks_summary.append((int(bid), b_df.reset_index(drop=True)))

    lines.append(
        f"- Bloques continuos sin huecos mensuales (>= {threshold_rows:,.0f} reg/mes):"
    )
    for _, b_df in blocks_summary:
        s_str = b_df["ym_str"].iloc[0]
        e_str = b_df["ym_str"].iloc[-1]
        lines.append(f"  * {s_str} a {e_str} ({len(b_df)} meses consecutivos completos).")

    outlier_months = df_m[~df_m["is_substantial"]]["ym_str"].tolist()
    if outlier_months:
        lines.append(
            f"- Meses con sub-registro severo, huecos de cargue o fechas erróneas (< {threshold_rows:,.0f} reg/mes, excluir): "
            f"{', '.join(outlier_months)} (además de meses ausentes como 2024-01)."
        )

    # Tomar el bloque continuo más largo como bloque principal sin saltos temporales
    longest_block = max(blocks_summary, key=lambda item: len(item[1]))[1]
    b_start = longest_block["ym_dt"].iloc[0].strftime("%Y-%m-01")
    b_last_month = longest_block["ym_dt"].iloc[-1]
    b_end = (b_last_month + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")

    n_ref = len(longest_block)
    if n_ref == 36:
        # Partición canónica 24m / 6m / 6m para 3 años exactos (2021-2023)
        train_months, val_months, test_months = 24, 6, 6
    else:
        val_months = max(2, round(n_ref * 0.15))
        test_months = max(2, round(n_ref * 0.15))
        train_months = max(1, n_ref - val_months - test_months)

    train_end_dt = longest_block["ym_dt"].iloc[train_months - 1] + pd.offsets.MonthEnd(0)
    val_start_dt = train_end_dt + pd.Timedelta(days=1)
    val_end_dt = longest_block["ym_dt"].iloc[train_months + val_months - 1] + pd.offsets.MonthEnd(0)
    test_start_dt = val_end_dt + pd.Timedelta(days=1)

    lines.append(
        f"- Opción A (Recomendada - sobre el bloque continuo más largo {longest_block['ym_str'].iloc[0]}..{longest_block['ym_str'].iloc[-1]}, "
        f"{n_ref} meses sin huecos, compatible con lookback={lookback_days}d y horizon={horizon_days}d):"
    )
    lines.append(f"  * Train ({train_months} meses): {b_start}  ->  {train_end_dt.strftime('%Y-%m-%d')}")
    lines.append(
        f"  * Val   ({val_months} meses):  {val_start_dt.strftime('%Y-%m-%d')}  ->  {val_end_dt.strftime('%Y-%m-%d')}"
    )
    lines.append(f"  * Test  ({test_months} meses):  {test_start_dt.strftime('%Y-%m-%d')}  ->  {b_end}")

    # Si existe un segundo bloque reciente de >= 10 meses (ej. 2025-01..2025-11), documentar Opción B out-of-time
    secondary_blocks = [b for _, b in blocks_summary if len(b) >= 10 and b is not longest_block]
    if secondary_blocks:
        sec = secondary_blocks[-1]
        sec_context_start = sec["ym_dt"].iloc[0].strftime("%Y-%m-01")
        sec_eval_start = (sec["ym_dt"].iloc[1]).strftime("%Y-%m-01")
        sec_end = (sec["ym_dt"].iloc[-1] + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        t2_end = (longest_block["ym_dt"].iloc[n_ref - 7] + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        v2_start = (longest_block["ym_dt"].iloc[n_ref - 6]).strftime("%Y-%m-01")
        lines.append(
            f"- Opción B (Evaluación out-of-time reciente usando bloque limpio {sec['ym_str'].iloc[0]}..{sec['ym_str'].iloc[-1]} sin cruzar el hueco 2024 ni 2025-12):"
        )
        lines.append(f"  * Train (30 meses): {b_start}  ->  {t2_end}")
        lines.append(f"  * Val   (6 meses):  {v2_start}  ->  {b_end}")
        lines.append(
            f"  * Test  ({len(sec)-1} meses efectivos): {sec_eval_start}  ->  {sec_end} "
            f"(usando {sec_context_start}.. como ventana de contexto inicial de 30d)."
        )
    return lines


def run_remote_inspection(config: Dict[str, Any]) -> Path:
    """Ejecuta todas las consultas SoQL de auditoría y escribe el reporte en disco."""
    api_cfg = config["socrata_api"]
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    ts_cfg = config["time_series"]

    div_url: str = api_cfg["divipola_url"]
    st_url: str = api_cfg["supertransporte_url"]
    timeout: int = int(api_cfg.get("timeout_seconds", 120))
    max_retries: int = int(api_cfg.get("max_retries", 3))

    div_dest = resolve_path(paths_cfg["divipola_raw_file"])
    report_path = resolve_path(paths_cfg["raw_inspection_report"])

    session = requests.Session()
    session.headers.update({"User-Agent": "RN-T3-MLOps-Inspector/1.0"})

    report_lines: List[str] = []

    def log_section(title: str) -> None:
        sep = "=" * 100
        block = f"\n{sep}\n{title}\n{sep}"
        print(block)
        report_lines.append(block)

    def log_line(text: str = "") -> None:
        print(text)
        report_lines.append(text)

    log_line("REPORTE DE INSPECCIÓN REMOTA VÍA API SOCRATA (SoQL) - MÓDULO 1 (DEMANDA INTERMUNICIPAL)")
    log_line(f"Fecha de ejecución (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log_line(f"Endpoint SuperTransporte: {st_url}")
    log_line(f"Endpoint DIVIPOLA:        {div_url}")

    # -------------------------------------------------------------------------
    # 1. Descarga de DIVIPOLA
    # -------------------------------------------------------------------------
    log_section("1. DESCARGA Y VALIDACIÓN DEL CATÁLOGO OFICIAL DIVIPOLA (DANE)")
    df_div, div_lookup = download_divipola(
        session=session,
        url=div_url,
        limit=int(api_cfg.get("divipola_limit", 2000)),
        dest_path=div_dest,
        timeout=timeout,
        max_retries=max_retries,
    )
    log_line(f"Archivo guardado en: {div_dest}")
    log_line(f"Total filas DIVIPOLA descargadas: {len(df_div):,}")
    log_line(f"Total códigos municipales únicos normalizados (5 dígitos): {len(div_lookup):,}")
    log_line(f"Columnas exactas DIVIPOLA API: {list(df_div.columns)}")

    # -------------------------------------------------------------------------
    # 2. Muestra de 5 filas y esquema real de SuperTransporte
    # -------------------------------------------------------------------------
    log_section("2. MUESTRA DE 5 FILAS Y ESQUEMA EXACTO DE COLUMNAS EN LA API DE SUPERTRANSPORTE")
    sample_df, sample_headers = fetch_soql_df(
        session=session,
        url=st_url,
        params={"$limit": int(api_cfg.get("sample_limit", 5))},
        timeout=timeout,
        max_retries=max_retries,
    )
    actual_cols = list(sample_df.columns)
    soda_fields = json.loads(sample_headers.get("X-SODA2-Fields", "[]"))
    soda_types = json.loads(sample_headers.get("X-SODA2-Types", "[]"))
    soda_type_map = dict(zip(soda_fields, soda_types))

    log_line(f"Cantidad de columnas confirmadas: {len(actual_cols)}")
    for idx, col in enumerate(actual_cols, start=1):
        s_type = soda_type_map.get(col, "desconocido")
        log_line(f"  {idx:02d}. {col:<26s} (tipo Socrata: {s_type})")
    log_line("\nMuestra cruda ($limit=5):")
    log_line(sample_df.to_string(index=False))

    # Resolver nombres reales de columnas en el endpoint API
    orig_col = _resolve_api_col(actual_cols, [data_cfg["origin_col"], "municipio_origen_ruta", "municipio_origen"])
    dest_col = _resolve_api_col(
        actual_cols, [data_cfg["destination_col"], "municipio_destino_ruta", "municipio_destino"]
    )
    date_col = _resolve_api_col(actual_cols, [data_cfg["datetime_col"], "fecha_despacho"])
    tipo_col = _resolve_api_col(actual_cols, [data_cfg["dispatch_type_col"], "tipo_despacho"])
    clase_col = _resolve_api_col(actual_cols, [data_cfg["vehicle_class_col"], "clase_vehiculo"])
    nivel_col = _resolve_api_col(actual_cols, [data_cfg["service_level_col"], "nivel_servicio"])
    pas_col = _resolve_api_col(actual_cols, [data_cfg["target_col"], "pasajeros"])
    desp_col = _resolve_api_col(actual_cols, [data_cfg["dispatches_col"], "despachos"])

    # -------------------------------------------------------------------------
    # 3. Validación real de categorías vía GROUP BY en SoQL
    # -------------------------------------------------------------------------
    log_section("3. VALIDACIÓN REAL DE CATEGORÍAS EN SoQL (GROUP BY)")
    validated_origin_val = data_cfg.get("dispatch_type_origin_value", "ORIGEN")

    for cat_col in (tipo_col, clase_col, nivel_col):
        soql_cat = (
            f"SELECT {cat_col}, count(*) AS total_registros, "
            f"sum({pas_col}) AS total_pasajeros, sum({desp_col}) AS total_despachos "
            f"GROUP BY {cat_col} ORDER BY total_registros DESC"
        )
        df_cat, _ = fetch_soql_df(
            session=session,
            url=st_url,
            params={"$query": soql_cat},
            timeout=timeout,
            max_retries=max_retries,
        )
        log_line(f"\nCategorías exactas en '{cat_col}':")
        log_line(df_cat.to_string(index=False))

        if cat_col == tipo_col and not df_cat.empty:
            raw_vals = [str(v) for v in df_cat[cat_col].dropna().tolist()]
            exact_matches = [v for v in raw_vals if "ORIGEN" in v.upper()]
            if exact_matches:
                validated_origin_val = exact_matches[0]

    log_line(f"\n-> Valor exacto validado para despachos en origen: {tipo_col} = '{validated_origin_val}'")

    # -------------------------------------------------------------------------
    # 4. Totales globales, rango de fechas y conteo de anomalías
    # -------------------------------------------------------------------------
    log_section("4. VOLUMEN GLOBAL, RANGO HISTÓRICO Y AUDITORÍA DE ANOMALÍAS")
    soql_totals = (
        f"SELECT count(*) AS total_registros, "
        f"min({date_col}) AS fecha_min, max({date_col}) AS fecha_max, "
        f"sum({pas_col}) AS total_pasajeros, sum({desp_col}) AS total_despachos"
    )
    df_tot, _ = fetch_soql_df(session, st_url, {"$query": soql_totals}, timeout, max_retries)
    total_rows = int(float(df_tot["total_registros"].iloc[0]))
    fecha_min_global = str(df_tot["fecha_min"].iloc[0])
    fecha_max_global = str(df_tot["fecha_max"].iloc[0])
    total_pas_global = float(df_tot["total_pasajeros"].iloc[0])
    total_desp_global = float(df_tot["total_despachos"].iloc[0])

    log_line(f"Total registros en dataset remoto: {total_rows:,}")
    log_line(f"MIN({date_col}):                   {fecha_min_global}")
    log_line(f"MAX({date_col}):                   {fecha_max_global}")
    log_line(f"SUM({pas_col}) global:             {total_pas_global:,.0f}")
    log_line(f"SUM({desp_col}) global:            {total_desp_global:,.0f}")

    anomaly_queries = [
        (f"{pas_col} < 0", f"{pas_col} < 0"),
        (f"{pas_col} = 0", f"{pas_col} = 0"),
        (f"{pas_col} <= 0", f"{pas_col} <= 0"),
        (f"{pas_col} IS NULL", f"{pas_col} IS NULL"),
        (f"{desp_col} < 0", f"{desp_col} < 0"),
        (f"{desp_col} = 0", f"{desp_col} = 0"),
        (f"{desp_col} <= 0", f"{desp_col} <= 0"),
        (f"{desp_col} IS NULL", f"{desp_col} IS NULL"),
        (f"{date_col} IS NULL", f"{date_col} IS NULL"),
        (f"Bucles {orig_col} = {dest_col}", f"{orig_col} = {dest_col}"),
    ]
    log_line("\nConteo de anomalías numéricas, nulos y bucles:")
    for label, where_clause in anomaly_queries:
        q = f"SELECT count(*) AS cnt WHERE {where_clause}"
        df_a, _ = fetch_soql_df(session, st_url, {"$query": q}, timeout, max_retries)
        cnt = int(float(df_a["cnt"].iloc[0])) if not df_a.empty else 0
        pct = (100.0 * cnt / total_rows) if total_rows else 0.0
        log_line(f"  - {label:<38s}: {cnt:>12,d} filas ({pct:.3f}%)")

    # Auditoría exhaustiva de códigos DIVIPOLA (longitud != 5 y cruce con DANE) vía GROUP BY
    log_line("\nAuditoría de formato (5 dígitos) y validez DANE en códigos DIVIPOLA:")
    for mun_col in (orig_col, dest_col):
        q_mun = f"SELECT {mun_col}, count(*) AS cnt GROUP BY {mun_col} ORDER BY cnt DESC LIMIT 5000"
        df_m, _ = fetch_soql_df(session, st_url, {"$query": q_mun}, timeout, max_retries)
        df_m["cnt_num"] = pd.to_numeric(df_m["cnt"], errors="coerce").fillna(0).astype(int)

        raw_not_5_rows = 0
        raw_not_5_codes: List[Tuple[str, int]] = []
        not_in_div_exact_rows = 0
        not_in_div_zfill_rows = 0
        orphan_codes: List[Tuple[str, int]] = []

        for _, r in df_m.iterrows():
            val = "" if pd.isna(r[mun_col]) else str(r[mun_col]).strip()
            c_rows = int(r["cnt_num"])
            if len(val) != 5 or not val.isdigit():
                raw_not_5_rows += c_rows
                raw_not_5_codes.append((val or "NULL", c_rows))
            if val not in div_lookup:
                not_in_div_exact_rows += c_rows
            val_z5 = val.zfill(5) if val.isdigit() and len(val) <= 5 else val
            if val_z5 not in div_lookup:
                not_in_div_zfill_rows += c_rows
                orphan_codes.append((val or "NULL", c_rows))

        log_line(f"  Columna '{mun_col}' ({len(df_m):,} valores distintos):")
        log_line(
            f"    * Filas con string crudo != 5 dígitos:      {raw_not_5_rows:,} "
            f"({100.0 * raw_not_5_rows / total_rows:.3f}%) | códigos distintos={len(raw_not_5_codes)}"
        )
        if raw_not_5_codes:
            log_line(f"      Ejemplos crudos != 5 dígitos (top 10): {raw_not_5_codes[:10]}")
        log_line(
            f"    * Filas fuera de DIVIPOLA sin zfill(5):     {not_in_div_exact_rows:,} "
            f"({100.0 * not_in_div_exact_rows / total_rows:.3f}%)"
        )
        log_line(
            f"    * Filas fuera de DIVIPOLA aún con zfill(5): {not_in_div_zfill_rows:,} "
            f"({100.0 * not_in_div_zfill_rows / total_rows:.3f}%) | códigos huérfanos={len(orphan_codes)}"
        )
        if orphan_codes:
            log_line(f"      Top códigos huérfanos no presentes en DIVIPOLA: {orphan_codes[:15]}")

    # -------------------------------------------------------------------------
    # 5. Top 25 rutas nacionales en ORIGEN y verificación de pares ida/vuelta
    # -------------------------------------------------------------------------
    top_pool_limit = int(api_cfg.get("top_routes_pool_limit", 500))
    top_display = int(api_cfg.get("top_routes_display", 25))
    rev_rank_max = int(api_cfg.get("high_volume_reverse_rank_max", 50))

    log_section(
        f"5. TOP {top_display} RUTAS NACIONALES EN ORIGEN ({tipo_col} = '{validated_origin_val}', ORDEN POR SUM({pas_col}) DESC)"
    )
    soql_top_routes = (
        f"SELECT {orig_col}, {dest_col}, "
        f"sum({pas_col}) AS total_pasajeros, sum({desp_col}) AS total_despachos, "
        f"count(*) AS total_registros, min({date_col}) AS fecha_min, max({date_col}) AS fecha_max "
        f"WHERE {tipo_col} = '{validated_origin_val}' AND {orig_col} != {dest_col} "
        f"GROUP BY {orig_col}, {dest_col} "
        f"ORDER BY total_pasajeros DESC "
        f"LIMIT {top_pool_limit}"
    )
    df_routes, _ = fetch_soql_df(session, st_url, {"$query": soql_top_routes}, timeout, max_retries)
    df_routes["total_pasajeros_num"] = pd.to_numeric(df_routes["total_pasajeros"], errors="coerce").fillna(0.0)
    df_routes["total_despachos_num"] = pd.to_numeric(df_routes["total_despachos"], errors="coerce").fillna(0.0)
    df_routes["rank"] = range(1, len(df_routes) + 1)

    route_pool_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, r in df_routes.iterrows():
        o_raw = str(r[orig_col]).strip()
        d_raw = str(r[dest_col]).strip()
        route_pool_map[(o_raw, d_raw)] = {
            "rank": int(r["rank"]),
            "pasajeros": float(r["total_pasajeros_num"]),
            "despachos": float(r["total_despachos_num"]),
        }

    top25 = df_routes.head(top_display).copy().reset_index(drop=True)
    top25_pairs: List[Tuple[str, str]] = [
        (str(r[orig_col]).strip(), str(r[dest_col]).strip()) for _, r in top25.iterrows()
    ]

    # Consultar rutas inversas que no hayan entrado al Top 500
    missing_rev_pairs = [(d, o) for (o, d) in top25_pairs if (d, o) not in route_pool_map]
    if missing_rev_pairs:
        or_clauses = " OR ".join(
            [f"({orig_col} = '{ro}' AND {dest_col} = '{rd}')" for (ro, rd) in missing_rev_pairs]
        )
        q_rev = (
            f"SELECT {orig_col}, {dest_col}, sum({pas_col}) AS total_pasajeros, sum({desp_col}) AS total_despachos "
            f"WHERE {tipo_col} = '{validated_origin_val}' AND ({or_clauses}) "
            f"GROUP BY {orig_col}, {dest_col} LIMIT 100"
        )
        df_rev_extra, _ = fetch_soql_df(session, st_url, {"$query": q_rev}, timeout, max_retries)
        for _, r in df_rev_extra.iterrows():
            ro, rd = str(r[orig_col]).strip(), str(r[dest_col]).strip()
            route_pool_map[(ro, rd)] = {
                "rank": 9999,
                "pasajeros": float(pd.to_numeric(r["total_pasajeros"], errors="coerce") or 0.0),
                "despachos": float(pd.to_numeric(r["total_despachos"], errors="coerce") or 0.0),
            }

    # Consultar clase_vehiculo principal para las Top 25 rutas
    top25_or_filter = " OR ".join(
        [f"({orig_col} = '{o}' AND {dest_col} = '{d}')" for (o, d) in top25_pairs]
    )
    q_veh = (
        f"SELECT {orig_col}, {dest_col}, {clase_col}, sum({pas_col}) AS pas_clase "
        f"WHERE {tipo_col} = '{validated_origin_val}' AND ({top25_or_filter}) "
        f"GROUP BY {orig_col}, {dest_col}, {clase_col} LIMIT 2000"
    )
    df_veh, _ = fetch_soql_df(session, st_url, {"$query": q_veh}, timeout, max_retries)
    df_veh["pas_clase_num"] = pd.to_numeric(df_veh["pas_clase"], errors="coerce").fillna(0.0)
    main_vehicle_map: Dict[Tuple[str, str], str] = {}
    for (o, d), grp in df_veh.groupby([orig_col, dest_col]):
        grp_sorted = grp.sort_values("pas_clase_num", ascending=False)
        tot_p = grp_sorted["pas_clase_num"].sum()
        top_v = str(grp_sorted[clase_col].iloc[0])
        share = (100.0 * float(grp_sorted["pas_clase_num"].iloc[0]) / tot_p) if tot_p > 0 else 0.0
        main_vehicle_map[(str(o).strip(), str(d).strip())] = f"{top_v} ({share:.0f}%)"

    # Consultar dias_activos exactos por lotes (date_trunc_ymd) sin COUNT(DISTINCT)
    batch_size = int(api_cfg.get("active_days_batch_size", 5))
    active_days_map: Dict[Tuple[str, str], int] = {}
    for start_idx in range(0, len(top25_pairs), batch_size):
        batch = top25_pairs[start_idx : start_idx + batch_size]
        batch_or = " OR ".join([f"({orig_col} = '{o}' AND {dest_col} = '{d}')" for (o, d) in batch])
        q_days = (
            f"SELECT {orig_col}, {dest_col}, date_trunc_ymd({date_col}) AS dia, count(*) AS n "
            f"WHERE {tipo_col} = '{validated_origin_val}' AND ({batch_or}) "
            f"GROUP BY {orig_col}, {dest_col}, dia LIMIT 50000"
        )
        df_days, _ = fetch_soql_df(session, st_url, {"$query": q_days}, timeout, max_retries)
        if not df_days.empty:
            for (o, d), grp in df_days.groupby([orig_col, dest_col]):
                active_days_map[(str(o).strip(), str(d).strip())] = int(grp["dia"].nunique())

    # Umbral de alto volumen inverso: estar en Top rev_rank_max nacional (o al menos >= 25% del P25 del Top 25)
    p25_floor = 0.50 * float(top25["total_pasajeros_num"].iloc[-1]) if not top25.empty else 0.0

    table_rows: List[Dict[str, Any]] = []
    for idx, r in top25.iterrows():
        o_raw = str(r[orig_col]).strip()
        d_raw = str(r[dest_col]).strip()
        rev_info = route_pool_map.get((d_raw, o_raw), {"rank": None, "pasajeros": 0.0, "despachos": 0.0})
        rev_pas = float(rev_info["pasajeros"])
        rev_rank = rev_info["rank"]
        has_high_rev = bool(
            (rev_rank is not None and rev_rank <= rev_rank_max) or (rev_pas >= p25_floor and rev_pas > 0)
        )
        rev_rank_str = f"#{rev_rank}" if (rev_rank is not None and rev_rank <= top_pool_limit) else ">500/0"

        table_rows.append(
            {
                "#": idx + 1,
                "origen": _format_mun(o_raw, div_lookup),
                "destino": _format_mun(d_raw, div_lookup),
                "total_pasajeros": f"{float(r['total_pasajeros_num']):,.0f}",
                "total_despachos": f"{float(r['total_despachos_num']):,.0f}",
                "fecha_min": str(r["fecha_min"])[:10],
                "fecha_max": str(r["fecha_max"])[:10],
                "dias_activos": active_days_map.get((o_raw, d_raw), 0),
                "vehiculo_principal": main_vehicle_map.get((o_raw, d_raw), "N/D"),
                "inversa_alto_vol": has_high_rev,
                "inversa_detalle": f"{rev_pas:,.0f} pas ({rev_rank_str})",
            }
        )

    df_top25_table = pd.DataFrame(table_rows)
    log_line(
        f"Criterio booleano 'inversa_alto_vol': True si la ruta inversa (destino -> origen) "
        f"está en el Top {rev_rank_max} nacional de ORIGEN o supera {p25_floor:,.0f} pasajeros."
    )
    log_line("")
    log_line(df_top25_table.to_string(index=False))

    # Resumen de corredores bidireccionales simétricos (Ida <-> Vuelta dentro del pool Top 500)
    corridor_seen = set()
    corridor_list: List[Dict[str, Any]] = []
    for (o_raw, d_raw), ab in route_pool_map.items():
        if ab["rank"] > top_pool_limit:
            continue
        key = tuple(sorted([o_raw, d_raw]))
        if key in corridor_seen:
            continue
        corridor_seen.add(key)
        a, b = key
        info_ab = route_pool_map.get((a, b))
        info_ba = route_pool_map.get((b, a))
        if info_ab and info_ba and info_ab["pasajeros"] > 0 and info_ba["pasajeros"] > 0:
            corridor_list.append(
                {
                    "corredor_A": _format_mun(a, div_lookup),
                    "corredor_B": _format_mun(b, div_lookup),
                    "pasajeros_par": info_ab["pasajeros"] + info_ba["pasajeros"],
                    "pas_A_B": f"{info_ab['pasajeros']:,.0f} (#{info_ab['rank']})",
                    "pas_B_A": f"{info_ba['pasajeros']:,.0f} (#{info_ba['rank']})",
                    "ratio_simetria": min(info_ab["pasajeros"], info_ba["pasajeros"])
                    / max(info_ab["pasajeros"], info_ba["pasajeros"]),
                }
            )

    df_corridors = (
        pd.DataFrame(corridor_list)
        .sort_values("pasajeros_par", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    if not df_corridors.empty:
        df_corridors.insert(0, "#", range(1, len(df_corridors) + 1))
        df_corridors["pasajeros_par"] = df_corridors["pasajeros_par"].map(lambda x: f"{x:,.0f}")
        df_corridors["ratio_simetria"] = df_corridors["ratio_simetria"].map(lambda x: f"{x:.2f}")
        log_line("\nTOP 15 CORREDORES BIDIRECCIONALES (PARES IDA <-> VUELTA CON AMBOS SENTIDOS EN ALTO VOLUMEN):")
        log_line(df_corridors.to_string(index=False))

    # -------------------------------------------------------------------------
    # 6. Serie mensual agregada y propuesta de cortes temporales
    # -------------------------------------------------------------------------
    log_section(
        f"6. SERIE MENSUAL HISTÓRICA ({tipo_col} = '{validated_origin_val}') Y PROPUESTA DE CORTES TEMPORALES"
    )
    soql_monthly = (
        f"SELECT date_trunc_ym({date_col}) AS anio_mes, "
        f"sum({pas_col}) AS total_pasajeros, sum({desp_col}) AS total_despachos, "
        f"count(*) AS total_registros "
        f"WHERE {tipo_col} = '{validated_origin_val}' AND {date_col} IS NOT NULL "
        f"GROUP BY anio_mes ORDER BY anio_mes ASC LIMIT 5000"
    )
    df_monthly, _ = fetch_soql_df(session, st_url, {"$query": soql_monthly}, timeout, max_retries)
    if not df_monthly.empty:
        df_m_disp = df_monthly.copy()
        df_m_disp["anio_mes"] = df_m_disp["anio_mes"].astype(str).str.slice(0, 7)
        for num_c in ("total_pasajeros", "total_despachos", "total_registros"):
            df_m_disp[num_c] = (
                pd.to_numeric(df_m_disp[num_c], errors="coerce")
                .fillna(0.0)
                .map(lambda v: f"{v:,.0f}")
            )
        log_line(df_m_disp.to_string(index=False))

    log_line("\nPROPUESTA TÉCNICA DE CORTES TEMPORALES (TRAIN / VAL / TEST):")
    for proposal_line in _propose_temporal_splits(
        df_monthly=df_monthly,
        horizon_days=int(ts_cfg["forecast_horizon"]),
        lookback_days=int(ts_cfg["lookback_window"]),
    ):
        log_line(proposal_line)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\n[OK] Reporte guardado exitosamente en: {report_path}")
    return report_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspección remota vía API Socrata (SoQL) de SuperTransporte y DIVIPOLA (M1)."
    )
    add_config_override_args(parser)
    args = parser.parse_args(argv)

    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(
        config["reproducibility"]["seed"],
        config["reproducibility"].get("deterministic_torch", True),
    )
    run_remote_inspection(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
