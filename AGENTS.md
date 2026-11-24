# AGENTS.md — Reglas Obligatorias de Ingeniería y MLOps

Este documento define las reglas no negociables para agentes de IA y desarrolladores que trabajen en este repositorio (**Sistema Integrado de Transporte — RN-T3**). Todas las futuras sesiones deben cumplir estrictamente estos principios.

---

## 1. Reproducibilidad Estricta
- Todo script de preprocesamiento, entrenamiento, evaluación o inferencia que involucre aleatoriedad **DEBE** invocar `set_seed(seed)` desde `src/utils/seed.py`.
- La semilla global por defecto se lee desde `config/base.yaml` (`reproducibility.seed: 42`).
- `set_seed(seed)` debe fijar simultáneamente:
  - `PYTHONHASHSEED`
  - `random.seed(seed)`
  - `numpy.random.seed(seed)`
  - `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)` y modo determinista de CuDNN (`deterministic=True`, `benchmark=False`).
- Nunca usar números aleatorios sin sembrar ni depender del orden no determinista de conjuntos (`set`) o del sistema de archivos (`os.listdir` sin `sorted()`).

## 2. Cero Hardcoding (Config-Driven + CLI Overrides)
- **Prohibido** dejar hiperparámetros, rutas de archivos, umbrales de decisión o constantes de negocio (capacidad de buses, jornada de conductores, presupuesto de campaña, umbrales de alerta) quemados en el código Python.
- Todo valor configurable debe declararse en `config/base.yaml` o en el YAML del módulo correspondiente (`config/m1_demanda.yaml`, `config/m2_vision.yaml`, `config/m3_recsys.yaml`).
- Toda configuración debe cargarse mediante `src/utils/config.py` (`load_module_config(...)`) y soportar sobreescritura por CLI vía `argparse` (`--override clave.subclave=valor`).

## 3. Presupuesto Justo y Evaluación de 30 Semillas
- Cada `train.py` (`m1`, `m2`, `m3`) **DEBE** soportar exactamente los siguientes modos de ejecución mediante `--mode`:
  1. `--mode demo`: ejecución rápida de verificación de extremo a extremo en **< 60 segundos** (pocas épocas / subconjunto reducido definido en `training.modes.demo`).
  2. `--mode single`: entrenamiento completo con **1 semilla** (`config/base.yaml` o `--seed`).
  3. `--mode eval30`: entrenamiento y evaluación estadística sobre **30 semillas** (`reproducibility.n_seeds_eval: 30`), guardando los resultados por semilla y el resumen estadístico (**media, desviación estándar, mínimo y máximo**) en **JSON y CSV** usando `src/utils/metrics_logger.py`.
- La comparación entre modelos baseline y modelos neuronales debe realizarse bajo el mismo esquema de partición de datos y el mismo presupuesto de evaluación.

## 4. Offline First (Ejecución sin Internet)
- Ningún script de entrenamiento, evaluación, predicción ni la aplicación web en Streamlit puede descargar datasets, pesos preentrenados ni activos externos desde Internet durante su ejecución.
- Todos los scripts deben leer únicamente datos procesados desde `data/processed/<modulo>/` y pesos/artefactos entrenados desde `models/<modulo>/`.
- `data/raw/*` está excluido en `.gitignore` (excepto `.gitkeep`). Los artefactos livianos y reproducibles en `data/processed/` y `models/` se rastrean en Git para garantizar ejecución 100% offline.

## 5. Contrato de Interfaz entre Modelos y Streamlit
- Cada módulo debe exponer funciones puras de inferencia en su archivo `predict.py`:
  - `src/modulo1_demanda/predict.py`
  - `src/modulo2_vision/predict.py`
  - `src/modulo3_recsys/predict.py`
- Reglas del contrato:
  - Las funciones de `predict.py` **solo cargan pesos desde `models/<modulo>/`**, validan entradas y devuelven estructuras limpias (diccionarios / DataFrames / arrays).
  - **Nunca** deben disparar entrenamiento ni descargas al ser importadas o llamadas desde `app/pages/`.
  - Las páginas de `app/pages/` importan únicamente funciones de `predict.py` (y lógica de negocio/visualización de su módulo) para mantener desacople total entre capa de ML y capa de UI.

## 6. Convenciones para Trabajo en Paralelo sin Conflictos de Git
Para permitir que 2 parejas trabajen simultáneamente con fricción mínima:
- **Aislamiento estricto por módulo**:
  - Cada módulo es dueño exclusivo de su carpeta en `src/moduloX_*`, `config/mX_*.yaml`, `data/processed/mX/`, `models/mX/`, `reports/figures/mX/` y su página en `app/pages/X_*.py`.
- **Archivos compartidos** (`config/base.yaml`, `src/utils/*`, `run.py`, `app/main.py`, `requirements.txt`, `reports/reporte_tecnico.md`, `reports/hallucinations.md`):
  - Realizar cambios pequeños, atómicos y coordinados mediante Pull Requests cortos.
- **Comandos unificados**:
  - Toda ejecución estándar debe poder lanzarse desde la raíz del repositorio con `run.py` en Windows y Linux:
    - `python run.py train --module <m1|m2|m3> --mode <demo|single|eval30>`
    - `python run.py evaluate --module <m1|m2|m3>`
    - `python run.py app`
