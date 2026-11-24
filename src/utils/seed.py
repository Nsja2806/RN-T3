"""Utilidades de reproducibilidad para Python, NumPy y PyTorch.

Garantiza determinismo estricto leyendo la semilla global desde ``config/base.yaml``
y fijando los generadores pseudoaleatorios de Python, NumPy y PyTorch (CPU/CUDA).
"""

from __future__ import annotations

import os
import random
from typing import List


def set_seed(seed: int, deterministic_torch: bool = True) -> int:
    """Fija las semillas globales de Python, NumPy y PyTorch.

    Args:
        seed: Entero no negativo usado como semilla pseudoaleatoria.
        deterministic_torch: Si es ``True``, activa el modo determinista de CuDNN
            y desactiva el autotuning de benchmarks para reproducibilidad exacta.

    Returns:
        La semilla entera aplicada.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                # Algunos entornos/ops antiguos no soportan warn_only; mantenemos cudnn determinista.
                pass
    except ImportError:
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """Inicializador reproducible para ``worker_init_fn`` en PyTorch DataLoader.

    Args:
        worker_id: Identificador entero del worker asignado por DataLoader.
    """
    try:
        import torch

        worker_seed = torch.initial_seed() % (2**32)
    except ImportError:
        worker_seed = (42 + worker_id) % (2**32)

    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def get_seed_list(base_seed: int, n_seeds: int) -> List[int]:
    """Genera una lista determinista de ``n_seeds`` semillas consecutivas.

    Útil para el modo ``--mode eval30`` exigido en las reglas de evaluación estadística.

    Args:
        base_seed: Semilla inicial (por defecto ``reproducibility.seed_list_start``).
        n_seeds: Cantidad total de semillas a generar (por ejemplo, 30).

    Returns:
        Lista ordenada de enteros ``[base_seed, base_seed + 1, ...]``.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds debe ser >= 1, recibido: {n_seeds}")
    return [int(base_seed) + i for i in range(int(n_seeds))]
