"""Entrenamiento reproducible de las redes de M1: modos demo, hpo, single y eval30 (CPU, offline).

Responsabilidad:
    - ``--mode demo``: 1 semilla, pocas épocas y submuestra (< 60 s); checkpoint temporal en ``models/m1/demo``.
    - ``--mode hpo``: presupuesto acotado (6 configuraciones por arquitectura, ``hpo.configs``) evaluado en
      ``val`` con early stopping; guarda ``reports/metrics/m1_hpo_results.json``, ``fig_m1_hpo.png`` y
      ``models/m1/best_hparams.json``.
    - ``--mode single``: entrena la mejor configuración de MLP y GRU con la semilla base, guarda curvas de
      aprendizaje y persiste pesos + escaladores en ``models/m1/``.
    - ``--mode eval30``: 30 semillas (``seed .. seed+29``) por arquitectura, evaluadas en ``test`` con el
      protocolo rolling-origin de ``baselines.py``; guarda media/std/mejor/peor en CSV y JSON.
    Train usa solo ``train`` para ajustar pesos; ``val`` para early stopping/HPO; ``test`` solo en eval30/evaluate.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulo1_demanda.baselines import (
    FUTURE_CALENDAR_COLS,
    evaluate_forecast_records,
    run_rolling_origin_evaluation,
)
from src.modulo1_demanda.models import MultiHorizonLoss, build_model, count_trainable_parameters
from src.modulo1_demanda.nn_data import (
    NNPredictor,
    model_forward,
    predict_windows,
    prepare_context,
    save_checkpoint,
)
from src.utils.config import add_config_override_args, load_module_config, resolve_path
from src.utils.metrics_logger import save_eval30_results_grouped, save_single_run_metrics
from src.utils.seed import get_seed_list, set_seed

ARCHS_ALL: Tuple[str, ...] = ("mlp", "gru")
ARCH_LABEL = {"mlp": "MLP global", "gru": "GRU global"}


# --------------------------------------------------------------------------------------
# Hiperparámetros
# --------------------------------------------------------------------------------------
def default_hparams(arch: str, config: Dict[str, Any]) -> Dict[str, Any]:
    m, t = config["model"][arch], config["training"]
    return {
        "hidden_size": int(m["hidden_size"]),
        "num_layers": int(m["num_layers"]),
        "dropout": float(m["dropout"]),
        "learning_rate": float(t["learning_rate"]),
        "weight_decay": float(t["weight_decay"]),
    }


def resolve_hparams(arch: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Defaults del YAML sobrescritos por la mejor configuración del HPO si existe."""
    hp = default_hparams(arch, config)
    best_path = resolve_path(config["paths"]["best_hparams_file"])
    if best_path.exists():
        stored = json.loads(best_path.read_text(encoding="utf-8"))
        if arch in stored:
            hp.update(stored[arch]["hparams"])
    return hp


def _resolve_archs(arch: str) -> List[str]:
    if arch == "all":
        return list(ARCHS_ALL)
    if arch in ARCHS_ALL:
        return [arch]
    raise ValueError(f"--arch inválido '{arch}'. Opciones: mlp, gru, all.")


# --------------------------------------------------------------------------------------
# Entrenamiento de un modelo
# --------------------------------------------------------------------------------------
def _split_loss(model, windows, loss_fn: MultiHorizonLoss) -> Tuple[float, float, float]:
    mu, q90 = predict_windows(model, windows)
    total, poisson, pinball = loss_fn(mu, q90, windows["y_scaled"])
    return float(total), float(poisson), float(pinball)


def fit_model(
    arch: str,
    hp: Dict[str, Any],
    ctx: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
    epochs: int,
    patience: int,
    max_samples: Optional[int] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Entrena con early stopping sobre la pérdida de ``val`` y devuelve el mejor modelo + historial."""
    t_cfg = config["training"]
    set_seed(seed, bool(config["reproducibility"].get("deterministic_torch", True)))
    model = build_model(
        arch,
        hp,
        num_routes=len(ctx["arrays"].route_ids),
        lookback=ctx["lookback"],
        horizon=ctx["horizon"],
        n_cal=ctx["n_cal"],
        emb_dim=ctx["emb_dim"],
    )
    loss_fn = MultiHorizonLoss(float(t_cfg["lambda_pinball"]), float(t_cfg["pinball_quantile"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(hp["learning_rate"]), weight_decay=float(hp["weight_decay"])
    )
    clip = float(t_cfg.get("gradient_clip_norm", 1.0))
    batch_size = int(t_cfg["batch_size"])
    train_w, val_w = ctx["windows"]["train"], ctx["windows"]["val"]

    gen = torch.Generator().manual_seed(int(seed))
    n_all = train_w["route"].shape[0]
    selected = torch.randperm(n_all, generator=gen)[:max_samples] if max_samples and max_samples < n_all else torch.arange(n_all)

    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_poisson": [], "val_pinball": []}
    best_val, best_epoch, best_state, bad_epochs = float("inf"), 0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = selected[torch.randperm(len(selected), generator=gen)]
        running, seen = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            batch = {k: v[idx] for k, v in train_w.items()}
            mu, q90 = model_forward(model, batch, slice(None))
            loss, _, _ = loss_fn(mu, q90, batch["y_scaled"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running += float(loss.detach()) * len(idx)
            seen += len(idx)
        val_total, val_pois, val_pin = _split_loss(model, val_w, loss_fn)
        history["train_loss"].append(running / max(seen, 1))
        history["val_loss"].append(val_total)
        history["val_poisson"].append(val_pois)
        history["val_pinball"].append(val_pin)
        if val_total < best_val - 1e-6:
            best_val, best_epoch, bad_epochs = val_total, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history.update(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        epochs_run=len(history["train_loss"]),
        train_seconds=round(time.perf_counter() - t0, 3),
        n_parameters=count_trainable_parameters(model),
    )
    return model, history


def rolling_eval(model, ctx: Dict[str, Any], origins) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evalúa con el protocolo rolling-origin compartido con los baselines."""
    predictor = NNPredictor(model, ctx["arrays"], ctx["lookback"], ctx["horizon"], ctx["floor_frac"])
    records = run_rolling_origin_evaluation(ctx["arrays"].df, origins, ctx["horizon"], predictor.predict_from_origin)
    metrics = evaluate_forecast_records(records, ctx["mase_scales"], ctx["horizon_blocks"], ctx["quantile"])
    return metrics, records


def _fmt(m: Dict[str, Any]) -> str:
    o = m["overall"]
    return (
        f"MAE={o['mae']:.1f} RMSE={o['rmse']:.1f} MASE={o['mase']:.3f} "
        f"PinballP90={o['pinball_p90']:.1f} CobP90={o['coverage_p90_pct']:.1f}%"
    )


def _set_threads(config: Dict[str, Any]) -> None:
    n = config["training"].get("torch_num_threads")
    if n:
        torch.set_num_threads(int(n))


# --------------------------------------------------------------------------------------
# Modos
# --------------------------------------------------------------------------------------
def _run_demo(config, archs, ctx) -> Dict[str, Any]:
    t0 = time.perf_counter()
    mode_cfg = config["training"]["modes"]["demo"]
    seed = int(config["reproducibility"]["seed"])
    out_dir = resolve_path(config["paths"]["demo_model_dir"])
    results: Dict[str, Any] = {}
    for arch in archs:
        hp = default_hparams(arch, config)
        model, hist = fit_model(
            arch, hp, ctx, config, seed, int(mode_cfg["epochs"]),
            int(config["training"]["early_stopping_patience"]), mode_cfg.get("max_samples"),
        )
        val_metrics, _ = rolling_eval(model, ctx, ctx["val_origins"])
        save_checkpoint(out_dir / f"{arch}_demo.pt", model, arch, hp, ctx,
                        {"mode": "demo", "seed": seed, "best_epoch": hist["best_epoch"]})
        results[arch] = {"history": hist, "val": val_metrics}
        print(
            f"[demo:{arch}] params={hist['n_parameters']:,} épocas={hist['epochs_run']} "
            f"mejor={hist['best_epoch']} train={hist['train_seconds']:.1f}s | VAL {_fmt(val_metrics)}"
        )
    elapsed = time.perf_counter() - t0
    limit = float(config["runtime"].get("demo_max_seconds", 60))
    print(f"[demo] Tiempo total: {elapsed:.1f}s (límite {limit:.0f}s) -> {'OK' if elapsed < limit else 'EXCEDIDO'}")
    return results


def _plot_hpo(results: Dict[str, Any], baselines_val: Dict[str, float], out_path: Path, dpi: int = 300) -> None:
    archs = [a for a in ARCHS_ALL if a in results]
    fig, axes = plt.subplots(1, len(archs), figsize=(7.2 * len(archs), 4.8), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        trials = results[arch]["trials"]
        maes = [t["val"]["overall"]["mae"] for t in trials]
        best = int(np.argmin(maes))
        colors = ["#d95f02" if i == best else "#5c88b5" for i in range(len(trials))]
        ax.bar(range(1, len(trials) + 1), maes, color=colors)
        for i, t in enumerate(trials):
            hp = t["hparams"]
            ax.text(i + 1, maes[i] + 2, f"h={hp['hidden_size']}\nd={hp['dropout']}\nlr={hp['learning_rate']:g}\nwd={hp['weight_decay']:g}\n{t['history']['train_seconds']:.0f}s",
                    ha="center", va="bottom", fontsize=6.5)
        for name, val, style in (("seasonal_naive", baselines_val.get("seasonal_naive"), "--"),
                                 ("ridge_calendar", baselines_val.get("ridge_calendar"), ":")):
            if val is not None:
                ax.axhline(val, color="#333333", linestyle=style, linewidth=1.1, label=f"{name} ({val:.0f})")
        ax.set_ylim(min(maes + [v for v in baselines_val.values()]) * 0.8, max(maes + list(baselines_val.values())) * 1.25)
        ax.set_xlabel("Configuración HPO")
        ax.set_ylabel("MAE en val (rolling origin, H=30)")
        ax.set_title(f"{ARCH_LABEL[arch]} – mejor: config {best + 1}", fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("HPO M1: MAE en validación por configuración (6 por arquitectura)", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _run_hpo(config, archs, ctx) -> Dict[str, Any]:
    hpo_cfg = config["hpo"]
    seed = int(config["reproducibility"]["seed"])
    sel_metric = str(hpo_cfg.get("selection_metric", "mae"))
    results: Dict[str, Any] = {}
    t_all = time.perf_counter()
    for arch in archs:
        trials: List[Dict[str, Any]] = []
        for i, cand in enumerate(hpo_cfg["configs"][arch], start=1):
            hp = default_hparams(arch, config)
            hp.update({k: cand[k] for k in cand})
            model, hist = fit_model(arch, hp, ctx, config, seed, int(hpo_cfg["epochs"]),
                                    int(hpo_cfg["early_stopping_patience"]))
            val_metrics, _ = rolling_eval(model, ctx, ctx["val_origins"])
            trials.append({"config_id": i, "hparams": hp, "history": {k: v for k, v in hist.items() if k not in ("train_loss", "val_loss", "val_poisson", "val_pinball")},
                           "val": val_metrics})
            print(f"[hpo:{arch}] cfg {i}/6 h={hp['hidden_size']} d={hp['dropout']} lr={hp['learning_rate']:g} "
                  f"wd={hp['weight_decay']:g} | params={hist['n_parameters']:,} ép={hist['best_epoch']}/{hist['epochs_run']} "
                  f"{hist['train_seconds']:.1f}s | VAL {_fmt(val_metrics)}")
        best_idx = int(np.argmin([t["val"]["overall"][sel_metric] for t in trials]))
        results[arch] = {"selection_metric": sel_metric, "best_config_id": trials[best_idx]["config_id"],
                         "best_hparams": trials[best_idx]["hparams"], "trials": trials,
                         "total_seconds": round(sum(t["history"]["train_seconds"] for t in trials), 2)}
        print(f"[hpo:{arch}] MEJOR config {trials[best_idx]['config_id']}: {trials[best_idx]['hparams']}")

    out_json = resolve_path(config["paths"]["hpo_results_file"])
    payload = {"protocol": {"seed": seed, "epochs": hpo_cfg["epochs"], "patience": hpo_cfg["early_stopping_patience"],
                            "selection": f"min val {sel_metric} (rolling origin, {len(ctx['val_origins'])} orígenes, H={ctx['horizon']})",
                            "total_wall_seconds": round(time.perf_counter() - t_all, 2)},
               "results": results}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_store = resolve_path(config["paths"]["best_hparams_file"])
    previous = json.loads(best_store.read_text(encoding="utf-8")) if best_store.exists() else {}
    for arch in archs:
        previous[arch] = {"hparams": results[arch]["best_hparams"], "config_id": results[arch]["best_config_id"],
                          "val_mae": results[arch]["trials"][results[arch]["best_config_id"] - 1]["val"]["overall"]["mae"]}
    best_store.parent.mkdir(parents=True, exist_ok=True)
    best_store.write_text(json.dumps(previous, indent=2), encoding="utf-8")

    baselines_val: Dict[str, float] = {}
    bl_path = resolve_path(config["paths"]["baselines_metrics_file"])
    if bl_path.exists():
        bl = json.loads(bl_path.read_text(encoding="utf-8"))
        baselines_val = {k: v["val"]["overall"]["mae"] for k, v in bl["models"].items()}
    _plot_hpo(results, baselines_val, resolve_path(config["paths"]["fig_hpo"]), int(config["eda"]["dpi"]))
    print(f"[OK] HPO: {out_json} | {best_store} | {config['paths']['fig_hpo']}")
    return payload


def _plot_learning_curves(config: Dict[str, Any]) -> None:
    hist_dir = resolve_path(config["paths"]["history_dir"])
    available = [a for a in ARCHS_ALL if (hist_dir / f"m1_{a}_history.json").exists()]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(7.2 * len(available), 4.6), squeeze=False)
    for ax, arch in zip(axes[0], available):
        h = json.loads((hist_dir / f"m1_{arch}_history.json").read_text(encoding="utf-8"))
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color="#1f4e78", label="Train (con dropout)")
        ax.plot(ep, h["val_loss"], color="#d95f02", label="Val")
        ax.axvline(h["best_epoch"], color="#c0392b", linestyle="--", label=f"Early stopping: época {h['best_epoch']}")
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida (Poisson NLL + λ·Pinball P90)")
        ax.set_title(f"{ARCH_LABEL[arch]} (semilla {h.get('seed')})", fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("Curvas de aprendizaje M1: entrenamiento vs. validación", fontweight="bold")
    fig.tight_layout()
    out = resolve_path(config["paths"]["fig_learning_curves"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(config["eda"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _run_single(config, archs, ctx) -> Dict[str, Any]:
    mode_cfg = config["training"]["modes"]["single"]
    seed = int(config["reproducibility"]["seed"])
    model_dir = resolve_path(config["paths"]["model_dir"])
    hist_dir = resolve_path(config["paths"]["history_dir"])
    results: Dict[str, Any] = {}
    for arch in archs:
        hp = resolve_hparams(arch, config)
        model, hist = fit_model(arch, hp, ctx, config, seed, int(mode_cfg["epochs"]),
                                int(config["training"]["early_stopping_patience"]), mode_cfg.get("max_samples"))
        val_metrics, _ = rolling_eval(model, ctx, ctx["val_origins"])
        save_checkpoint(model_dir / f"{arch}_best.pt", model, arch, hp, ctx,
                        {"mode": "single", "seed": seed, "best_epoch": hist["best_epoch"]})
        hist_dir.mkdir(parents=True, exist_ok=True)
        (hist_dir / f"m1_{arch}_history.json").write_text(json.dumps({**hist, "seed": seed, "hparams": hp}), encoding="utf-8")
        results[arch] = {"hparams": hp, "history": {k: v for k, v in hist.items() if not isinstance(v, list)}, "val": val_metrics}
        print(f"[single:{arch}] hp={hp} params={hist['n_parameters']:,} mejor época={hist['best_epoch']}/{hist['epochs_run']} "
              f"{hist['train_seconds']:.1f}s | VAL {_fmt(val_metrics)}")
    ctx_arrays = ctx["arrays"]
    (model_dir / "scalers.json").write_text(
        json.dumps({"route_ids": ctx_arrays.route_ids, "route_train_mean": [float(v) for v in ctx_arrays.route_train_mean],
                    "scale_floor_frac": ctx["floor_frac"], "calendar_cols": list(FUTURE_CALENDAR_COLS),
                    "lookback": ctx["lookback"], "horizon": ctx["horizon"]}, indent=2), encoding="utf-8")
    save_single_run_metrics(results, resolve_path(config["paths"]["metrics_file"]), {"mode": "single", "seed": seed})
    _plot_learning_curves(config)
    print(f"[OK] Pesos en {model_dir} | curvas: {config['paths']['fig_learning_curves']}")
    return results


def _run_eval30(config, archs, ctx) -> Dict[str, Any]:
    mode_cfg = config["training"]["modes"]["eval30"]
    seeds = get_seed_list(int(config["reproducibility"]["seed_list_start"]), int(mode_cfg["n_seeds"]))
    patience = int(config["training"]["early_stopping_patience"])
    hps = {a: resolve_hparams(a, config) for a in archs}
    records: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for k, seed in enumerate(seeds, start=1):
        for arch in archs:
            model, hist = fit_model(arch, hps[arch], ctx, config, seed, int(mode_cfg["epochs"]), patience)
            m, _ = rolling_eval(model, ctx, ctx["test_origins"])
            hb = m["by_horizon_block"]
            records.append({
                "arch": arch, "seed": seed, **{k2: m["overall"][k2] for k2 in ("mae", "rmse", "mase", "pinball_p90", "coverage_p90")},
                "mae_h1_7": hb["days_1_7"]["mae"], "mae_h8_14": hb["days_8_14"]["mae"], "mae_h15_30": hb["days_15_30"]["mae"],
                "best_epoch": hist["best_epoch"], "train_seconds": hist["train_seconds"], "val_loss": round(hist["best_val_loss"], 6),
            })
        print(f"[eval30] semilla {seed} ({k}/{len(seeds)}) | " + " | ".join(
            f"{r['arch']} MAE={r['mae']:.1f}" for r in records[-len(archs):]) + f" | acumulado {time.perf_counter() - t0:.0f}s")
    payload = save_eval30_results_grouped(
        records, "arch", resolve_path(config["paths"]["eval30_csv"]), resolve_path(config["paths"]["eval30_json"]),
        {"protocol": "rolling_origin_test", "seeds": [seeds[0], seeds[-1]], "hparams": hps,
         "test_origins": len(ctx["test_origins"]), "horizon": ctx["horizon"]},
    )
    for arch in archs:
        s = payload["summary"][arch]
        print(f"[eval30:{arch}] MAE {s['mae']['mean']:.1f}±{s['mae']['std']:.1f} (mejor {s['mae']['min']:.1f}, peor {s['mae']['max']:.1f}) | "
              f"MASE {s['mase']['mean']:.3f}±{s['mase']['std']:.3f} | PinballP90 {s['pinball_p90']['mean']:.1f} | CobP90 {100 * s['coverage_p90']['mean']:.1f}%")
    print(f"[OK] {config['paths']['eval30_csv']} y {config['paths']['eval30_json']}")
    return payload


def run_train(config: Dict[str, Any], mode: str = "demo", arch: str = "all") -> Dict[str, Any]:
    """Punto de entrada: ejecuta el modo solicitado para las arquitecturas indicadas."""
    if mode not in ("demo", "hpo", "single", "eval30"):
        raise ValueError(f"Modo inválido '{mode}'.")
    _set_threads(config)
    archs = _resolve_archs(arch)
    ctx = prepare_context(config, with_train_windows=True)
    print(f"[m1] modo={mode} arch={archs} | muestras train={ctx['windows']['train']['route'].shape[0]:,} "
          f"val={ctx['windows']['val']['route'].shape[0]:,} | hilos torch={torch.get_num_threads()}")
    runner = {"demo": _run_demo, "hpo": _run_hpo, "single": _run_single, "eval30": _run_eval30}[mode]
    return runner(config, archs, ctx)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento de redes del Módulo 1 (Demanda y Flota).")
    parser.add_argument("--mode", choices=["demo", "hpo", "single", "eval30"], default="demo")
    parser.add_argument("--arch", choices=["mlp", "gru", "all"], default="all")
    add_config_override_args(parser)
    args = parser.parse_args(argv)
    config = load_module_config("m1", overrides=args.override, seed_override=args.seed)
    set_seed(config["reproducibility"]["seed"], config["reproducibility"].get("deterministic_torch", True))
    run_train(config=config, mode=args.mode, arch=args.arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
