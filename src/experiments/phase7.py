"""Phase 7: monthly refits, local/global comparison, and seed sensitivity.

Selections use validation only. Each command is explicit because these runs
are expensive and are intended for a remote CUDA environment.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.panel import Panel, load_panel_matrix
from src.evaluation.backtest import backtest, monthly_refit_backtest
from src.evaluation.validation import score_validation
from src.evaluation.metrics import evaluate_predictions
from src.models.recurrent import (
    LSTMMultiOutputAdapter, LSTMRecursiveAdapter, LSTMSeq2SeqAdapter,
)
from src.models.lightgbm_adapter import LightGBMDirectAdapter
from src.models.torch_forecasters import DLinearAdapter, MLPAdapter, NLinearAdapter
from src.models.neuralforecast_adapters import NeuralForecastAdapter
from src.models.statsforecast_adapters import (
    auto_ets_adapter, naive_adapter, seasonal_naive_24_adapter,
    seasonal_naive_168_adapter,
)

RECURRENT = {
    "lstm_recursive": LSTMRecursiveAdapter,
    "lstm_multioutput": LSTMMultiOutputAdapter,
    "lstm_seq2seq": LSTMSeq2SeqAdapter,
}
BASELINES = {
    "naive": naive_adapter,
    "seasonal_naive_24": seasonal_naive_24_adapter,
    "seasonal_naive_168": seasonal_naive_168_adapter,
    "auto_ets": auto_ets_adapter,
}


def _selected_recurrent(output_dir: Path) -> tuple[str, type, dict]:
    selected = json.loads(
        (output_dir / "metrics" / "phase5_selected.json").read_text(
            encoding="utf-8")
    )
    name = min(selected, key=lambda item: selected[item]["validation_mase"])
    return name, RECURRENT[name], selected[name]["config"]


def _baseline_selection(panel: Panel, output_dir: Path) -> tuple[str, object, dict]:
    rows = []
    for name, factory in BASELINES.items():
        config = {
            "lookback": 168, "horizon": 24, "history_months": 30,
            "seed": 42, "mase_seasonality": 24,
            "calendar_encoding": "cyclic",
            "config_id": f"{name}-validation",
        }
        result = score_validation(factory(), panel, config)
        rows.append({"name": name, "validation_mase": result["metrics"]["mase"]})
    summary = pd.DataFrame(rows)
    destination = output_dir / "metrics" / "phase7_baseline_selection.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)
    winner = summary.sort_values("validation_mase").iloc[0]["name"]
    return winner, BASELINES[winner], {
        "lookback": 168, "horizon": 24, "history_months": 30,
        "seed": 42, "mase_seasonality": 24,
        "config_id": f"{winner}-monthly",
    }


def _deep_selection(panel: Panel, output_dir: Path,
                    own_only: bool = False) -> tuple[str, object, dict]:
    recurrent_name, recurrent_factory, recurrent_config = _selected_recurrent(
        output_dir)
    torch_base = {
        "lookback": 168, "horizon": 24, "history_months": 30,
        "seed": 42, "mase_seasonality": 24,
        "normalization": "per_series", "device": "cuda",
        "max_epochs": 20, "patience": 5,
        "training_stride": 1, "validation_stride": 24,
        "max_train_windows_per_client": 256,
        "artifact_dir": str(output_dir / "models" / "phase7_selection"),
    }
    neural_base = {
        "lookback": 168, "horizon": 24, "history_months": 30,
        "seed": 42, "mase_seasonality": 24,
        "normalization": "per_series", "max_steps": 1000,
        "early_stop_patience_steps": 5, "val_check_steps": 100,
        "artifact_dir": str(output_dir / "models" / "phase7_selection"),
    }
    candidates = {
        recurrent_name: (recurrent_factory, recurrent_config),
        "mlp": (MLPAdapter, torch_base),
        "dlinear": (DLinearAdapter, torch_base),
        "nlinear": (NLinearAdapter, torch_base),
        **({} if own_only else {
            name.lower(): (lambda model_name=name: NeuralForecastAdapter(model_name),
                           neural_base)
            for name in ("NHITS", "TCN", "PatchTST", "TFT", "DeepAR")
        }),
    }
    selection_path = (output_dir / "metrics" /
                      ("phase7_own_selection.json" if own_only
                       else "phase7_deep_selection.json"))
    if selection_path.exists():
        saved = json.loads(selection_path.read_text(encoding="utf-8"))
        name = saved["model"]
        if name not in candidates:
            raise ValueError(f"Saved model is not a current candidate: {name}")
        return name, candidates[name][0], saved["config"]
    candidate_path = selection_path.with_name(
        selection_path.stem.replace("_selection", "_candidates") + ".json"
    )
    completed = (json.loads(candidate_path.read_text(encoding="utf-8"))
                 if candidate_path.exists() else {})
    rows = []
    best = None
    for name, (factory, base) in candidates.items():
        config = {**base, "config_id": f"{name}-deep-selection"}
        cached = completed.get(name)
        if cached is not None and cached["config"] == config:
            score = cached["validation_mase"]
            best_epoch = cached["best_epoch"]
        else:
            validation = score_validation(factory(), panel, config)
            score = validation["metrics"]["mase"]
            best_epoch = validation["fit_metadata"].get("best_epoch")
            if score is None or not pd.notna(score):
                raise ValueError(f"Undefined validation MASE for {name}")
            score = float(score)
            best_epoch = int(best_epoch) if best_epoch is not None else None
            completed[name] = {
                "config": config, "validation_mase": score,
                "best_epoch": best_epoch,
            }
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(
                json.dumps(completed, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        rows.append({"model": name, "validation_mase": score})
        if score is not None and (best is None or score < best[0]):
            if best_epoch is not None:
                config["best_epoch"] = int(best_epoch)
            best = (score, name, factory, config)
    destination = (output_dir / "metrics" /
                   ("phase7_own_selection.csv" if own_only
                    else "phase7_deep_selection.csv"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    if best is None:
        raise ValueError("No deep learning model had defined validation MASE")
    _, name, factory, config = best
    selection_path.write_text(
        json.dumps({"model": name, "config": config}, indent=2),
        encoding="utf-8",
    )
    return name, factory, config


def run_monthly(panel: Panel, output_dir: Path) -> pd.DataFrame:
    baseline_name, baseline_factory, baseline_config = _baseline_selection(
        panel, output_dir)
    recurrent_name, recurrent_factory, recurrent_config = _deep_selection(
        panel, output_dir)
    # Use the complete 2012-to-origin history for both fixed and monthly runs.
    # history_months=100 makes the fixed backtest include the full train split;
    # monthly_refit_backtest itself grows from the fixed train start.
    choices = [
        (baseline_name, baseline_factory,
         {**baseline_config, "history_months": 100}),
        (recurrent_name, recurrent_factory,
         {**recurrent_config, "history_months": 100,
          "config_id": f"{recurrent_name}-monthly"}),
    ]
    rows = []
    by_month_rows = []
    for name, factory, config in choices:
        fixed = backtest(factory(), panel,
                         config={**config,
                                 "refit_train_validation": (
                                     name == recurrent_name and
                                     config.get("best_epoch") is not None
                                 ),
                                 "config_id": f"{name}-fixed-reference"},
                         output_dir=output_dir)
        monthly = monthly_refit_backtest(
            factory, panel, config=config, output_dir=output_dir
        )
        for month in (10, 11, 12):
            fixed_month = fixed.predictions.loc[
                pd.to_datetime(fixed.predictions["ds"]).dt.month.eq(month)
            ]
            refit_month = monthly.predictions.loc[
                pd.to_datetime(monthly.predictions["ds"]).dt.month.eq(month)
            ]
            fixed_metrics = evaluate_predictions(fixed_month).overall
            refit_metrics = evaluate_predictions(refit_month).overall
            by_month_rows.append({
                "model": name, "month": f"2014-{month:02d}",
                "fixed_mae": fixed_metrics["mae"],
                "fixed_mase": fixed_metrics["mase"],
                "monthly_mae": refit_metrics["mae"],
                "monthly_mase": refit_metrics["mase"],
            })
        rows.append({
            "model": name, "fixed_mase": fixed.metrics["mase"],
            "monthly_mase": monthly.metrics["mase"],
            "fixed_mae": fixed.metrics["mae"],
            "monthly_mae": monthly.metrics["mae"],
            "monthly_fit_count": monthly.experiment["fit_count"],
        })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "metrics" / "phase7_monthly.csv", index=False)
    pd.DataFrame(by_month_rows).to_csv(
        output_dir / "metrics" / "phase7_monthly_by_month.csv", index=False
    )
    return result


def run_local(panel: Panel, output_dir: Path) -> pd.DataFrame:
    name, factory, config = _deep_selection(panel, output_dir, own_only=True)
    ids = tuple(panel.client_ids[:10])
    if len(ids) != 10:
        raise ValueError("Local/global experiment requires ten fixed clients")
    config = {**config, "refit_train_validation": config.get("best_epoch") is not None}
    global_result = backtest(
        factory(), panel,
        config={**config, "config_id": f"{name}-global-local-comparison"},
        output_dir=output_dir,
    )
    global_metrics = global_result.by_client.set_index("unique_id")
    rows = []
    for position, unique_id in enumerate(ids):
        local_panel = Panel(
            (unique_id,), panel.timestamps, panel.values[position:position + 1]
        )
        result = backtest(
            factory(), local_panel,
            config={**config, "config_id": f"{name}-local-{unique_id}"},
            output_dir=output_dir,
        )
        rows.append({
            "unique_id": unique_id, "model": name,
            "global_mase": float(global_metrics.loc[unique_id, "mase"]),
            "local_mase": result.metrics["mase"],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "metrics" / "phase7_local_global.csv",
                      index=False)
    return comparison


def run_seeds(panel: Panel, output_dir: Path) -> pd.DataFrame:
    name, factory, config = _deep_selection(panel, output_dir)
    rows = []
    for seed in (42, 43, 44):
        result = backtest(
            factory(), panel,
            config={**config, "seed": seed,
                    "refit_train_validation": config.get("best_epoch") is not None,
                    "config_id": f"{name}-seed-{seed}"},
            output_dir=output_dir,
        )
        rows.append({"model": name, "seed": seed, "mase": result.metrics["mase"]})
    summary = pd.DataFrame(rows)
    summary["mean_mase"] = float(summary["mase"].mean())
    summary["sd_mase"] = float(summary["mase"].std(ddof=1))
    summary.to_csv(output_dir / "metrics" / "phase7_seeds.csv", index=False)
    return summary


HISTORY_MODELS = ("seasonal_naive_24", "lightgbm", "dlinear",
                  "recurrent", "patchtst")


def run_history(panel: Panel, output_dir: Path,
                selected_models: tuple[str, ...] = HISTORY_MODELS) -> pd.DataFrame:
    """Predeclared 3/12/30-month sensitivity; no test-driven tuning."""
    if not set(selected_models).issubset(HISTORY_MODELS):
        raise ValueError(f"Unknown history model: {selected_models}")
    recurrent_name, recurrent_factory, recurrent_config = _selected_recurrent(
        output_dir)
    models = {
        "seasonal_naive_24": (seasonal_naive_24_adapter, {}),
        "lightgbm": (LightGBMDirectAdapter, {
            "lookback": 168, "training_stride": 24,
            "validation_stride": 24, "n_estimators": 500,
            "max_train_windows_per_client": 256,
        }),
        "dlinear": (DLinearAdapter, {
            "lookback": 168, "normalization": "per_series",
            "device": "cuda", "max_epochs": 20, "patience": 5,
            "training_stride": 1, "validation_stride": 24,
            "max_train_windows_per_client": 256,
        }),
        "recurrent": (recurrent_factory, recurrent_config),
        "patchtst": (lambda: NeuralForecastAdapter("PatchTST"), {
            "lookback": 168, "normalization": "per_series",
            "max_steps": 1000, "batch_size": 32,
            "early_stop_patience_steps": 5, "val_check_steps": 100,
        }),
    }
    destination = output_dir / "metrics" / "phase7_history.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = (pd.read_csv(destination).to_dict("records")
            if destination.exists() else [])
    for key in selected_models:
        factory, options = models[key]
        name = recurrent_name if key == "recurrent" else key
        for months in (3, 12, 30):
            config = {
                "lookback": 168, "horizon": 24,
                "seed": 42, "mase_seasonality": 24,
                "calendar_encoding": "cyclic",
                "artifact_dir": str(output_dir / "models" / "phase7_history"),
                **options,
                "history_months": months,
                "config_id": f"{name}-history-{months}",
            }
            if any(previous["config_id"] == config["config_id"]
                   for previous in rows):
                continue
            validation = score_validation(factory(), panel, config)
            best_epoch = validation["fit_metadata"].get("best_epoch")
            if best_epoch is not None and name != "lightgbm":
                config = {**config, "best_epoch": int(best_epoch),
                          "refit_train_validation": True}
            result = backtest(factory(), panel, config=config,
                              output_dir=output_dir)
            row = {
                "model": name, "history_months": months,
                "validation_mase": validation["metrics"]["mase"],
                "test_mase": result.metrics["mase"],
                "config_id": config["config_id"],
            }
            rows = [previous for previous in rows
                    if previous["config_id"] != row["config_id"]]
            rows.append(row)
            summary = pd.DataFrame(rows).sort_values(["model", "history_months"])
            summary.to_csv(destination, index=False)
    return pd.DataFrame(rows).sort_values(["model", "history_months"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("monthly", "local", "seeds", "history"))
    parser.add_argument("--panel", type=Path,
                        default=Path("data/processed/hourly.parquet"))
    parser.add_argument("--client-ids", type=Path,
                        default=Path("data/processed/client_ids.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--history-models", nargs="+", choices=HISTORY_MODELS,
                        default=list(HISTORY_MODELS),
                        help="Run selected history families; results merge into one CSV")
    args = parser.parse_args()
    panel = load_panel_matrix(args.panel, args.client_ids)
    action = {"monthly": run_monthly, "local": run_local,
              "seeds": run_seeds, "history": run_history}[args.stage]
    if args.stage == "history":
        result = run_history(panel, args.output_dir,
                             tuple(args.history_models))
    else:
        result = action(panel, args.output_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
