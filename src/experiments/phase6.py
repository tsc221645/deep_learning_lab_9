"""Run phase 6 Nixtla models with the shared 92-origin evaluator.

Configurations are declared before test evaluation; test metrics are never
used to select or change hyperparameters.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.panel import load_panel_matrix
from src.evaluation.backtest import backtest
from src.evaluation.validation import score_validation
from src.models.chronos import probabilistic_metrics
from src.models.neuralforecast_adapters import MODEL_NAMES, NeuralForecastAdapter


def configuration(name: str, output_dir: Path) -> dict:
    return {
        "lookback": 168, "horizon": 24, "history_months": 30,
        "normalization": "per_series", "calendar_encoding": "cyclic",
        "seed": 42, "max_steps": 1000, "batch_size": 32,
        "learning_rate": 1e-3, "early_stop_patience_steps": 5,
        "val_check_steps": 100, "mase_seasonality": 24,
        "artifact_dir": str(output_dir / "models" / "phase6"),
        "config_id": f"{name.lower()}-L168-h30-v1",
    }


def run(panel_path: Path, ids_path: Path, output_dir: Path,
        names: list[str]) -> pd.DataFrame:
    panel = load_panel_matrix(panel_path, ids_path)
    rows = []
    for name in names:
        model = NeuralForecastAdapter(name)
        config = configuration(name, output_dir)
        validation = score_validation(
            NeuralForecastAdapter(name), panel,
            {**config, "config_id": f"{config['config_id']}-validation"},
        )
        result = backtest(model, panel, config=config, output_dir=output_dir)
        row = {
            "model": name, "config_id": config["config_id"],
            "validation_mase": validation["metrics"]["mase"],
            "L": config["lookback"], "H": config["horizon"],
            "train_time_s": result.experiment["train_time_s"],
            "inference_time_s": result.experiment["inference_time_s"],
            "params_total": result.experiment["params_total"],
            "params_trainable": result.experiment["params_trainable"],
            **result.metrics,
        }
        if name == "DeepAR":
            quantiles = model.probabilistic_predictions(result.predictions)
            destination = (
                output_dir / "predictions" / "deepar" /
                config["config_id"] / "quantiles.parquet"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            quantiles.to_parquet(destination, index=False)
            row.update(probabilistic_metrics(quantiles))
        rows.append(row)
    summary = pd.DataFrame(rows)
    destination = output_dir / "metrics" / "phase6_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)
    (destination.with_suffix(".json")).write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hourly.parquet"))
    parser.add_argument("--client-ids", type=Path, default=Path("data/processed/client_ids.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=list(MODEL_NAMES))
    args = parser.parse_args()
    print(run(args.panel, args.client_ids, args.output_dir,
              args.models).to_string(index=False))


if __name__ == "__main__":
    main()
