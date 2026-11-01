"""Small integration check for B models: 8 clients, 1 origin, minimal fitting.

Smoke outputs are isolated and must never be used as final test metrics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.panel import Panel, load_panel_matrix
from src.evaluation.backtest import backtest
from src.models.chronos import ChronosBoltAdapter
from src.models.neuralforecast_adapters import NeuralForecastAdapter
from src.models.recurrent import (
    LSTMRecursiveAdapter, LSTMMultiOutputAdapter, LSTMSeq2SeqAdapter,
)

FACTORIES = {
    "lstm_recursive": LSTMRecursiveAdapter,
    "lstm_multioutput": LSTMMultiOutputAdapter,
    "lstm_seq2seq": LSTMSeq2SeqAdapter,
    **{name.lower(): (lambda selected=name: NeuralForecastAdapter(selected))
       for name in ("NHITS", "TCN", "PatchTST", "TFT", "DeepAR")},
    "chronos": ChronosBoltAdapter,
}


def run(panel_path: Path, ids_path: Path, output_dir: Path,
        selected: list[str]) -> pd.DataFrame:
    panel = load_panel_matrix(panel_path, ids_path)
    count = min(8, len(panel.client_ids))
    panel = Panel(panel.client_ids[:count], panel.timestamps,
                  panel.values[:count])
    rows = []
    for key in selected:
        model = FACTORIES[key]()
        config = {
            "lookback": 168, "horizon": 24, "history_months": 3,
            "seed": 42, "normalization": "per_series",
            "calendar_encoding": "cyclic", "mase_seasonality": 24,
            "max_epochs": 1, "patience": 1, "max_steps": 3,
            "val_check_steps": 1, "early_stop_patience_steps": 2,
            "max_train_windows_per_client": 8, "batch_size": 8,
            "valid_batch_size": 2, "windows_batch_size": 8,
            "inference_windows_batch_size": 8,
            "device": "cuda", "config_id": f"{key}-smoke",
            "artifact_dir": str(output_dir / "models"),
        }
        if key == "chronos":
            config["normalization"] = "raw"
        result = backtest(
            model, panel, origins=[pd.Timestamp("2014-10-01")],
            config=config, output_dir=output_dir,
        )
        if len(result.predictions) != count * 24:
            raise ValueError(f"Incomplete smoke forecast for {key}")
        if not np.isfinite(result.predictions["y_hat"]).all():
            raise ValueError(f"Non-finite smoke forecast for {key}")
        if key in {"deepar", "chronos"}:
            probabilistic = model.probabilistic_predictions(result.predictions)
            if (probabilistic["p10"] > probabilistic["p90"]).any():
                raise ValueError(f"Crossing quantiles for {key}")
        rows.append({"model": key, "clients": count, "origins": 1,
                     "predictions": len(result.predictions)})
    summary = pd.DataFrame(rows)
    destination = output_dir / "smoke_b_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path,
                        default=Path("data/processed/hourly.parquet"))
    parser.add_argument("--client-ids", type=Path,
                        default=Path("data/processed/client_ids.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/smoke_b"))
    parser.add_argument("--models", nargs="+", choices=tuple(FACTORIES),
                        default=[name for name in FACTORIES if name != "chronos"])
    args = parser.parse_args()
    print(run(args.panel, args.client_ids, args.output_dir,
              args.models).to_string(index=False))


if __name__ == "__main__":
    main()
