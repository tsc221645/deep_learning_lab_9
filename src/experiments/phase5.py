"""Select recurrent configurations on validation, then run the common test backtest.

This module is deliberately a CLI; importing it never starts training.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.panel import Panel, load_panel_matrix
from src.data.splits import DEFAULT_SPLITS
from src.evaluation.backtest import backtest
from src.models.recurrent import (
    LSTMMultiOutputAdapter,
    LSTMRecursiveAdapter,
    LSTMSeq2SeqAdapter,
)

FACTORIES = {
    "lstm_recursive": LSTMRecursiveAdapter,
    "lstm_multioutput": LSTMMultiOutputAdapter,
    "lstm_seq2seq": LSTMSeq2SeqAdapter,
}


def _slice(panel: Panel, start: pd.Timestamp, end: pd.Timestamp) -> Panel:
    mask = (panel.timestamps >= start) & (panel.timestamps < end)
    return Panel(panel.client_ids, panel.timestamps[mask], panel.values[:, mask])


def candidate_configs(model: str, artifact_dir: Path) -> list[dict]:
    """One controlled L experiment, with the same validation block for all L."""
    base = {
        "horizon": 24, "history_months": 30, "seed": 42,
        "normalization": "per_series", "calendar_encoding": "cyclic",
        "device": "cuda", "max_epochs": 20, "patience": 5,
        "batch_size": 256, "learning_rate": 1e-3,
        "hidden_size": 64, "num_layers": 2, "dropout": 0.1,
        "training_stride": 1, "validation_stride": 24,
        "max_train_windows_per_client": 256,
        "artifact_dir": str(artifact_dir),
        "teacher_forcing": 0.5 if model == "lstm_seq2seq" else 0.0,
    }
    return [
        {**base, "lookback": lookback, "config_id": f"{model}-L{lookback}-h30"}
        for lookback in (48, 168, 336)
    ]


def select_on_validation(panel: Panel, output_dir: Path) -> dict[str, dict]:
    train = _slice(panel, *DEFAULT_SPLITS["train"])
    validation = _slice(panel, *DEFAULT_SPLITS["validation"])
    rows = []
    selected = {}
    for model_name, factory in FACTORIES.items():
        for config in candidate_configs(model_name, output_dir / "models" / "phase5"):
            metadata = factory().fit(train, validation, config)
            log = pd.read_csv(metadata["training_log"])
            best = log.loc[log["epoch"].eq(metadata["best_epoch"])].iloc[0]
            score = float(best["validation_mase"])
            if not pd.notna(score):
                raise ValueError(f"Undefined validation MASE: {config['config_id']}")
            rows.append({
                "model": model_name, "config_id": config["config_id"],
                "lookback": config["lookback"], "best_epoch": metadata["best_epoch"],
                "validation_mae_kwh": float(best["validation_mae_kwh"]),
                "validation_mase": score,
                "training_log": metadata["training_log"],
            })
            current = selected.get(model_name)
            if current is None or score < current["validation_mase"]:
                selected[model_name] = {
                    "validation_mase": score,
                    "config": {**config, "best_epoch": int(metadata["best_epoch"])},
                }
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "phase5_validation.csv", index=False)
    (metrics_dir / "phase5_selected.json").write_text(
        json.dumps(selected, indent=2, allow_nan=False), encoding="utf-8"
    )
    return selected


def evaluate_selected(panel: Panel, output_dir: Path) -> pd.DataFrame:
    selected_path = output_dir / "metrics" / "phase5_selected.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if set(selected) != set(FACTORIES):
        raise ValueError("Selection file must contain all three LSTM strategies")
    rows = []
    for model_name, factory in FACTORIES.items():
        config = selected[model_name]["config"]
        # The epoch count was fixed on validation before any test prediction.
        config = {**config, "refit_train_validation": True}
        result = backtest(factory(), panel, config=config, output_dir=output_dir)
        rows.append({
            "model": model_name, "config_id": config["config_id"],
            "validation_mase": selected[model_name]["validation_mase"],
            **result.metrics,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "metrics" / "phase5_test.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test"))
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hourly.parquet"))
    parser.add_argument("--client-ids", type=Path, default=Path("data/processed/client_ids.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    panel = load_panel_matrix(args.panel, args.client_ids)
    if args.stage == "select":
        result = select_on_validation(panel, args.output_dir)
        print(json.dumps(result, indent=2))
    else:
        print(evaluate_selected(panel, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
