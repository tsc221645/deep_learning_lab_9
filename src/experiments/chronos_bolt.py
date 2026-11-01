"""Run Chronos-Bolt zero-shot with the same 92-origin point backtest."""

import argparse
import json
from pathlib import Path

from src.data.panel import load_panel_matrix
from src.evaluation.backtest import backtest
from src.models.chronos import ChronosBoltAdapter, probabilistic_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hourly.parquet"))
    parser.add_argument("--client-ids", type=Path, default=Path("data/processed/client_ids.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-id", default="amazon/chronos-bolt-small")
    parser.add_argument("--context-length", type=int, default=168)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    panel = load_panel_matrix(args.panel, args.client_ids)
    model = ChronosBoltAdapter(args.model_id, args.context_length, args.batch_size)
    config = {
        "lookback": args.context_length, "horizon": 24, "history_months": 30,
        "config_id": f"chronos-bolt-L{args.context_length}",
        "normalization": "raw", "seed": 42,
    }
    result = backtest(model, panel, config=config, output_dir=args.output_dir)
    probabilistic = model.probabilistic_predictions(result.predictions)
    metrics = probabilistic_metrics(probabilistic)
    destination = args.output_dir / "predictions" / "chronos_bolt_quantiles.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    probabilistic.to_parquet(destination, index=False)
    metrics_path = args.output_dir / "metrics" / "chronos_bolt_probabilistic.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({**result.metrics, **metrics}, indent=2))


if __name__ == "__main__":
    main()
