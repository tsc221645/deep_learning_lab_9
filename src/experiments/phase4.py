"""Run the phase 4 baselines and model-sensitivity experiments."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.panel import Panel, load_panel_matrix
from src.evaluation.backtest import backtest
from src.models.lightgbm_adapter import LightGBMDirectAdapter
from src.models.statsforecast_adapters import (
    auto_ets_adapter,
    naive_adapter,
    seasonal_naive_24_adapter,
    seasonal_naive_168_adapter,
)
from src.models.torch_forecasters import (
    DLinearAdapter,
    MLPAdapter,
    NLinearAdapter,
)

DEFAULT_MODELS = (
    "naive",
    "seasonal_naive_24",
    "seasonal_naive_168",
    "auto_ets",
    "lightgbm",
    "mlp",
    "dlinear",
    "nlinear",
)

PRIMARY_CONFIG_IDS = {
    "naive-h30", "seasonal_naive_24-h30", "seasonal_naive_168-h30",
    "auto_ets-h30", "lightgbm-h30", "mlp-L168-per_series-h30",
    "dlinear-L168-per_series-h30", "nlinear-L168-per_series-h30",
}


def build_experiments(
    artifact_dir: Path,
    *,
    smoke: bool = False,
) -> list[tuple[str, Any, dict[str, Any]]]:
    """Return unique, named model configurations in reproducible order."""
    experiments: list[tuple[str, Any, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()

    def add(
        name: str,
        factory: Any,
        config: dict[str, Any],
    ) -> None:
        if smoke:
            name = f"{name}-smoke"
        config = {
            "lookback": 168,
            "horizon": 24,
            "history_months": 30,
            "seed": 42,
            "mase_seasonality": 24,
            "calendar_encoding": "cyclic",
            "artifact_dir": str(artifact_dir),
            **config,
        }
        config["config_id"] = name
        signature = (
            name.split("-")[0],
            json.dumps(
                {key: value for key, value in config.items() if key != "config_id"},
                sort_keys=True,
            ),
        )
        if name in seen_ids:
            raise ValueError(f"Duplicate phase 4 config_id would overwrite artifacts: {name}")
        if signature not in seen:
            experiments.append((name, factory, config))
            seen.add(signature)
            seen_ids.add(name)

    factories = {
        "naive": naive_adapter,
        "seasonal_naive_24": seasonal_naive_24_adapter,
        "seasonal_naive_168": seasonal_naive_168_adapter,
        "auto_ets": auto_ets_adapter,
        "lightgbm": LightGBMDirectAdapter,
        "mlp": MLPAdapter,
        "dlinear": DLinearAdapter,
        "nlinear": NLinearAdapter,
    }

    for model_name, factory in factories.items():
        if model_name in {"mlp", "dlinear", "nlinear"}:
            neural_config = {
                "normalization": "per_series",
                "lookback": 168,
                "max_epochs": 1 if smoke else 20,
                "patience": 1 if smoke else 5,
                "max_train_windows_per_client": 8 if smoke else 256,
                "training_stride": 1,
                "validation_stride": 24,
                "device": "cuda",
            }
            if model_name == "mlp":
                neural_config["hidden_size"] = 32 if smoke else 128
            add(
                f"{model_name}-L168-per_series-h30",
                factory,
                neural_config,
            )
        else:
            base_config: dict[str, Any] = {"history_months": 30}
            if model_name == "lightgbm":
                base_config.update(
                    {
                        "training_stride": 24,
                        "max_train_windows_per_client": 8 if smoke else 256,
                        "n_estimators": 3 if smoke else 500,
                        "patience": 2 if smoke else 30,
                    }
                )
            add(f"{model_name}-h30", factory, base_config)

    if not smoke:
        for seasonal_name, factory in (
            ("seasonal_naive_24", seasonal_naive_24_adapter),
            ("seasonal_naive_168", seasonal_naive_168_adapter),
        ):
            for history_months in (3, 12):
                add(
                    f"{seasonal_name}-h{history_months}",
                    factory,
                    {"history_months": history_months},
                )
        for history_months in (3, 12):
            add(
                f"lightgbm-h{history_months}",
                LightGBMDirectAdapter,
                {
                    "history_months": history_months,
                    "training_stride": 24,
                    "max_train_windows_per_client": 256,
                    "n_estimators": 500,
                    "patience": 30,
                },
            )
            add(
                f"dlinear-L168-per_series-h{history_months}",
                DLinearAdapter,
                {
                    "history_months": history_months,
                    "normalization": "per_series",
                    "max_epochs": 20,
                    "patience": 5,
                    "max_train_windows_per_client": 256,
                    "training_stride": 1,
                    "validation_stride": 24,
                    "device": "cuda",
                },
            )
        for model_name, factory in (
            ("mlp", MLPAdapter),
            ("dlinear", DLinearAdapter),
            ("nlinear", NLinearAdapter),
        ):
            for lookback in (48, 168, 336):
                for normalization in ("none", "global", "per_series"):
                    if lookback == 168 and normalization == "per_series":
                        # Already registered above as the default configuration.
                        continue
                    add(
                        f"{model_name}-L{lookback}-{normalization}-h30",
                        factory,
                        {
                            "lookback": lookback,
                            "normalization": normalization,
                            "max_epochs": 20,
                            "patience": 5,
                            "max_train_windows_per_client": 256,
                            "training_stride": 1,
                            "validation_stride": 24,
                        },
                    )
    return experiments


def run_phase4(
    panel: Panel,
    *,
    output_dir: Path = Path("outputs"),
    models: tuple[str, ...] = DEFAULT_MODELS,
    smoke: bool = False,
    primary_only: bool = False,
) -> pd.DataFrame:
    """Execute selected phase 4 experiments and save a compact summary."""
    unknown = set(models).difference(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    if set(models).intersection({"mlp", "dlinear", "nlinear"}):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for MLP, DLinear, and NLinear. "
                "Install the CUDA-enabled requirements before running."
            )
    artifact_dir = output_dir / "models" / "phase4"
    experiments = [
        experiment
        for experiment in build_experiments(artifact_dir, smoke=smoke)
        if experiment[0].split("-")[0] in models
        and (not primary_only or smoke or experiment[0] in PRIMARY_CONFIG_IDS)
    ]
    evaluation_panel = panel
    if smoke:
        from src.data.splits import DEFAULT_SPLITS, daily_backtest_origins

        sample_size = min(8, len(panel.client_ids))
        evaluation_panel = Panel(
            panel.client_ids[:sample_size],
            panel.timestamps,
            panel.values[:sample_size],
        )
        test_start, test_end = DEFAULT_SPLITS["test"]
        origins = daily_backtest_origins(
            evaluation_panel.timestamps, test_start, test_end
        )[:1]
    else:
        origins = None

    rows = []
    for name, factory, config in experiments:
        result = backtest(
            factory(),
            evaluation_panel,
            origins=origins,
            config=config,
            output_dir=output_dir,
        )
        rows.append(
            {
                "experiment": name,
                "model": result.experiment["model"],
                "config_id": result.experiment["config_id"],
                "lookback": config["lookback"],
                "horizon": config["horizon"],
                "history_months": config["history_months"],
                "normalization": config.get("normalization", "raw"),
                "params_total": result.experiment["params_total"],
                "params_trainable": result.experiment["params_trainable"],
                "train_time_s": result.experiment["train_time_s"],
                "inference_time_s": result.experiment["inference_time_s"],
                **result.metrics,
            }
        )

    summary_dir = output_dir / "metrics"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_stem = "phase4_smoke_summary" if smoke else "phase4_summary"
    summary_path = summary_dir / f"{summary_stem}.csv"
    summary = pd.DataFrame(rows)
    if summary_path.exists():
        summary = pd.concat([pd.read_csv(summary_path), summary],
                            ignore_index=True).drop_duplicates(
                                "config_id", keep="last")
    summary.to_csv(summary_path, index=False)
    (summary_dir / f"{summary_stem}.json").write_text(
        summary.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/hourly.parquet"),
    )
    parser.add_argument(
        "--client-ids",
        type=Path,
        default=Path("data/processed/client_ids.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one origin with reduced training sizes to check the pipeline.",
    )
    parser.add_argument("--primary-only", action="store_true",
                        help="Run only the eight primary configurations")
    args = parser.parse_args()
    panel = load_panel_matrix(args.panel, args.client_ids)
    summary = run_phase4(
        panel,
        output_dir=args.output_dir,
        models=tuple(args.models),
        smoke=args.smoke,
        primary_only=args.primary_only,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
