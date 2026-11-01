"""Leakage-safe rolling-origin evaluation for global multi-series adapters."""

import hashlib
import json
import platform
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from src.data.calendars import calendar_features
from src.data.panel import Panel
from src.data.splits import DEFAULT_SPLITS, daily_backtest_origins
from src.evaluation.metrics import MetricTables, evaluate_predictions, seasonal_mase_scales

HOUR = pd.Timedelta(1, unit="h")


class ForecastAdapter(Protocol):
    """Interface implemented by each model family."""

    name: str

    def predict(
        self,
        history: Panel,
        future_calendar: pd.DataFrame,
        horizon: int = 24,
        origin: pd.Timestamp | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, float | int | None]
    by_client: pd.DataFrame
    by_horizon: pd.DataFrame
    by_hour: pd.DataFrame
    christmas_week: pd.DataFrame
    experiment: dict[str, Any]


def backtest(
    model: ForecastAdapter,
    panel: Panel,
    *,
    origins: Sequence[pd.Timestamp] | None = None,
    split_ranges: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] = DEFAULT_SPLITS,
    config: Mapping[str, Any] | None = None,
    output_dir: Path | None = Path("outputs"),
) -> BacktestResult:
    """Fit once, then forecast each daily origin without exposing its targets."""
    settings = _settings(config)
    _validate_panel(panel)
    ranges = _normalize_splits(split_ranges)
    _validate_split_boundaries(panel, ranges)
    train_all = _panel_slice(panel, *ranges["train"])
    validation = _panel_slice(panel, *ranges["validation"])
    if not len(train_all.timestamps):
        raise ValueError("Training split cannot be empty")
    train = _tail_months(train_all, settings["history_months"])
    fit_train, fit_validation = train, validation
    if settings["refit_train_validation"]:
        if settings["best_epoch"] is None or settings["best_epoch"] < 1:
            raise ValueError(
                "Final refit on train+validation requires best_epoch from validation"
            )
        fit_train = _tail_months(
            _concat_panels(train_all, validation), settings["history_months"]
        )
        fit_validation = None
    selected_origins = _validated_origins(
        panel, ranges["test"], settings["horizon"], origins
    )
    mase_train = fit_train if settings["refit_train_validation"] else train
    scale_table = seasonal_mase_scales(
        mase_train.values,
        mase_train.timestamps,
        mase_train.client_ids,
        mase_train.timestamps[0],
        mase_train.timestamps[-1] + HOUR,
        seasonality=settings["mase_seasonality"],
    )
    fit_period = _fit_period(
        mase_train.timestamps[0], mase_train.timestamps[-1] + HOUR
    )
    _reset_gpu_peak()
    fit_metadata, train_time_s = _timed_fit(
        model, fit_train, fit_validation, settings
    )

    predictions, inference_time_s = _collect_predictions(
        model,
        panel,
        selected_origins,
        settings,
        scale_table,
        fit_period,
    )
    peak_memory = _gpu_peak_memory_mb()
    return _finish_result(
        model,
        predictions,
        settings,
        train_time_s=train_time_s,
        inference_time_s=inference_time_s,
        gpu_peak_mb=peak_memory,
        fit_metadata=fit_metadata,
        fit_count=1 if callable(getattr(model, "fit", None)) else 0,
        output_dir=output_dir,
    )


def monthly_refit_backtest(
    model_factory: Callable[[], ForecastAdapter],
    panel: Panel,
    *,
    split_ranges: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] = DEFAULT_SPLITS,
    config: Mapping[str, Any] | None = None,
    output_dir: Path | None = Path("outputs"),
) -> BacktestResult:
    """Refit at each test-month boundary using only observations before it."""
    settings = _settings(config)
    _validate_panel(panel)
    ranges = _normalize_splits(split_ranges)
    _validate_split_boundaries(panel, ranges)
    test_start, test_end = ranges["test"]
    month_starts = pd.date_range(
        test_start.normalize(), test_end.normalize(), freq="MS", inclusive="left"
    )
    if (
        test_start != test_start.normalize()
        or test_start.day != 1
        or test_end != test_end.normalize()
        or test_end.day != 1
    ):
        raise ValueError("Monthly refits require month-aligned test boundaries")
    if not len(month_starts):
        raise ValueError("Test split contains no complete calendar month")

    _reset_gpu_peak()
    all_predictions = []
    fit_records = []
    total_train_time = 0.0
    total_inference_time = 0.0
    fit_count = 0
    for month_start in month_starts:
        month_end = min(month_start + pd.offsets.MonthBegin(1), test_end)
        month_origins = _validated_origins(
            panel,
            (month_start, month_end),
            settings["horizon"],
            origins=None,
        )
        train_start = ranges["train"][0]
        # Monthly refits use all observations since the fixed training start.
        # A rolling history_months cap would not be an expanding window.
        train = _panel_slice(panel, train_start, month_start)
        if not len(train.timestamps):
            raise ValueError(
                f"No training history is available before {month_start}"
            )
        model = model_factory()
        month_config = {
            **settings,
            "refit_month": month_start.isoformat(),
            "final_epochs": settings.get("best_epoch"),
        }
        fit_metadata, train_time_s = _timed_fit(model, train, None, month_config)
        scales = seasonal_mase_scales(
            train.values,
            train.timestamps,
            train.client_ids,
            train.timestamps[0],
            month_start,
            seasonality=settings["mase_seasonality"],
        )
        period = _fit_period(train.timestamps[0], train.timestamps[-1] + HOUR)
        forecasts, inference_time_s = _collect_predictions(
            model,
            panel,
            month_origins,
            settings,
            scales,
            period,
        )
        all_predictions.append(forecasts)
        total_train_time += train_time_s
        total_inference_time += inference_time_s
        fit_count += 1 if callable(getattr(model, "fit", None)) else 0
        fit_records.append(
            {
                "month": month_start.date().isoformat(),
                "train_end_exclusive": month_start.isoformat(),
                "train_time_s": train_time_s,
                "fit_metadata": fit_metadata,
            }
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    return _finish_result(
        model,
        predictions,
        settings,
        train_time_s=total_train_time,
        inference_time_s=total_inference_time,
        gpu_peak_mb=_gpu_peak_memory_mb(),
        fit_metadata={"monthly_refits": fit_records},
        fit_count=fit_count,
        output_dir=output_dir,
    )


def _collect_predictions(
    model: ForecastAdapter,
    panel: Panel,
    origins: pd.DatetimeIndex,
    settings: dict[str, Any],
    scales: pd.DataFrame,
    fit_period: str,
) -> tuple[pd.DataFrame, float]:
    horizon = settings["horizon"]
    scale_values = scales["mase_scale"].to_numpy(dtype=np.float64)
    if len(scales) != len(panel.client_ids):
        raise ValueError("MASE scales do not cover every panel client")

    rows = []
    _synchronize_cuda()
    inference_started = time.perf_counter()
    for origin in origins:
        origin_index = panel.timestamps.get_loc(origin)
        if origin_index < settings["lookback"]:
            raise ValueError(f"Insufficient history before origin {origin}")
        history = Panel(
            client_ids=panel.client_ids,
            timestamps=panel.timestamps[:origin_index],
            values=panel.values[:, :origin_index],
        )
        if history.timestamps[-1] != origin - HOUR:
            raise ValueError(f"History does not end at 23:00 before {origin}")

        target_times = panel.timestamps[origin_index : origin_index + horizon]
        if len(target_times) != horizon or target_times[0] != origin:
            raise ValueError(f"Origin {origin} does not have a full target horizon")
        future_calendar = calendar_features(
            target_times, encoding=settings["calendar_encoding"]
        )
        if not future_calendar.index.equals(target_times):
            raise ValueError("Future calendar does not align with target timestamps")
        if any(column in future_calendar.columns for column in ("y", "y_true")):
            raise ValueError("Future calendar must not contain target values")

        forecast = np.asarray(
            model.predict(history, future_calendar, horizon=horizon, origin=origin),
            dtype=np.float64,
        )
        if forecast.shape != (len(panel.client_ids), horizon):
            raise ValueError(
                f"Model returned shape {forecast.shape}, expected "
                f"({len(panel.client_ids)}, {horizon})"
            )
        if not np.isfinite(forecast).all():
            raise ValueError(f"Model returned non-finite forecasts at {origin}")

        # Targets are read only after the model has returned its forecast.
        actual = panel.values[:, origin_index : origin_index + horizon]
        rows.append(
            pd.DataFrame(
                {
                    "unique_id": np.repeat(panel.client_ids, horizon),
                    "ds": np.tile(target_times.to_numpy(), len(panel.client_ids)),
                    "model": _model_name(model),
                    "y_hat": forecast.reshape(-1),
                    "origin": origin,
                    "y_true": actual.reshape(-1),
                    "horizon_step": np.tile(np.arange(1, horizon + 1), len(panel.client_ids)),
                    "mase_scale": np.repeat(scale_values, horizon),
                    "fit_period": fit_period,
                    "seed": settings["seed"],
                    "config_id": settings["config_id"],
                }
            )
        )
    _synchronize_cuda()
    elapsed = time.perf_counter() - inference_started
    predictions = pd.concat(rows, ignore_index=True)
    predictions = predictions.sort_values(
        ["origin", "unique_id", "ds"], kind="stable"
    ).reset_index(drop=True)
    unique_columns = ["unique_id", "ds", "model", "config_id"]
    if predictions.duplicated(unique_columns).any():
        raise ValueError("Backtest generated duplicate client/timestamp predictions")
    return predictions, elapsed


def _fit(
    model: ForecastAdapter,
    train: Panel,
    validation: Panel | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fit_method = getattr(model, "fit", None)
    if not callable(fit_method):
        return {"fit_skipped": "zero-shot adapter has no fit method"}
    result = fit_method(train, validation, config)
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise TypeError("fit must return a mapping or None")
    return dict(result)


def _timed_fit(
    model: ForecastAdapter,
    train: Panel,
    validation: Panel | None,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    if not callable(getattr(model, "fit", None)):
        return _fit(model, train, validation, config), 0.0
    _synchronize_cuda()
    started = time.perf_counter()
    metadata = _fit(model, train, validation, config)
    _synchronize_cuda()
    return metadata, time.perf_counter() - started


def _finish_result(
    model: ForecastAdapter,
    predictions: pd.DataFrame,
    settings: dict[str, Any],
    *,
    train_time_s: float,
    inference_time_s: float,
    gpu_peak_mb: float | str,
    fit_metadata: Mapping[str, Any],
    fit_count: int,
    output_dir: Path | None,
) -> BacktestResult:
    tables: MetricTables = evaluate_predictions(predictions)
    name = _model_name(model)
    params_total, params_trainable = _parameter_counts(model)
    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "N/A",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    experiment: dict[str, Any] = {
        "model": name,
        "config_id": settings["config_id"],
        "seed": settings["seed"],
        "L": settings["lookback"],
        "H": settings["horizon"],
        "history_months": settings["history_months"],
        "normalization": settings["normalization"],
        "params_total": params_total,
        "params_trainable": params_trainable,
        "train_time_s": train_time_s,
        "inference_time_s": inference_time_s,
        "gpu_peak_mb": gpu_peak_mb,
        "hardware": hardware,
        "metrics": tables.overall,
        "fit_count": fit_count,
        "fit_metadata": _json_safe(dict(fit_metadata)),
        "refit_train_validation": settings["refit_train_validation"],
        "best_epoch": settings.get("best_epoch"),
        "artifacts": {},
    }
    if output_dir is not None:
        experiment["artifacts"] = _save_artifacts(
            predictions, tables, experiment, Path(output_dir)
        )
    return BacktestResult(
        predictions=predictions,
        metrics=tables.overall,
        by_client=tables.by_client,
        by_horizon=tables.by_horizon,
        by_hour=tables.by_hour,
        christmas_week=tables.christmas_week,
        experiment=experiment,
    )


def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    provided = dict(config or {})
    settings: dict[str, Any] = {
        **provided,
        "lookback": int(provided.get("lookback", 168)),
        "horizon": int(provided.get("horizon", 24)),
        "seed": int(provided.get("seed", 42)),
        "history_months": int(provided.get("history_months", 30)),
        "normalization": provided.get("normalization", "raw"),
        "calendar_encoding": provided.get("calendar_encoding", "cyclic"),
        "mase_seasonality": int(provided.get("mase_seasonality", 24)),
        "refit_train_validation": bool(
            provided.get("refit_train_validation", False)
        ),
        "best_epoch": (
            int(provided["best_epoch"])
            if provided.get("best_epoch") is not None
            else None
        ),
    }
    if settings["lookback"] < 1 or settings["horizon"] < 1:
        raise ValueError("lookback and horizon must be positive")
    if settings["calendar_encoding"] not in {"cyclic", "onehot"}:
        raise ValueError("calendar_encoding must be cyclic or onehot")
    config_payload = {
        key: _json_safe(value)
        for key, value in provided.items()
        if key != "config_id"
    }
    config_id = str(
        provided.get("config_id")
        or hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()[:12]
    )
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", config_id):
        raise ValueError("config_id may contain only letters, numbers, '.', '_' and '-'")
    settings["config_id"] = config_id
    return settings


def _normalize_splits(
    split_ranges: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    missing = {"train", "validation", "test"}.difference(split_ranges)
    if missing:
        raise ValueError(f"Split ranges are missing: {sorted(missing)}")
    ranges = {
        name: (pd.Timestamp(start), pd.Timestamp(end))
        for name, (start, end) in split_ranges.items()
    }
    if any(
        start.tz is not None or end.tz is not None
        for start, end in ranges.values()
    ):
        raise ValueError("Split boundaries must use naive local timestamps")
    if not (ranges["train"][0] < ranges["train"][1]):
        raise ValueError("Training split must have positive duration")
    if not (
        ranges["train"][1] <= ranges["validation"][0]
        and ranges["validation"][0] < ranges["validation"][1]
        and ranges["validation"][1] <= ranges["test"][0]
        and ranges["test"][0] < ranges["test"][1]
    ):
        raise ValueError("Train, validation, and test splits must be chronological")
    return ranges


def _validated_origins(
    panel: Panel,
    test_range: tuple[pd.Timestamp, pd.Timestamp],
    horizon: int,
    origins: Sequence[pd.Timestamp] | None,
) -> pd.DatetimeIndex:
    start, end = test_range
    if origins is None:
        selected = daily_backtest_origins(
            panel.timestamps, split_start=start, split_end=end, horizon=horizon
        )
    else:
        selected = pd.DatetimeIndex(origins)
    if selected.empty:
        raise ValueError("At least one backtest origin is required")
    if (
        selected.hasnans
        or selected.tz is not None
        or not selected.is_monotonic_increasing
    ):
        raise ValueError("Origins must be naive and chronologically sorted")
    if selected.has_duplicates:
        raise ValueError("Backtest origins must be unique")
    if any(origin.hour != 0 for origin in selected):
        raise ValueError("Backtest origins must be local midnight")
    for origin in selected:
        index = panel.timestamps.get_indexer([origin])[0]
        if index < 0 or origin < start or origin + horizon * HOUR > end:
            raise ValueError(f"Origin {origin} is outside the complete test split")
        if index + horizon > len(panel.timestamps):
            raise ValueError(f"Origin {origin} has an incomplete forecast horizon")
    return selected


def _panel_slice(
    panel: Panel, start: pd.Timestamp, end: pd.Timestamp
) -> Panel:
    mask = (panel.timestamps >= start) & (panel.timestamps < end)
    return Panel(
        client_ids=panel.client_ids,
        timestamps=panel.timestamps[mask],
        values=panel.values[:, mask],
    )


def _concat_panels(first: Panel, second: Panel) -> Panel:
    if first.client_ids != second.client_ids:
        raise ValueError("Cannot combine panels with different client IDs")
    if len(first.timestamps) and len(second.timestamps):
        if first.timestamps[-1] + HOUR != second.timestamps[0]:
            raise ValueError("Train and validation panels are not contiguous")
    return Panel(
        client_ids=first.client_ids,
        timestamps=first.timestamps.append(second.timestamps),
        values=np.concatenate((first.values, second.values), axis=1),
    )


def _tail_months(panel: Panel, months: int) -> Panel:
    if months < 1:
        raise ValueError("history_months must be a positive integer")
    if not len(panel.timestamps):
        return panel
    exclusive_end = panel.timestamps[-1] + HOUR
    start = exclusive_end - pd.DateOffset(months=months)
    return _panel_slice(panel, start, exclusive_end)


def _validate_panel(panel: Panel) -> None:
    timestamps = pd.DatetimeIndex(panel.timestamps)
    if not panel.client_ids:
        raise ValueError("Panel must contain at least one client")
    if timestamps.tz is not None:
        raise ValueError("Panel timestamps must be naive local time")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Panel timestamps must be unique and increasing")
    if len(timestamps) < 2 or not timestamps.to_series().diff().iloc[1:].eq(
        HOUR
    ).all():
        raise ValueError("Panel timestamps must be a continuous hourly grid")
    if len(set(panel.client_ids)) != len(panel.client_ids):
        raise ValueError("Panel client IDs must be unique")
    if not np.isfinite(panel.values).all():
        raise ValueError("Panel targets must be finite")


def _validate_split_boundaries(
    panel: Panel, ranges: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]]
) -> None:
    timestamps = panel.timestamps
    for name, (start, end) in ranges.items():
        start_index = timestamps.searchsorted(start)
        if start_index == len(timestamps) or timestamps[start_index] != start:
            raise ValueError(f"{name} split start {start} is missing from the panel")
        end_index = timestamps.searchsorted(end)
        if end_index < len(timestamps):
            if timestamps[end_index] != end:
                raise ValueError(f"{name} split end {end} is not on the hourly grid")
        elif end != timestamps[-1] + HOUR:
            raise ValueError(f"{name} split end {end} is outside the panel")


def _fit_period(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.isoformat()}/{pd.Timestamp(end).isoformat()}"


def _model_name(model: ForecastAdapter) -> str:
    name = getattr(model, "name", None)
    return str(name if name else type(model).__name__)


def _parameter_counts(model: ForecastAdapter) -> tuple[int | None, int | None]:
    count_method = getattr(model, "parameter_counts", None)
    if callable(count_method):
        counts = count_method()
        if (
            not isinstance(counts, tuple)
            or len(counts) != 2
            or any(
                count is not None and (not isinstance(count, int) or count < 0)
                for count in counts
            )
        ):
            raise TypeError("parameter_counts must return two nonnegative integers")
        return counts
    module = model
    if not callable(getattr(module, "parameters", None)):
        module = getattr(model, "module", getattr(model, "model", None))
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None, None
    values = list(parameters())
    return (
        sum(parameter.numel() for parameter in values),
        sum(parameter.numel() for parameter in values if parameter.requires_grad),
    )


def _reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_peak_memory_mb() -> float | str:
    if not torch.cuda.is_available():
        return "N/A"
    return round(torch.cuda.max_memory_allocated() / (1024**2), 3)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Cannot serialize experiment metadata of type {type(value).__name__}"
    )


def _save_artifacts(
    predictions: pd.DataFrame,
    tables: MetricTables,
    experiment: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    slug = (
        re.sub(r"[^a-zA-Z0-9._-]+", "-", experiment["model"])
        .strip("-")
        .lower()
        or "model"
    )
    result_dir = output_dir / "predictions" / slug / experiment["config_id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "predictions": result_dir / "predictions.parquet",
        "metrics": result_dir / "metrics.json",
        "by_client": result_dir / "metrics_by_client.csv",
        "by_horizon": result_dir / "metrics_by_horizon.csv",
        "by_hour": result_dir / "metrics_by_hour.csv",
        "christmas_week": result_dir / "metrics_christmas_week.csv",
        "experiment": result_dir / "experiment.json",
    }
    predictions.to_parquet(files["predictions"], index=False)
    tables.by_client.to_csv(files["by_client"], index=False)
    tables.by_horizon.to_csv(files["by_horizon"], index=False)
    tables.by_hour.to_csv(files["by_hour"], index=False)
    tables.christmas_week.to_csv(files["christmas_week"], index=False)
    files["metrics"].write_text(
        json.dumps(_json_safe(tables.overall), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths = {key: str(path) for key, path in files.items()}
    experiment["artifacts"] = paths
    files["experiment"].write_text(
        json.dumps(_json_safe(experiment), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return paths
