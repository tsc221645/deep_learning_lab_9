"""Point-forecast metrics on the original energy scale."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAPE_ZERO_THRESHOLD = 1e-8
PREDICTION_COLUMNS = {"unique_id", "ds", "y_true", "y_hat", "mase_scale"}


@dataclass(frozen=True)
class MetricTables:
    overall: dict[str, float | int | None]
    by_client: pd.DataFrame
    by_horizon: pd.DataFrame
    by_hour: pd.DataFrame
    christmas_week: pd.DataFrame


def seasonal_mase_scales(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    client_ids: tuple[str, ...],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    seasonality: int = 24,
) -> pd.DataFrame:
    """Compute one seasonal-naive scale per client from training data only."""
    values = np.asarray(values, dtype=np.float64)
    timestamps = pd.DatetimeIndex(timestamps)
    if values.ndim != 2 or values.shape != (len(client_ids), len(timestamps)):
        raise ValueError("values must have shape (clients, timestamps)")
    if seasonality < 1:
        raise ValueError("seasonality must be positive")

    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    mask = (timestamps >= start) & (timestamps < end)
    training = values[:, mask]
    if training.shape[1] <= seasonality:
        raise ValueError("Training data is too short for the MASE seasonality")
    if not np.isfinite(training).all():
        raise ValueError("Training targets must be finite")

    scales = np.abs(training[:, seasonality:] - training[:, :-seasonality]).mean(
        axis=1
    )
    return pd.DataFrame({"unique_id": client_ids, "mase_scale": scales})


def evaluate_predictions(predictions: pd.DataFrame) -> MetricTables:
    """Calculate aggregate and diagnostic metrics from long-format predictions."""
    missing = PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Predictions cannot be empty")

    frame = predictions.copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    numeric = frame[["y_true", "y_hat", "mase_scale"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric[:, :2]).all():
        raise ValueError("Targets and forecasts must be finite")
    if np.isinf(numeric[:, 2]).any() or np.isnan(numeric[:, 2]).any():
        raise ValueError("MASE scales must be finite")
    if (numeric[:, 2] < 0).any():
        raise ValueError("MASE scales cannot be negative")

    frame["abs_error"] = np.abs(frame["y_true"] - frame["y_hat"])
    frame["sq_error"] = np.square(frame["y_true"] - frame["y_hat"])
    mase_group_columns = ["unique_id"]
    if "fit_period" in frame.columns:
        mase_group_columns.append("fit_period")

    overall = _summarize(frame, mase_group_columns)
    by_client = _grouped_metrics(frame, ["unique_id"], mase_group_columns)
    if "horizon_step" in frame.columns:
        by_horizon = _grouped_metrics(frame, ["horizon_step"], mase_group_columns)
    else:
        by_horizon = pd.DataFrame()
    frame["hour"] = frame["ds"].dt.hour
    by_hour = _grouped_metrics(frame, ["hour"], mase_group_columns)

    christmas_mask = (
        frame["ds"].dt.month.eq(12)
        & frame["ds"].dt.day.between(22, 28)
    )
    christmas = frame.loc[christmas_mask].copy()
    christmas["date"] = christmas["ds"].dt.date.astype(str)
    christmas_week = _grouped_metrics(christmas, ["date"], mase_group_columns)

    return MetricTables(
        overall=overall,
        by_client=by_client,
        by_horizon=by_horizon,
        by_hour=by_hour,
        christmas_week=christmas_week,
    )


def _summarize(
    frame: pd.DataFrame, mase_group_columns: list[str]
) -> dict[str, float | int | None]:
    absolute_errors = frame["abs_error"].to_numpy(dtype=np.float64)
    targets = frame["y_true"].to_numpy(dtype=np.float64)
    target_magnitude = np.abs(targets)
    mape_mask = target_magnitude > MAPE_ZERO_THRESHOLD
    denominator = float(target_magnitude.sum())
    wape_pct = (
        100.0 * float(absolute_errors.sum()) / denominator
        if denominator > 0
        else None
    )
    mape_pct = (
        100.0
        * float((absolute_errors[mape_mask] / target_magnitude[mape_mask]).mean())
        if mape_mask.any()
        else None
    )

    series = (
        frame.groupby(mase_group_columns, sort=True)
        .agg(
            mae=("abs_error", "mean"),
            mase_scale=("mase_scale", "first"),
            scale_count=("mase_scale", "nunique"),
        )
        .reset_index()
    )
    if (series["scale_count"] > 1).any():
        raise ValueError("Each client and fit period must have one MASE scale")
    defined = series["mase_scale"] > 0
    mase_values = series.loc[defined, "mae"] / series.loc[defined, "mase_scale"]
    mase = float(mase_values.mean()) if not mase_values.empty else None

    return {
        "n_observations": int(len(frame)),
        "n_series": int(frame["unique_id"].nunique()),
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(frame["sq_error"].mean())),
        "wape_pct": wape_pct,
        "mape_pct": mape_pct,
        "mape_excluded_count": int((~mape_mask).sum()),
        "mase": mase,
        "mase_undefined_series_count": int((~defined).sum()),
    }


def _grouped_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
    mase_group_columns: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_columns + ["mae", "rmse", "mase"])
    rows = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(group_columns, key_values, strict=True)),
                **_summarize(group, mase_group_columns),
            }
        )
    return pd.DataFrame(rows)
