"""Chronological target windows and day-ahead backtest origins."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_SPLITS = {
    "train": (pd.Timestamp("2012-01-01"), pd.Timestamp("2014-07-01")),
    "validation": (pd.Timestamp("2014-07-01"), pd.Timestamp("2014-10-01")),
    "test": (pd.Timestamp("2014-10-01"), pd.Timestamp("2015-01-01")),
}
HOUR_DELTA = pd.Timedelta(1, unit="h")


def _validate_hourly_index(timestamps: pd.DatetimeIndex) -> None:
    if timestamps.tz is not None:
        raise ValueError("Use naive Portuguese local timestamps")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Timestamps must be unique and increasing")
    if len(timestamps) > 1 and not timestamps.to_series().diff().iloc[1:].eq(
        HOUR_DELTA
    ).all():
        raise ValueError("Timestamps must form a continuous hourly grid")


def target_origin_indices(
    timestamps: pd.DatetimeIndex,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    horizon: int = 24,
    stride: int = 1,
    lookback: int = 168,
) -> np.ndarray:
    """Return origins whose complete target window stays inside one split."""
    timestamps = pd.DatetimeIndex(timestamps)
    _validate_hourly_index(timestamps)
    if horizon < 1 or stride < 1 or lookback < 0:
        raise ValueError("horizon and stride must be positive; lookback nonnegative")
    start, end = pd.Timestamp(split_start), pd.Timestamp(split_end)
    start_index = int(timestamps.searchsorted(start))
    end_index = int(timestamps.searchsorted(end))
    if start_index >= len(timestamps) or timestamps[start_index] != start:
        raise ValueError(f"Split start {start} is not present in the panel")
    if end_index < len(timestamps) and timestamps[end_index] != end:
        raise ValueError(f"Split end {end} is not present in the panel")
    if end_index - start_index < horizon:
        return np.empty(0, dtype=np.int64)

    origins = np.arange(
        max(start_index, lookback),
        end_index - horizon + 1,
        stride,
        dtype=np.int64,
    )
    for origin in origins:
        if timestamps[origin] < start or timestamps[origin + horizon - 1] >= end:
            raise ValueError("A target window crosses its chronological split")
    return origins


def temporal_split_origins(
    timestamps: pd.DatetimeIndex,
    lookback: int = 168,
    horizon: int = 24,
    stride: int = 1,
    split_ranges: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] = DEFAULT_SPLITS,
) -> dict[str, np.ndarray]:
    """Create train, validation, and test origins from the fixed date ranges."""
    origins = {
        name: target_origin_indices(
            timestamps,
            start,
            end,
            horizon=horizon,
            stride=stride,
            lookback=lookback,
        )
        for name, (start, end) in split_ranges.items()
    }
    validate_no_target_overlap(origins, horizon)
    return origins


def validate_no_target_overlap(
    split_origins: Mapping[str, Sequence[int]], horizon: int
) -> None:
    """Reject random windows that overlap or violate chronological split order."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    target_sets = {
        split: {
            target_index
            for origin in origins
            for target_index in range(int(origin), int(origin) + horizon)
        }
        for split, origins in split_origins.items()
    }
    split_names = list(target_sets)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            first_targets = target_sets[first]
            second_targets = target_sets[second]
            if first_targets.intersection(second_targets):
                raise ValueError(
                    f"Target timestamps overlap between {first} and {second}"
                )
            if (
                first_targets
                and second_targets
                and max(first_targets) >= min(second_targets)
            ):
                raise ValueError(
                    f"Target timestamps are out of chronological order: {first} then {second}"
                )


def daily_backtest_origins(
    timestamps: pd.DatetimeIndex,
    split_start: pd.Timestamp = DEFAULT_SPLITS["test"][0],
    split_end: pd.Timestamp = DEFAULT_SPLITS["test"][1],
    horizon: int = 24,
) -> pd.DatetimeIndex:
    """Return one midnight forecast origin per eligible test day."""
    timestamps = pd.DatetimeIndex(timestamps)
    _validate_hourly_index(timestamps)
    start, end = pd.Timestamp(split_start), pd.Timestamp(split_end)
    start_index = int(timestamps.searchsorted(start))
    end_index = int(timestamps.searchsorted(end))
    if start_index >= len(timestamps) or timestamps[start_index] != start:
        raise ValueError(f"Backtest start {start} is not present in the panel")
    origins = timestamps[start_index:end_index:24]
    if any(origin.hour != 0 for origin in origins):
        raise ValueError("Daily backtest origins must be at midnight")
    if len(origins) and origins[-1] + horizon * HOUR_DELTA > end:
        raise ValueError("The final origin does not have a full target horizon")
    return origins


def best_epoch_from_validation(
    validation_losses: Sequence[float], monitored_split: str = "validation"
) -> int:
    """Select an epoch using validation loss only, never test loss."""
    if monitored_split != "validation":
        raise ValueError("Early stopping may monitor validation, not test")
    losses = np.asarray(validation_losses, dtype=np.float64)
    if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all():
        raise ValueError("validation_losses must be a non-empty finite sequence")
    return int(np.argmin(losses)) + 1
