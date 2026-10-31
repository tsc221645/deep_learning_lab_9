"""Known-in-advance calendar covariates for Portuguese local time."""

from collections.abc import Iterable

import holidays
import numpy as np
import pandas as pd


def portugal_holidays(years: Iterable[int]) -> dict:
    """Return Portuguese public holidays keyed by local calendar date."""
    return dict(holidays.country_holidays("PT", years=sorted(set(years))))


def calendar_features(
    timestamps: pd.DatetimeIndex, encoding: str = "cyclic"
) -> pd.DataFrame:
    """Build future-known hour, weekday, month, and holiday features."""
    timestamps = pd.DatetimeIndex(timestamps)
    if timestamps.tz is not None:
        raise ValueError("Use the dataset's naive Portuguese local timestamps")
    if encoding not in {"cyclic", "onehot"}:
        raise ValueError("encoding must be 'cyclic' or 'onehot'")

    holiday_map = portugal_holidays(timestamps.year)
    is_holiday = np.fromiter(
        (timestamp.date() in holiday_map for timestamp in timestamps),
        dtype=np.float32,
        count=len(timestamps),
    )
    if encoding == "cyclic":
        hour = timestamps.hour.to_numpy()
        weekday = timestamps.dayofweek.to_numpy()
        month = timestamps.month.to_numpy() - 1
        features = {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "is_holiday": is_holiday,
        }
        return pd.DataFrame(features, index=timestamps, dtype=np.float32)

    features = np.concatenate(
        [
            np.eye(24, dtype=np.float32)[timestamps.hour],
            np.eye(7, dtype=np.float32)[timestamps.dayofweek],
            np.eye(12, dtype=np.float32)[timestamps.month - 1],
            is_holiday[:, None],
        ],
        axis=1,
    )
    columns = (
        [f"hour_{hour:02d}" for hour in range(24)]
        + [f"weekday_{day}" for day in range(7)]
        + [f"month_{month:02d}" for month in range(1, 13)]
        + ["is_holiday"]
    )
    return pd.DataFrame(features, index=timestamps, columns=columns)
