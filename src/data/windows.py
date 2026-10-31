"""PyTorch windows with separate, known-in-advance future calendar features."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.calendars import calendar_features

HOUR_DELTA = pd.Timedelta(1, unit="h")


def validate_future_calendar_alignment(
    target_timestamps: pd.DatetimeIndex,
    feature_timestamps: pd.DatetimeIndex,
) -> None:
    """Reject future calendar covariates that are shifted from their targets."""
    targets = pd.DatetimeIndex(target_timestamps)
    features = pd.DatetimeIndex(feature_timestamps)
    if not targets.equals(features):
        raise ValueError("Future calendar timestamps must match target timestamps")


class MultiSeriesWindowDataset(Dataset):
    """Create one-step-stride windows over multiple hourly customer series."""

    def __init__(
        self,
        values: np.ndarray,
        timestamps: pd.DatetimeIndex,
        client_ids: Sequence[str],
        origin_indices: Sequence[int],
        lookback: int = 168,
        horizon: int = 24,
        calendar: pd.DataFrame | None = None,
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        timestamps = pd.DatetimeIndex(timestamps)
        client_ids = tuple(str(client_id) for client_id in client_ids)
        origins = np.asarray(origin_indices, dtype=np.int64)
        if values.ndim != 2 or values.shape != (len(client_ids), len(timestamps)):
            raise ValueError("values must have shape (clients, timestamps)")
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("Client IDs must be unique")
        if lookback < 1 or horizon < 1:
            raise ValueError("lookback and horizon must be positive")
        if not np.isfinite(values).all():
            raise ValueError("Window targets must be finite")
        if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
            raise ValueError("Timestamps must be unique and increasing")
        if len(timestamps) > 1 and not timestamps.to_series().diff().iloc[1:].eq(
            HOUR_DELTA
        ).all():
            raise ValueError("Dataset timestamps must be a continuous hourly grid")
        if len(origins) and (
            origins.min() < lookback or origins.max() + horizon > len(timestamps)
        ):
            raise ValueError("An origin does not have enough history or targets")
        if len(origins) > 1 and np.any(np.diff(origins) <= 0):
            raise ValueError("Origin indices must be strictly increasing")

        if calendar is None:
            calendar = calendar_features(timestamps)
        if not calendar.index.equals(timestamps):
            raise ValueError("Calendar features must align with the full timestamp index")

        self.values = values
        self.timestamps = timestamps
        self.client_ids = client_ids
        self.origin_indices = origins
        self.lookback = lookback
        self.horizon = horizon
        self.calendar_values = calendar.to_numpy(dtype=np.float32)
        self.calendar_timestamps = pd.DatetimeIndex(calendar.index)
        self.calendar_columns = tuple(calendar.columns)

    def __len__(self) -> int:
        return len(self.client_ids) * len(self.origin_indices)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        origins_per_client = len(self.origin_indices)
        client_index, origin_position = divmod(index, origins_per_client)
        origin_index = int(self.origin_indices[origin_position])
        history_start = origin_index - self.lookback
        target_end = origin_index + self.horizon
        target_times = self.timestamps[origin_index:target_end]
        validate_future_calendar_alignment(
            target_times, self.calendar_timestamps[origin_index:target_end]
        )

        historical_calendar = self.calendar_values[history_start:origin_index]
        history_values = self.values[client_index, history_start:origin_index, None]
        history = np.concatenate((history_values, historical_calendar), axis=1)
        targets = self.values[client_index, origin_index:target_end]
        future_calendar = self.calendar_values[origin_index:target_end]

        return {
            "X": torch.from_numpy(np.ascontiguousarray(history)),
            "y": torch.from_numpy(np.ascontiguousarray(targets)),
            "future_calendar": torch.from_numpy(np.ascontiguousarray(future_calendar)),
            "unique_id": self.client_ids[client_index],
            "origin": self.timestamps[origin_index].isoformat(sep=" "),
            "target_timestamps": [
                timestamp.isoformat(sep=" ") for timestamp in target_times
            ],
        }
