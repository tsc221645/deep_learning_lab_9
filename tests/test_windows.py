import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.windows import (
    MultiSeriesWindowDataset,
    validate_future_calendar_alignment,
)


def test_window_shapes_and_future_calendar_are_separate():
    timestamps = pd.date_range("2012-01-01", periods=240, freq="h")
    values = np.vstack([np.arange(240), np.arange(240) + 100]).astype(np.float32)
    dataset = MultiSeriesWindowDataset(
        values,
        timestamps,
        ["a", "b"],
        origin_indices=[168, 169],
        lookback=168,
        horizon=24,
    )

    sample = dataset[0]
    assert sample["X"].shape == (168, 8)
    assert sample["y"].shape == (24,)
    assert sample["future_calendar"].shape == (24, 7)
    assert sample["unique_id"] == "a"
    assert sample["origin"] == timestamps[168].isoformat(sep=" ")
    assert sample["target_timestamps"][0] == timestamps[168].isoformat(sep=" ")
    assert sample["X"][-1, 0] == values[0, 167]
    assert sample["y"][0] == values[0, 168]

    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["X"].shape == (2, 168, 8)
    assert batch["y"].shape == (2, 24)
    assert batch["future_calendar"].shape == (2, 24, 7)


def test_shifted_future_calendar_is_rejected():
    targets = pd.date_range("2014-10-01", periods=24, freq="h")
    shifted = targets + pd.Timedelta(1, unit="h")

    with pytest.raises(ValueError, match="must match target timestamps"):
        validate_future_calendar_alignment(targets, shifted)
