import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    best_epoch_from_validation,
    daily_backtest_origins,
    temporal_split_origins,
    validate_no_target_overlap,
)


def test_temporal_splits_keep_complete_targets_inside_each_date_range():
    timestamps = pd.date_range(
        "2011-01-01 01:00", "2014-12-31 23:00", freq="h"
    )

    origins = temporal_split_origins(timestamps, lookback=168, horizon=24)

    assert {name: len(indices) for name, indices in origins.items()} == {
        "train": 21865,
        "validation": 2185,
        "test": 2185,
    }


def test_random_window_split_with_overlapping_targets_is_rejected():
    with pytest.raises(ValueError, match="Target timestamps overlap"):
        validate_no_target_overlap(
            {"train": [10, 40], "validation": [20], "test": [70]}, horizon=24
        )


def test_random_disjoint_windows_out_of_order_are_rejected():
    with pytest.raises(ValueError, match="out of chronological order"):
        validate_no_target_overlap(
            {"train": [50], "validation": [10], "test": [80]}, horizon=24
        )


def test_daily_test_origins_are_92_midnights():
    timestamps = pd.date_range(
        "2011-01-01 01:00", "2014-12-31 23:00", freq="h"
    )

    origins = daily_backtest_origins(timestamps)

    assert len(origins) == 92
    assert origins[0] == pd.Timestamp("2014-10-01")
    assert origins[-1] == pd.Timestamp("2014-12-31")


def test_early_stopping_rejects_test_metrics():
    assert best_epoch_from_validation([0.8, 0.4, 0.5]) == 2
    with pytest.raises(ValueError, match="not test"):
        best_epoch_from_validation([0.8, 0.4], monitored_split="test")
