import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    _aggregate_available_hours,
    aggregate_quarter_hourly,
    choose_client_ids,
)


def test_aggregate_consecutive_quarters_to_hourly_kwh():
    timestamps = pd.date_range("2012-01-01 00:00", periods=8, freq="15min")
    values_kw = np.full((8, 2), 4.0)

    hourly_timestamps, hourly_values = aggregate_quarter_hourly(
        timestamps, values_kw
    )

    assert hourly_timestamps.equals(
        pd.DatetimeIndex(["2012-01-01 00:00", "2012-01-01 01:00"])
    )
    np.testing.assert_array_equal(hourly_values, np.full((2, 2), 4.0))


def test_aggregate_keeps_local_day_grid_and_source_zeros():
    timestamps = pd.date_range("2012-03-25 00:00", periods=96, freq="15min")
    values_kw = np.full((96, 1), 4.0)
    values_kw[4:8] = 0.0

    hourly_timestamps, hourly_values = aggregate_quarter_hourly(
        timestamps, values_kw
    )

    assert len(hourly_timestamps) == 24
    assert hourly_timestamps[1] == pd.Timestamp("2012-03-25 01:00")
    assert hourly_values[1, 0] == 0


def test_aggregate_rejects_irregular_or_incomplete_hours():
    timestamps = pd.date_range("2012-01-01 00:15", periods=8, freq="15min")
    values_kw = np.ones((8, 1))

    with pytest.raises(ValueError, match="exactly four readings"):
        aggregate_quarter_hourly(timestamps, values_kw)


def test_streaming_aggregation_discards_only_incomplete_boundaries():
    timestamps = pd.date_range("2011-01-01 00:15", periods=8, freq="15min")
    values_kw = np.full((8, 1), 4.0)

    hours, energy, pending_timestamps, pending_values = _aggregate_available_hours(
        timestamps, values_kw, allow_initial_partial=True
    )

    assert hours.equals(pd.DatetimeIndex(["2011-01-01 01:00"]))
    np.testing.assert_array_equal(energy, np.array([[4.0]]))
    assert pending_timestamps.equals(pd.DatetimeIndex(["2011-01-01 02:00"]))
    np.testing.assert_array_equal(pending_values, np.array([[4.0]]))


def test_client_selection_is_sorted_and_reproducible():
    clients = ["client_4", "client_1", "client_3", "client_2"]

    first = choose_client_ids(clients, limit=2, seed=42)
    second = choose_client_ids(clients, limit=2, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(first) == 2
