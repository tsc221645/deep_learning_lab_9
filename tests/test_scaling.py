import numpy as np
import pandas as pd

from src.training.scaling import (
    TemporalScaler,
    inverse_window_normalization,
    normalize_windows,
)


def test_temporal_scaler_fits_only_the_training_dates_and_inverts():
    timestamps = pd.date_range("2012-01-01", periods=4, freq="h")
    values = np.array([[1.0, 3.0, 1000.0, 2000.0]])
    scaler = TemporalScaler("global").fit(
        values,
        timestamps,
        train_start=timestamps[0],
        train_end=timestamps[2],
    )

    assert scaler.center_[0, 0] == 2.0
    scaled = scaler.transform(values[:, :2])
    np.testing.assert_allclose(scaler.inverse_transform(scaled), values[:, :2])


def test_per_series_scaler_and_window_scaler_restore_original_values():
    timestamps = pd.date_range("2012-01-01", periods=4, freq="h")
    values = np.array([[1.0, 3.0, 9.0, 11.0], [10.0, 14.0, 30.0, 40.0]])
    scaler = TemporalScaler("per_series").fit(
        values,
        timestamps,
        train_start=timestamps[0],
        train_end=timestamps[2],
    )
    transformed = scaler.transform(values)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), values)

    history = np.array([[1.0, 2.0, 3.0]])
    targets = np.array([[4.0, 5.0]])
    normalized_history, normalized_targets, center, scale = normalize_windows(
        history, targets
    )
    restored = inverse_window_normalization(normalized_targets, center, scale)
    np.testing.assert_allclose(restored, targets)
    assert normalized_history.shape == history.shape
