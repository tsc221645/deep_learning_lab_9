import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate_predictions, seasonal_mase_scales


def test_point_metrics_match_known_values_and_exclude_zero_scales():
    predictions = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.date_range("2014-12-22", periods=4, freq="h"),
            "y_true": [1.0, 2.0, 0.0, 0.0],
            "y_hat": [2.0, 2.0, 0.0, 1.0],
            "mase_scale": [2.0, 2.0, 0.0, 0.0],
            "horizon_step": [1, 2, 1, 2],
        }
    )

    tables = evaluate_predictions(predictions)

    assert tables.overall["mae"] == pytest.approx(0.5)
    assert tables.overall["rmse"] == pytest.approx(np.sqrt(0.5))
    assert tables.overall["wape_pct"] == pytest.approx(200 / 3)
    assert tables.overall["mape_pct"] == pytest.approx(50)
    assert tables.overall["mape_excluded_count"] == 2
    assert tables.overall["mase"] == pytest.approx(0.25)
    assert tables.overall["mase_undefined_series_count"] == 1
    assert len(tables.christmas_week) == 1
    assert tables.christmas_week.iloc[0]["date"] == "2014-12-22"


def test_metrics_mark_undefined_ratios_without_epsilon_substitution():
    predictions = pd.DataFrame(
        {
            "unique_id": ["flat"],
            "ds": [pd.Timestamp("2014-10-01")],
            "y_true": [0.0],
            "y_hat": [0.0],
            "mase_scale": [0.0],
        }
    )

    metrics = evaluate_predictions(predictions).overall

    assert metrics["wape_pct"] is None
    assert metrics["mape_pct"] is None
    assert metrics["mase"] is None
    assert metrics["mase_undefined_series_count"] == 1


def test_mase_scale_uses_only_the_requested_training_interval():
    timestamps = pd.date_range("2014-01-01", periods=8 * 24, freq="h")
    values = np.vstack(
        [
            np.arange(len(timestamps), dtype=np.float64),
            np.ones(len(timestamps), dtype=np.float64),
        ]
    )
    values[0, 4 * 24 :] = 1_000_000

    scales = seasonal_mase_scales(
        values,
        timestamps,
        ("changing", "flat"),
        pd.Timestamp("2014-01-01"),
        pd.Timestamp("2014-01-05"),
        seasonality=24,
    )

    assert scales["mase_scale"].tolist() == pytest.approx([24.0, 0.0])
