import numpy as np
import pandas as pd
import pytest

from src.data.calendars import calendar_features
from src.data.panel import Panel
from src.experiments.phase4 import build_experiments
from src.models.lightgbm_adapter import LightGBMDirectAdapter
from src.models.statsforecast_adapters import (
    auto_ets_adapter,
    naive_adapter,
    seasonal_naive_24_adapter,
    seasonal_naive_168_adapter,
)
from src.models.torch_forecasters import (
    DLinearAdapter,
    MLPAdapter,
    NLinearAdapter,
)


def small_panels():
    timestamps = pd.date_range("2014-01-01", periods=288, freq="h")
    time = np.arange(len(timestamps), dtype=np.float32)
    values = np.vstack(
        (
            4 + np.sin(time / 24) + time / 1000,
            8 + np.cos(time / 24) + time / 500,
        )
    ).astype(np.float32)
    train = Panel(("a", "b"), timestamps[:240], values[:, :240])
    validation = Panel(("a", "b"), timestamps[240:], values[:, 240:])
    return train, validation, values


def test_experiment_matrix_covers_required_history_and_normalization_variants(
    tmp_path,
):
    experiments = build_experiments(tmp_path)
    names = {name for name, _, _ in experiments}

    assert "seasonal_naive_24-h3" in names
    assert "seasonal_naive_168-h12" in names
    assert "lightgbm-h30" in names
    assert "dlinear-L168-per_series-h3" in names
    for model in ("mlp", "dlinear", "nlinear"):
        for lookback in (48, 168, 336):
            for normalization in ("none", "global", "per_series"):
                assert f"{model}-L{lookback}-{normalization}-h30" in names

    smoke_names = {
        name
        for name, _, _ in build_experiments(tmp_path, smoke=True)
    }
    assert len(smoke_names) == 8
    assert all(name.endswith("-smoke") for name in smoke_names)


@pytest.mark.parametrize(
    "adapter_type", [MLPAdapter, DLinearAdapter, NLinearAdapter]
)
def test_torch_adapters_train_and_predict_original_scale(adapter_type, tmp_path):
    train, validation, values = small_panels()
    adapter = adapter_type()
    metadata = adapter.fit(
        train,
        validation,
        {
            "lookback": 48,
            "horizon": 24,
            "normalization": "per_series",
            "max_epochs": 1,
            "patience": 1,
            "batch_size": 4,
            "max_train_windows_per_client": 2,
            "hidden_size": 16,
            "seed": 7,
            "artifact_dir": str(tmp_path),
            "config_id": "torch-smoke",
        },
    )
    history = Panel(
        train.client_ids,
        train.timestamps.append(validation.timestamps),
        values,
    )
    future_times = pd.date_range(
        history.timestamps[-1] + pd.Timedelta(1, unit="h"),
        periods=24,
        freq="h",
    )
    forecast = adapter.predict(
        history,
        calendar_features(future_times),
        horizon=24,
        origin=future_times[0],
    )

    assert forecast.shape == (2, 24)
    assert np.isfinite(forecast).all()
    assert metadata["best_epoch"] == 1
    training_log = pd.read_csv(metadata["training_log"])
    assert {"train_loss", "validation_loss", "train_mae_kwh", "train_mase"} <= set(
        training_log.columns
    )


@pytest.mark.parametrize(
    "factory",
    [
        naive_adapter,
        seasonal_naive_24_adapter,
        seasonal_naive_168_adapter,
        auto_ets_adapter,
    ],
)
def test_statsforecast_adapters_forward_new_history(factory):
    train, validation, values = small_panels()
    adapter = factory()
    metadata = adapter.fit(train, validation, {})
    history = Panel(
        train.client_ids,
        train.timestamps.append(validation.timestamps[:12]),
        values[:, :252],
    )
    future_times = pd.date_range(
        history.timestamps[-1] + pd.Timedelta(1, unit="h"),
        periods=24,
        freq="h",
    )
    forecast = adapter.predict(
        history,
        calendar_features(future_times),
        horizon=24,
        origin=future_times[0],
    )

    assert metadata["validation_used_for_fitting"] is False
    if adapter.model_name == "AutoETS":
        assert metadata["ets_model"] == "AAA"
    assert forecast.shape == (2, 24)
    assert np.isfinite(forecast).all()


def test_lightgbm_direct_uses_horizon_specific_calendar_and_lags(tmp_path):
    pytest.importorskip("lightgbm")
    train, validation, values = small_panels()
    adapter = LightGBMDirectAdapter()
    metadata = adapter.fit(
        train,
        validation,
        {
            "lookback": 168,
            "horizon": 24,
            "training_stride": 24,
            "validation_stride": 24,
            "max_train_windows_per_client": 2,
            "n_estimators": 3,
            "patience": 2,
            "min_child_samples": 1,
            "num_leaves": 5,
            "n_jobs": 1,
            "artifact_dir": str(tmp_path),
            "config_id": "lightgbm-smoke",
        },
    )
    history = Panel(
        train.client_ids,
        train.timestamps.append(validation.timestamps),
        values,
    )
    future_times = pd.date_range(
        history.timestamps[-1] + pd.Timedelta(1, unit="h"),
        periods=24,
        freq="h",
    )
    forecast = adapter.predict(
        history,
        calendar_features(future_times),
        horizon=24,
        origin=future_times[0],
    )

    assert metadata["strategy"] == "direct, one regressor per horizon step"
    assert metadata["lags"][-1] == 168
    assert adapter.parameter_counts()[0] > 0
    assert forecast.shape == (2, 24)
    assert np.isfinite(forecast).all()
    training_log = pd.read_csv(metadata["training_log"])
    assert {"l1", "l2", "mase"} <= set(training_log["metric"])
    model_files = list(
        (tmp_path / "lightgbm-direct" / "lightgbm-smoke").glob("model_h*.txt")
    )
    assert len(model_files) == 24
