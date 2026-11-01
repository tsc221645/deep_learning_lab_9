import numpy as np
import pandas as pd
import pytest
import torch

from src.data.panel import Panel
from src.evaluation.backtest import backtest, monthly_refit_backtest

HOUR = pd.Timedelta(1, unit="h")
TEST_SPLITS = {
    "train": (pd.Timestamp("2014-01-01"), pd.Timestamp("2014-08-01")),
    "validation": (pd.Timestamp("2014-08-01"), pd.Timestamp("2014-10-01")),
    "test": (pd.Timestamp("2014-10-01"), pd.Timestamp("2015-01-01")),
}


class LastValueModel:
    name = "last-value"

    def __init__(self):
        self.fit_calls = []
        self.origins = []

    def fit(self, train, val, config):
        self.fit_calls.append((train, val, config))
        return {"epochs": 3}

    def predict(self, history, future_calendar, horizon=24, origin=None):
        assert origin is not None
        assert history.timestamps[-1] == origin - HOUR
        assert future_calendar.index[0] == origin
        assert len(future_calendar) == horizon
        assert "y" not in future_calendar and "y_true" not in future_calendar
        self.origins.append(origin)
        return np.repeat(history.values[:, -1:], horizon, axis=1)


def small_panel():
    timestamps = pd.date_range("2014-01-01", "2015-01-01", freq="h", inclusive="left")
    values = np.vstack(
        [
            1 + np.arange(len(timestamps), dtype=np.float32) / 1000,
            2 + np.arange(len(timestamps), dtype=np.float32) / 500,
        ]
    )
    return Panel(("a", "b"), timestamps, values)


def test_backtest_predicts_92_by_24_for_each_client_without_future_inputs():
    model = LastValueModel()

    result = backtest(
        model,
        small_panel(),
        split_ranges=TEST_SPLITS,
        config={"lookback": 168, "horizon": 24, "seed": 7},
        output_dir=None,
    )

    assert len(model.fit_calls) == 1
    assert len(model.origins) == 92
    assert result.predictions.shape[0] == 92 * 24 * 2
    assert result.predictions["origin"].nunique() == 92
    assert result.predictions.groupby("origin").size().eq(48).all()
    assert result.predictions["ds"].min() == pd.Timestamp("2014-10-01")
    assert result.predictions["ds"].max() == pd.Timestamp("2014-12-31 23:00")
    assert result.predictions["horizon_step"].min() == 1
    assert result.predictions["horizon_step"].max() == 24
    assert len(result.by_horizon) == 24
    if torch.cuda.is_available():
        assert isinstance(result.experiment["gpu_peak_mb"], float)
        assert result.experiment["gpu_peak_mb"] >= 0
    else:
        assert result.experiment["gpu_peak_mb"] == "N/A"
    assert result.experiment["fit_count"] == 1


def test_fit_receives_additional_model_configuration():
    model = LastValueModel()
    backtest(
        model,
        small_panel(),
        split_ranges=TEST_SPLITS,
        origins=[pd.Timestamp("2014-10-01")],
        config={
            "learning_rate": 0.03,
            "max_epochs": 7,
            "artifact_dir": "custom-artifacts",
        },
        output_dir=None,
    )

    assert model.fit_calls[0][2]["learning_rate"] == 0.03
    assert model.fit_calls[0][2]["max_epochs"] == 7
    assert model.fit_calls[0][2]["artifact_dir"] == "custom-artifacts"


def test_backtest_rejects_bad_forecast_shape():
    class BadModel(LastValueModel):
        def predict(self, history, future_calendar, horizon=24, origin=None):
            return np.zeros((len(history.client_ids), horizon - 1))

    with pytest.raises(ValueError, match="expected"):
        backtest(
            BadModel(),
            small_panel(),
            split_ranges=TEST_SPLITS,
            origins=[pd.Timestamp("2014-10-01")],
            output_dir=None,
        )


def test_zero_shot_adapter_can_skip_fit_without_fake_training_time():
    class ZeroShot:
        name = "zero-shot"

        def predict(self, history, future_calendar, horizon=24, origin=None):
            return np.zeros((len(history.client_ids), horizon))

    result = backtest(
        ZeroShot(),
        small_panel(),
        split_ranges=TEST_SPLITS,
        origins=[pd.Timestamp("2014-10-01")],
        output_dir=None,
    )

    assert result.experiment["fit_count"] == 0
    assert result.experiment["train_time_s"] == 0.0
    assert result.metrics["n_observations"] == 48


def test_final_refit_uses_fixed_validation_epoch_on_train_plus_validation():
    model = LastValueModel()

    backtest(
        model,
        small_panel(),
        split_ranges=TEST_SPLITS,
        origins=[pd.Timestamp("2014-10-01")],
        config={
            "lookback": 168,
            "horizon": 24,
            "best_epoch": 3,
            "history_months": 3,
            "refit_train_validation": True,
        },
        output_dir=None,
    )

    train, validation, config = model.fit_calls[0]
    assert validation is None
    assert train.timestamps[0] == pd.Timestamp("2014-07-01")
    assert train.timestamps[-1] == pd.Timestamp("2014-09-30 23:00")
    assert config["best_epoch"] == 3


def test_final_refit_uses_fixed_validation_epoch_on_train_plus_validation():
    model = LastValueModel()

    backtest(
        model,
        small_panel(),
        split_ranges=TEST_SPLITS,
        origins=[pd.Timestamp("2014-10-01")],
        config={
            "lookback": 168,
            "horizon": 24,
            "best_epoch": 3,
            "history_months": 3,
            "refit_train_validation": True,
        },
        output_dir=None,
    )

    train, validation, config = model.fit_calls[0]
    assert validation is None
    assert train.timestamps[0] == pd.Timestamp("2014-07-01")
    assert train.timestamps[-1] == pd.Timestamp("2014-09-30 23:00")
    assert config["best_epoch"] == 3


def test_backtest_writes_verifiable_prediction_and_metric_artifacts(tmp_path):
    result = backtest(
        LastValueModel(),
        small_panel(),
        split_ranges=TEST_SPLITS,
        origins=[pd.Timestamp("2014-10-01")],
        output_dir=tmp_path,
    )

    artifacts = result.experiment["artifacts"]
    prediction_file = pd.read_parquet(artifacts["predictions"])
    assert len(prediction_file) == 2 * 24
    assert {"unique_id", "ds", "model", "y_hat", "origin", "y_true"} <= set(
        prediction_file.columns
    )
    experiment_file = (
        tmp_path
        / "predictions"
        / "last-value"
        / result.experiment["config_id"]
        / "experiment.json"
    )
    assert experiment_file.exists()


def test_monthly_refits_only_with_data_before_each_month():
    instances = []

    def create_model():
        model = LastValueModel()
        instances.append(model)
        return model

    result = monthly_refit_backtest(
        create_model,
        small_panel(),
        split_ranges=TEST_SPLITS,
        config={"lookback": 168, "horizon": 24, "best_epoch": 3,
                "history_months": 3},
        output_dir=None,
    )

    assert len(instances) == 3
    assert result.experiment["fit_count"] == 3
    assert len(result.experiment["fit_metadata"]["monthly_refits"]) == 3
    assert len(result.predictions) == 92 * 24 * 2
    for model, month in zip(instances, ("2014-10-01", "2014-11-01", "2014-12-01")):
        train, validation, config = model.fit_calls[0]
        assert validation is None
        assert train.timestamps[0] == pd.Timestamp("2014-01-01")
        assert train.timestamps[-1] < pd.Timestamp(month)
        assert config["final_epochs"] == 3
