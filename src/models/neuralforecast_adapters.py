"""Nixtla models behind the laboratory's shared backtest interface.

All normalization statistics come from ``train``. NeuralForecast's internal
scaler is disabled so an explicit validation frame cannot affect the scaler.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.calendars import calendar_features
from src.data.panel import Panel
from src.training.scaling import TemporalScaler


MODEL_NAMES = ("NHITS", "TCN", "PatchTST", "TFT", "DeepAR")
FUTURE_CALENDAR_MODELS = {"NHITS", "TCN", "TFT", "DeepAR"}


def _frame(panel: Panel, values: np.ndarray, use_calendar: bool,
           encoding: str) -> pd.DataFrame:
    n_clients, n_times = values.shape
    frame = pd.DataFrame({
        "unique_id": np.repeat(panel.client_ids, n_times),
        "ds": np.tile(panel.timestamps.to_numpy(), n_clients),
        "y": values.reshape(-1),
    })
    if use_calendar:
        calendar = calendar_features(panel.timestamps, encoding=encoding)
        for column in calendar:
            frame[column] = np.tile(calendar[column].to_numpy(), n_clients)
    return frame


class NeuralForecastAdapter:
    def __init__(self, model_name: str) -> None:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown NeuralForecast model: {model_name}")
        self.name = model_name
        self.forecaster: Any = None
        self.scaler: TemporalScaler | None = None
        self.quantiles_by_origin: dict[pd.Timestamp, np.ndarray] = {}

    def fit(self, train: Panel, val: Panel | None,
            config: Mapping[str, Any]) -> dict[str, Any]:
        from neuralforecast import NeuralForecast
        from neuralforecast import models

        self.client_ids = train.client_ids
        self.lookback = int(config.get("lookback", 168))
        self.horizon = int(config.get("horizon", 24))
        self.calendar_encoding = str(config.get("calendar_encoding", "cyclic"))
        if val is not None and val.client_ids != train.client_ids:
            raise ValueError("Validation clients differ from training clients")
        self.scaler = TemporalScaler(
            str(config.get("normalization", "per_series"))
        ).fit(
            train.values, train.timestamps, train.timestamps[0],
            train.timestamps[-1] + pd.Timedelta(hours=1),
        )
        uses_calendar = self.name in FUTURE_CALENDAR_MODELS
        train_frame = _frame(
            train, self.scaler.transform(train.values), uses_calendar,
            self.calendar_encoding,
        )
        val_frame = None
        if val is not None:
            val_frame = _frame(
                val, self.scaler.transform(val.values), uses_calendar,
                self.calendar_encoding,
            )
        calendar_columns = (
            list(calendar_features(train.timestamps[:1],
                                   encoding=self.calendar_encoding).columns)
            if uses_calendar else []
        )
        kwargs: dict[str, Any] = {
            "h": self.horizon,
            "input_size": self.lookback,
            "max_steps": int(config.get("max_steps", 1000)),
            "batch_size": int(config.get("batch_size", 32)),
            "valid_batch_size": int(config.get("valid_batch_size", 8)),
            "windows_batch_size": int(config.get("windows_batch_size", 64)),
            "inference_windows_batch_size": int(
                config.get("inference_windows_batch_size", 32)
            ),
            "learning_rate": float(config.get("learning_rate", 1e-3)),
            "random_seed": int(config.get("seed", 42)),
            "scaler_type": "identity",
            "alias": self.name,
            "accelerator": "gpu",
            "devices": 1,
            "enable_checkpointing": False,
            "early_stop_patience_steps": (
                int(config.get("early_stop_patience_steps", 5))
                if val is not None else -1
            ),
            "val_check_steps": int(config.get("val_check_steps", 100)),
        }
        if uses_calendar:
            kwargs["futr_exog_list"] = calendar_columns
        if self.name == "TFT":
            kwargs["hidden_size"] = int(config.get("hidden_size", 64))
        artifact_dir = config.get("artifact_dir")
        if artifact_dir is not None:
            from pytorch_lightning.loggers import CSVLogger

            logger = CSVLogger(
                save_dir=str(artifact_dir), name=self.name,
                version=str(config.get("config_id", "default")),
            )
            kwargs["logger"] = logger
        model = getattr(models, self.name)(**kwargs)
        self.forecaster = NeuralForecast(models=[model], freq="h")
        self.forecaster.fit(df=train_frame, val_df=val_frame)
        metadata = {
            "library": "neuralforecast",
            "model": self.name,
            "validation_used": val is not None,
            "scaler_fit_end_exclusive": self.scaler.fit_end_.isoformat(),
            "calendar_columns": calendar_columns,
            "batch_size": kwargs["batch_size"],
            "valid_batch_size": kwargs["valid_batch_size"],
            "windows_batch_size": kwargs["windows_batch_size"],
            "inference_windows_batch_size": kwargs["inference_windows_batch_size"],
        }
        if artifact_dir is not None:
            metadata["training_log"] = str(Path(logger.log_dir) / "metrics.csv")
        return metadata

    def predict(self, history: Panel, future_calendar: pd.DataFrame,
                horizon: int = 24,
                origin: pd.Timestamp | None = None) -> np.ndarray:
        if self.forecaster is None or self.scaler is None:
            raise RuntimeError("Call fit before predict")
        if history.client_ids != self.client_ids or horizon != self.horizon:
            raise ValueError("Panel clients or horizon differ from fitted model")
        if origin is None or history.timestamps[-1] != origin - pd.Timedelta(hours=1):
            raise ValueError("History must end one hour before origin")
        if not future_calendar.index.equals(
            pd.date_range(origin, periods=horizon, freq="h")
        ):
            raise ValueError("Future calendar timestamps are misaligned")
        expected = calendar_features(future_calendar.index,
                                     encoding=self.calendar_encoding)
        if not future_calendar.equals(expected):
            raise ValueError("Future calendar values are misaligned")
        context = Panel(
            history.client_ids, history.timestamps[-self.lookback:],
            history.values[:, -self.lookback:],
        )
        uses_calendar = self.name in FUTURE_CALENDAR_MODELS
        past = _frame(context, self.scaler.transform(context.values),
                      uses_calendar, self.calendar_encoding)
        future = None
        if uses_calendar:
            future = pd.DataFrame({
                "unique_id": np.repeat(self.client_ids, horizon),
                "ds": np.tile(future_calendar.index.to_numpy(), len(self.client_ids)),
            })
            for column in future_calendar:
                future[column] = np.tile(future_calendar[column].to_numpy(),
                                         len(self.client_ids))
        forecast = self.forecaster.predict(
            df=past, futr_df=future,
            **({"quantiles": [0.1, 0.5, 0.9]} if self.name == "DeepAR" else {}),
        )
        expected_index = pd.MultiIndex.from_product(
            (self.client_ids, future_calendar.index), names=("unique_id", "ds")
        )
        forecast = forecast.set_index(["unique_id", "ds"]).reindex(expected_index)
        if forecast.isna().any().any():
            raise ValueError("NeuralForecast returned incomplete predictions")
        if self.name == "DeepAR":
            columns = [f"{self.name}_ql{level}" for level in (0.1, 0.5, 0.9)]
            if not set(columns).issubset(forecast):
                raise ValueError(f"DeepAR quantile columns missing: {columns}")
            q = forecast[columns].to_numpy(dtype=np.float64).reshape(
                len(self.client_ids), horizon, 3
            )
            if (q[:, :, 0] > q[:, :, 2]).any():
                raise ValueError("DeepAR returned crossing P10/P90 quantiles")
            self.quantiles_by_origin[pd.Timestamp(origin)] = np.stack(
                [self.scaler.inverse_transform(q[:, :, i]) for i in range(3)],
                axis=2,
            )
            scaled = q[:, :, 1]
        else:
            scaled = forecast[self.name].to_numpy(dtype=np.float64).reshape(
                len(self.client_ids), horizon
            )
        return self.scaler.inverse_transform(scaled)

    def parameter_counts(self) -> tuple[int | None, int | None]:
        if self.forecaster is None:
            return None, None
        model = self.forecaster.models[0]
        params = list(model.parameters())
        return sum(p.numel() for p in params), sum(
            p.numel() for p in params if p.requires_grad
        )

    def probabilistic_predictions(self, point_predictions: pd.DataFrame) -> pd.DataFrame:
        if self.name != "DeepAR":
            raise ValueError("Only DeepAR exposes quantiles in this adapter")
        frames = []
        for origin, group in point_predictions.groupby("origin", sort=True):
            q = self.quantiles_by_origin[pd.Timestamp(origin)]
            times = pd.DatetimeIndex(group["ds"].unique()).sort_values()
            key = pd.MultiIndex.from_product(
                (self.client_ids, times), names=("unique_id", "ds")
            )
            ordered = group.set_index(["unique_id", "ds"]).loc[key].reset_index()
            if len(ordered) != q.shape[0] * q.shape[1]:
                raise ValueError("Quantile cache does not match point predictions")
            for index, level in enumerate((10, 50, 90)):
                ordered[f"p{level}"] = q[:, :, index].reshape(-1)
            frames.append(ordered)
        return pd.concat(frames, ignore_index=True)
